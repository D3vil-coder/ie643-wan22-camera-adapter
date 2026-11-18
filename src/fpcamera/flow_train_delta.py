"""Paired camera-velocity edit supervision around the frozen native Wan model.

Both motion commands receive the same noisy STATIC input. The target is the
frozen teacher prediction plus (static_clean - requested_clean) / sigma.
This experimental objective prevents future camera motion in the noisy input
from explaining away the requested control. The original uniform trainer is
preserved in flow_train.py; no old checkpoint is an exact resume of this recipe.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from safetensors.torch import load_file,save_file

from .flow_cache import load_cached,NORMALIZATION
from .flow_control import FlowControlConfig,FlowControlLoRA,build_image_condition
from .runtime import file_sha256,write_json,gpu_lease
from .wan import (MODEL,REVISION,configure_library_output,require_pinned_runtime,
                  hash_module,pad_text_embeddings)


# # Target formula and image-conditioning timestep mask.
def camera_velocity_target(teacher_future, static_future, requested_future, sigma):
    """Teacher-relative camera edit; all inputs are future latents only.

    If teacher=(x-static)/sigma, this is exactly (x-requested)/sigma.
    Sigma is bounded away from zero by the declared training distribution.
    """
    if not (teacher_future.shape==static_future.shape==requested_future.shape):
        raise ValueError('Teacher/static/requested future latent shapes disagree')
    scale=sigma.float().reshape(-1,1,1,1,1)
    if not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Positive finite noise levels required')
    delta=(static_future.detach().float()-requested_future.detach().float())/scale
    return teacher_future.detach().float()+delta,delta


def token_timestep(clean,sigma):
    """TI2V image-conditioning time is zero for the complete first latent plane."""
    b,_,t,h,w=clean.shape
    mask=torch.ones(b,t,h//2,w//2,device=clean.device)
    mask[:,0]=0
    return mask.flatten(1)*sigma[:,None]*1000


# # Frozen Wan backbone loading and cached tensor assembly.
def _load_transformer(log):
    from diffusers import WanTransformer3DModel
    configure_library_output(log);versions=require_pinned_runtime()
    model=WanTransformer3DModel.from_pretrained(MODEL,subfolder='transformer',revision=REVISION,
        torch_dtype=torch.bfloat16,local_files_only=True)
    model.requires_grad_(False);model.eval();model.to('cuda')
    if model.config.in_channels!=48 or tuple(model.config.patch_size)!=(1,2,2):
        raise RuntimeError('Unexpected TI2V-5B architecture')
    return model,versions


def _input(item,device='cuda'):
    clean=item['latents'][None].to(device=device,dtype=torch.bfloat16)
    first=item['image_latent'][None].to(device=device,dtype=torch.bfloat16)
    text=pad_text_embeddings(item['text_embeddings'])[None].to(device=device,dtype=torch.bfloat16)
    contract=item['contract']
    condition=build_image_condition(first,item['homographies'][None],width=contract.width,
        height=contract.height,frames=contract.frames)
    return clean,first,text,condition


# # Atomic checkpoint and exact optimizer/RNG resume-state persistence.
def _save(adapter,optimizer,out,step,ledger,rng):
    path=out/f'adapter_{step:06d}.safetensors'
    state={k:v.cpu().float().contiguous() for k,v in adapter.adapter_state_dict().items()}
    temporary=path.with_suffix('.safetensors.partial')
    save_file(state,str(temporary),metadata={'architecture_json':json.dumps(adapter.architecture()),'step':str(step)})
    os.replace(temporary,path)
    meta={**ledger,'completed_updates':step,'checkpoint':path.name,'checkpoint_sha256':file_sha256(path),
          'architecture':adapter.architecture(),'status':'trained_checkpoint_requires_generated_motion_evaluation'}
    resume={'step':step,'checkpoint_sha256':meta['checkpoint_sha256'],'identity':ledger['identity'],
        'optimizer':optimizer.state_dict(),'python_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all(),'consumed_rows':ledger['consumed_rows']}
    temp=out/'resume.pt.partial';torch.save(resume,temp);os.replace(temp,out/'resume.pt')
    meta['resume_sha256']=file_sha256(out/'resume.pt')
    write_json(path.with_suffix('.json'),meta);write_json(out/'latest.json',meta)
    return meta


# # Paired-control training loop, validation gates, and provenance ledger.
def train(config,progress=None):
    """v1 trainer (Submission 1): fit the teacher-relative 'delta' objective on the v1 cache for the configured
    steps and write checkpoints under ``artifacts/flow_poc_all``.  Kept for provenance; v2 uses ``flow_train_v2``.
    """
    if not torch.cuda.is_available():raise RuntimeError('Molab CUDA GPU required')
    root=Path(config['project_root']);out=Path(config['output_dir'])
    allowed={root/'artifacts/flow_poc_all'}
    if out not in allowed:raise ValueError('Unexpected flow output directory')
    out.mkdir(parents=True,exist_ok=True)
    index=Path(config['index']);index_sha=file_sha256(index)
    rows=[json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    train_rows=[r for r in rows if r['split']=='train']
    if not train_rows:raise ValueError('No training clips')
    if len({r['normalization'] for r in train_rows})!=1 or train_rows[0]['normalization']!=NORMALIZATION:
        raise ValueError('Mixed or wrong VAE cache normalization')
    contracts={json.dumps(r['video_contract'],sort_keys=True) for r in train_rows}
    if len(contracts)!=1:raise ValueError('Mixed video contracts require an explicit new stage')
    cached_contract=json.loads(next(iter(contracts)))
    if any(cached_contract.get(k)!=v for k,v in config.get('video_contract',{}).items()):
        raise ValueError('Declared training video contract differs from its caches')
    groups={r['source_id'] for r in train_rows}
    by_mode={m:[r for r in train_rows if r['mode']==m] for m in sorted({r['mode'] for r in train_rows})}
    moving=[m for m in by_mode if m!='static']
    if not moving:raise ValueError('At least one moving control is required')
    moving=list(config.get('camera_modes',moving))
    if len(set(moving))!=len(moving) or len(moving)<2 or any(m not in by_mode or m=='static' for m in moving):
        raise ValueError('At least two distinct cached moving controls are required')
    source_rows={source:{} for source in sorted(groups)}
    for row in train_rows:
        if row['mode'] in source_rows[row['source_id']]:raise ValueError('Duplicate source/mode cache row')
        source_rows[row['source_id']][row['mode']]=row
    if any(any(m not in rows_for_source for m in ('static',*moving)) for rows_for_source in source_rows.values()):
        raise ValueError('Every paired source needs a static target and all requested motion targets')
    for rows_for_source in source_rows.values():
        if len({rows_for_source[m]['caption'] for m in ('static',*moving)})!=1:
            raise ValueError('Paired motion labels must use the same scene caption')
    source_order=sorted(source_rows)
    noise_cfg=config.get('noise_sampling',{'distribution':'truncated_normal','mean':.8,'std':.075,'lower':.6,'upper':1.})
    if (noise_cfg.get('distribution')!='truncated_normal' or
            not 0<float(noise_cfg['lower'])<=float(noise_cfg['mean'])<=float(noise_cfg['upper'])<=1 or
            not math.isfinite(float(noise_cfg['std'])) or float(noise_cfg['std'])<=0):
        raise ValueError('Invalid bounded camera noise distribution')
    seed=int(config.get('seed',20261025));rng=random.Random(seed)
    torch.manual_seed(seed);np.random.seed(seed)
    cfg=FlowControlConfig(**config.get('architecture',{}))
    total=int(config.get('updates',5000));stop=min(total,int(config.get('stop_after_updates',total)))
    effective=int(config.get('effective_batch',4))
    if int(config.get('microbatch',1))!=1:
        raise ValueError('First verified flow trainer uses microbatch1; benchmark larger batches separately')
    if total<1 or effective<2 or effective%2:raise ValueError('Effective batch must contain complete two-command pairs')
    lr=float(config.get('learning_rate',1e-4));warmup=int(config.get('warmup',100))
    hashes={name:file_sha256(Path(__file__).with_name(name)) for name in
            ('flow_train_delta.py','flow_control.py','flow_cache.py','flow_geometry.py','wan.py','control_lora.py')}
    identity={'model':MODEL,'revision':REVISION,'index_sha256':index_sha,'architecture':asdict(cfg),
        'video_contract':json.loads(next(iter(contracts))),'effective_batch':effective,'updates':total,
        'learning_rate':lr,'warmup':warmup,'seed':seed,'source_hashes':hashes,
        'objective':'teacher_relative_camera_velocity_on_shared_static_noisy_input',
        'target_formula':'frozen_prediction + (static_clean - requested_clean) / sigma',
        'noise_sampling':noise_cfg,'camera_modes':moving,'paired_motion_commands':2,
        'weight_decay':.01,'text_dropout':.1,'static_pair_probability':.1}
    stop_requested=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop_requested.__setitem__(0,True))
    ledger={'purpose':'paired_teacher_relative_camera_control_I2V_adapter','identity':identity,
        'available_encoded_rows':len(train_rows),'available_source_groups':len(groups),
        'mode_counts':dict(Counter(r['mode'] for r in train_rows)),'consumed_rows':{},
        'architecture':None,'quality_validation':'pending','original_backbone_frozen':True,
        'VAE_and_text_encoder_frozen_during_cache_encoding':True,'history':[],
        'baseline_teacher':'same frozen transformer with all adapter components disabled; no second model copy',
        'noisy_future_input':'same-source static clip; requested future motion is never mixed into model input',
        'inference_policy':{'camera_min_sigma':float(noise_cfg['lower']),'camera_guidance':1.,
            'description':'frozen native denoiser below the trained camera-noise window; no RGB post-warp'},
        'experimental_objective_requires_generated_video_evaluation':True}
    started=time.monotonic();last_checkpoint=started;step=0;inside_update=False
    with gpu_lease(root):
        model,versions=_load_transformer(out/'library.log')
        original=tuple(model.parameters());before=hash_module(model,parameters=original)
        expected=config.get('expected_backbone_sha256')
        if expected and expected!=before:raise RuntimeError('Frozen backbone differs from the verified baseline')
        ledger.update(backbone_sha256_before=before,versions=versions)
        probe=load_cached(train_rows[0],root)
        pc,pf,pt,pcondition=_input(probe)
        psigma=torch.tensor([.8],device='cuda')
        pnoise=torch.randn(pc.shape,device='cuda',dtype=pc.dtype,
            generator=torch.Generator(device='cuda').manual_seed(seed+177))
        px=torch.cat((pf,((1-psigma[:,None,None,None,None])*pc+psigma[:,None,None,None,None]*pnoise)[:,:,1:]),dim=2)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            frozen_prediction=model(hidden_states=px,timestep=token_timestep(pc,psigma),
                encoder_hidden_states=pt,return_dict=False)[0]
        adapter=FlowControlLoRA(model,cfg)
        adapter.set_condition(pcondition)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            zero_prediction=model(hidden_states=px,timestep=token_timestep(pc,psigma),
                encoder_hidden_states=pt,return_dict=False)[0]
        ledger['zero_init_max_abs']=float((frozen_prediction-zero_prediction).abs().max())
        if ledger['zero_init_max_abs']!=0:raise RuntimeError('Initial adapter changed the actual frozen Wan prediction')
        del probe,pc,pf,pt,pcondition,pnoise,px,frozen_prediction,zero_prediction
        ledger['architecture']=adapter.architecture()
        parameters=[p for p in adapter.parameters() if p.requires_grad]
        if {id(p) for p in parameters}&{id(p) for p in original}:raise RuntimeError('Original parameters entered optimizer')
        ledger['adapter_parameters']=sum(p.numel() for p in parameters)
        optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.01,fused=True)
        if config.get('checkpointing','full')=='full':model.enable_gradient_checkpointing()
        elif config.get('checkpointing')!='none':raise ValueError('Expected full or no activation checkpointing')
        model.train()
        resume=config.get('resume')
        if resume:
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            if saved['identity']!=identity:raise ValueError('Exact-resume configuration or source changed')
            latest=json.loads((out/'latest.json').read_text())
            checkpoint=out/latest['checkpoint']
            if file_sha256(checkpoint)!=saved['checkpoint_sha256'] or latest['resume_sha256']!=file_sha256(resume):
                raise ValueError('Resume checkpoint/state integrity mismatch')
            adapter.load_adapter_state_dict(load_file(str(checkpoint),device='cpu'))
            optimizer.load_state_dict(saved['optimizer']);rng.setstate(saved['python_rng'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            step=int(saved['step']);ledger['consumed_rows']=saved['consumed_rows']
            ledger['history']=latest.get('history',[]);ledger['resumed_from']=latest['checkpoint_sha256']
            if latest.get('backbone_sha256_before')!=before:raise ValueError('Resume backbone changed')
        elif config.get('initial_adapter'):
            initial=Path(config['initial_adapter']);meta=json.loads(initial.with_suffix('.json').read_text())
            if meta['architecture']!=adapter.architecture() or meta.get('backbone_unchanged') is not True:
                raise ValueError('Warm-start lacks the same flow architecture/unchanged backbone audit')
            if meta.get('backbone_sha256_after')!=before or file_sha256(initial)!=meta['checkpoint_sha256']:
                raise ValueError('Warm-start backbone or adapter hash differs')
            adapter.load_adapter_state_dict(load_file(str(initial),device='cpu'))
            ledger['warm_start']={'checkpoint_sha256':meta['checkpoint_sha256'],
                'previous_updates':meta['completed_updates'],'optimizer_reused':False}
        elif list(out.glob('adapter_*.safetensors')):
            raise FileExistsError('Existing flow checkpoints require explicit resume')
        write_json(out/'config.json',config)
        consumed=Counter(ledger['consumed_rows'])
        torch.cuda.reset_peak_memory_stats()
        try:
            while step<stop and not stop_requested[0]:
                update_started=time.monotonic()
                optimizer.zero_grad(set_to_none=True);inside_update=True;loss_sum=0.;sampled=[];sample_details=[]
                current_lr=lr*min(1.,(step+1)/max(1,warmup))*(.1+.9*.5*(1+math.cos(math.pi*step/total)))
                for group in optimizer.param_groups:group['lr']=current_lr
                pair_details=[]
                for pair_number in range(effective//2):
                    source=rng.choice(source_order);paired_rows=source_rows[source]
                    static_row=paired_rows['static'];static_item=load_cached(static_row,root)
                    static,first,text,static_condition=_input(static_item)
                    sigma=torch.empty(1,device='cuda')
                    torch.nn.init.trunc_normal_(sigma,mean=float(noise_cfg['mean']),std=float(noise_cfg['std']),
                        a=float(noise_cfg['lower']),b=float(noise_cfg['upper']))
                    noise=torch.randn_like(static)
                    noisy=(1-sigma[:,None,None,None,None])*static+sigma[:,None,None,None,None]*noise
                    noisy=torch.cat((first,noisy[:,:,1:]),dim=2)
                    timesteps=token_timestep(static,sigma)
                    if rng.random()<.1:
                        text=pad_text_embeddings(static_item['negative_text_embeddings'])[None].to(device='cuda',dtype=torch.bfloat16)
                    adapter.set_condition(static_condition)
                    with torch.no_grad(),adapter.disabled(),torch.autocast('cuda',dtype=torch.bfloat16):
                        teacher_prediction=model(hidden_states=noisy,timestep=timesteps,
                            encoder_hidden_states=text,return_dict=False)[0]
                    teacher_future=teacher_prediction[:,:,1:].detach().float()
                    del teacher_prediction
                    modes=['static','static'] if rng.random()<.1 else rng.sample(moving,2)
                    detached_predictions=[];detached_targets=[]
                    for mode in modes:
                        row=paired_rows[mode]
                        if mode=='static':
                            requested=static;condition=static_condition
                        else:
                            item=load_cached(row,root)
                            requested=item['latents'][None].to(device='cuda',dtype=torch.bfloat16)
                            if not torch.equal(requested[:,:,:1],first):raise ValueError('Paired image anchor differs')
                            condition=build_image_condition(first,item['homographies'][None],width=cached_contract['width'],
                                height=cached_contract['height'],frames=cached_contract['frames'])
                            del item
                        target,delta=camera_velocity_target(teacher_future,static[:,:,1:],requested[:,:,1:],sigma)
                        adapter.set_condition(condition)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            prediction=model(hidden_states=noisy,timestep=timesteps,
                                encoder_hidden_states=text,return_dict=False)[0]
                            loss=(prediction[:,:,1:].float()-target).square().mean()
                        if not torch.isfinite(loss):raise RuntimeError('Non-finite teacher-relative camera loss')
                        (loss/effective).backward();loss_sum+=float(loss.detach())/effective
                        predicted_residual=prediction[:,:,1:].detach().float()-teacher_future
                        energy=delta.square().sum()
                        detail={'record_id':row['record_id'],'source_id':source,'mode':mode,'sigma':float(sigma),
                            'static_input_record_id':static_row['record_id'],'pair_number':pair_number,
                            'camera_edit_loss':float(loss.detach()),'target_edit_rms':float(delta.square().mean().sqrt()),
                            'predicted_edit_rms':float(predicted_residual.square().mean().sqrt()),
                            'edit_target_projection':float((predicted_residual*delta).sum()/energy) if energy>0 else None}
                        sample_details.append(detail);sampled.append(row['record_id'])
                        detached_predictions.append(prediction[:,:,1:].detach())
                        detached_targets.append(delta.detach())
                        del requested,condition,target,delta,prediction,loss,predicted_residual,energy
                    separation=detached_predictions[0].float()-detached_predictions[1].float()
                    desired_separation=detached_targets[0]-detached_targets[1]
                    desired_energy=desired_separation.square().sum()
                    pair_details.append({'source_id':source,'modes':modes,'sigma':float(sigma),
                        'prediction_separation_rms':float(separation.square().mean().sqrt()),
                        'desired_separation_rms':float(desired_separation.square().mean().sqrt()),
                        'separation_target_projection':float((separation*desired_separation).sum()/desired_energy)
                            if desired_energy>0 else None})
                    del static_item,static,first,text,static_condition,sigma,noise,noisy,timesteps,teacher_future
                    del detached_predictions,detached_targets,separation,desired_separation,desired_energy
                if any(p.grad is not None for p in original):raise RuntimeError('Frozen original acquired a gradient')
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not torch.isfinite(norm):raise RuntimeError('Non-finite adapter gradient')
                if step<2:
                    ledger.setdefault('gradient_audit',[]).append({'update':step+1,
                        'nonzero':[name for name,p in adapter.named_parameters()
                            if p.grad is not None and torch.count_nonzero(p.grad).item()>0],
                        'missing':[name for name,p in adapter.named_parameters() if p.grad is None]})
                optimizer.step();step+=1;consumed.update(sampled);inside_update=False
                record={'step':step,'loss':loss_sum,'gradient_norm':float(norm),
                    'seconds':time.monotonic()-started,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
                    'update_seconds':time.monotonic()-update_started,
                    'learning_rate':current_lr,'sources_available':len(groups),'samples':sample_details,
                    'paired_command_diagnostics':pair_details,'noisy_input_contains_requested_future_motion':False}
                ledger['history'].append(record);ledger['consumed_rows']=dict(consumed)
                write_json(out/'progress.json',record)
                if progress:progress(record)
                if record['peak_gib']>float(config.get('max_peak_gib',80)):
                    stop_requested[0]=True;ledger['stopped_for_memory_budget']=True
                if step%int(config.get('checkpoint_updates',250))==0 or time.monotonic()-last_checkpoint>=600:
                    _save(adapter,optimizer,out,step,ledger,rng);last_checkpoint=time.monotonic()
        except BaseException as error:
            ledger['failure']={'type':type(error).__name__,'message':str(error)}
            raise
        finally:
            after=hash_module(model,parameters=original)
            ledger.update(backbone_sha256_after=after,backbone_unchanged=before==after,
                consumed_rows=dict(consumed),training_seconds=time.monotonic()-started)
            if inside_update:
                write_json(out/'interruption.json',{**ledger,'resume_status':'previous_complete_optimizer_boundary_retained'})
            else:_save(adapter,optimizer,out,step,ledger,rng)
            if after!=before:raise RuntimeError('Original frozen backbone changed')
        return ledger


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    train(json.loads(args.config.read_text()),progress=lambda record:print(json.dumps(record),flush=True))
