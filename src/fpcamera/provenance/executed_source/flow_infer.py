"""Generate from one image with the shared trained flow Control-LoRA."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageOps
import torch
from safetensors.torch import load_file

from .flow_geometry import VideoContract,camera_path
from .flow_control import FlowControlConfig,FlowControlLoRA,build_image_condition
from .flow_guidance import camera_prediction_guidance
from .flow_anchor import frozen_camera_difference
from .runtime import gpu_lease,file_sha256,write_json
from .wan import load_pipeline,hash_module,encode_first_image


def _verify_video(path,contract):
    import cv2
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame.shape!=(contract.height,contract.width,3):raise RuntimeError('Wrong decoded dimensions')
        count+=1
    cap.release()
    if count!=contract.frames or abs(fps-contract.fps)>.01:raise RuntimeError('Video decode contract mismatch')
    return {'decoded_frames':count,'fps':fps,'width':contract.width,'height':contract.height,
            'sha256':file_sha256(path),'bytes':path.stat().st_size}


def generate(root,checkpoint,image_path,output,*,mode='zoom_in',prompt=None,seed=42,
             steps=50,guidance=5.,flow_shift=5.,magnitude=1.,frozen_baseline=False,
             camera_request=None,camera_guidance=1.,component='full',camera_min_sigma=None,
             camera_residual_anchor='auto'):
    import imageio.v2 as imageio
    root,checkpoint,output=Path(root),Path(checkpoint),Path(output)
    import math
    if not math.isfinite(float(camera_guidance)) or float(camera_guidance)<0:
        raise ValueError('Camera guidance must be finite and nonnegative')
    if camera_min_sigma is not None and (not math.isfinite(float(camera_min_sigma)) or not 0<=float(camera_min_sigma)<=1):
        raise ValueError('Minimum camera sigma must be finite and in [0,1]')
    if frozen_baseline and float(camera_guidance)!=1.:
        raise ValueError('Frozen baseline must use the native camera-guidance passthrough')
    if component not in ('full','guide_only','attention_only','lora_only'):
        raise ValueError('Unknown camera-adapter component ablation')
    if frozen_baseline and component!='full':
        raise ValueError('Frozen baseline and partial-adapter ablations are distinct experiments')
    if frozen_baseline and camera_min_sigma is not None and float(camera_min_sigma)!=0.:
        raise ValueError('Frozen baseline does not use a camera activation schedule')
    if (output/'run.json').exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads(checkpoint.with_suffix('.json').read_text())
    if camera_residual_anchor=='auto':
        camera_residual_anchor=('frozen' if not frozen_baseline and meta.get('identity',{}).get('objective')==
            'teacher_relative_camera_velocity_on_shared_static_noisy_input' else 'none')
    if camera_residual_anchor not in ('none','frozen','teacher'):
        raise ValueError('Unknown camera residual anchor')
    if frozen_baseline and camera_residual_anchor!='none':raise ValueError('Frozen baseline has no learned residual anchor')
    if camera_min_sigma is None:
        camera_min_sigma=0. if frozen_baseline else float(meta.get('inference_policy',{}).get('camera_min_sigma',0.))
    if not math.isfinite(float(camera_min_sigma)) or not 0<=float(camera_min_sigma)<=1:
        raise ValueError('Checkpoint camera activation policy is invalid')
    if file_sha256(checkpoint)!=meta['checkpoint_sha256'] or meta.get('backbone_unchanged') is not True:
        raise ValueError('Checkpoint lacks matching digest and unchanged-backbone audit')
    if meta['architecture']['kind']!='wan_flow_control_lora':raise ValueError('Wrong adapter family')
    contract=VideoContract(**meta['identity']['video_contract'])
    image=ImageOps.fit(Image.open(image_path).convert('RGB'),(contract.width,contract.height),
        method=Image.Resampling.LANCZOS)
    image.save(output/'input.png')
    if camera_request is None:
        path,request=camera_path(mode,contract,magnitude=magnitude)
    else:
        request=json.loads(Path(camera_request).read_text()) if isinstance(camera_request,(str,Path)) else dict(camera_request)
        path=np.asarray(request['homographies_first_to_current'],dtype=np.float64)
        if (path.shape!=(contract.frames,3,3) or not np.isfinite(path).all()
            or not np.allclose(path[0],np.eye(3),atol=1e-6)
            or np.any(np.abs(np.linalg.det(path))<1e-8)):
            raise ValueError('Invalid custom camera homographies')
        supplied=request.get('contract',{})
        if any(supplied.get(k)!=getattr(contract,k) for k in ('width','height','frames','fps')):
            raise ValueError('Custom camera path and trained video contract disagree')
        mode='drawn_camera_view_path'
    write_json(output/'request.json',request)
    prompt=prompt if prompt is not None else 'A photograph of a scene. One continuous camera shot.'
    with gpu_lease(root):
        pipe=load_pipeline(flow_shift=flow_shift,log_path=output/'library.log',local_files_only=True)
        if float(camera_min_sigma)>0. and (pipe.scheduler.config.prediction_type!='flow_prediction'
                or not pipe.scheduler.config.use_flow_sigmas or pipe.scheduler.config.num_train_timesteps!=1000):
            raise ValueError('Camera sigma gating requires the verified native1000-step flow scheduler')
        original=tuple(pipe.transformer.parameters());before=hash_module(pipe.transformer,parameters=original)
        if before!=meta['backbone_sha256_after']:raise ValueError('Inference base differs from training')
        first=encode_first_image(pipe,np.asarray(image))[None].to('cuda')
        condition=build_image_condition(first,path,width=contract.width,height=contract.height,frames=contract.frames)
        adapter=FlowControlLoRA(pipe.transformer,FlowControlConfig(**meta['architecture']['config']))
        adapter.load_adapter_state_dict(load_file(str(checkpoint),device='cpu'))
        adapter.set_condition(condition)
        if component=='guide_only':
            for wrapper in adapter._wrappers.values():wrapper.enabled=False
        elif component=='attention_only':
            for path_name,wrapper in adapter._wrappers.items():
                wrapper.enabled='.attn1.' in path_name
        elif component=='lora_only':
            adapter.enabled=False
        active_adapter_parameters=(sum(p.numel() for p in adapter.guide_projection.parameters()) if adapter.enabled else 0)
        active_adapter_parameters+=sum(w.lora_A.numel()+w.lora_B.numel()
            for w in adapter._wrappers.values() if w.enabled)
        static_condition=None
        if camera_residual_anchor=='frozen' or (camera_residual_anchor=='none' and float(camera_guidance)!=1.):
            static_path,_=camera_path('static',contract)
            static_condition=build_image_condition(first,static_path,width=contract.width,
                height=contract.height,frames=contract.frames)
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        camera_schedule=[]
        guidance_stats=None
        def callback(_pipe,step,timestep,values):
            if camera_residual_anchor in ('frozen','teacher') and guidance_stats['native_calls']!=(step+1)*guidance_stats['native_calls_per_step']:
                raise RuntimeError('Native CFG call count disagrees with frozen camera anchoring protocol')
            noise_level=float(timestep)/1000.
            active=not frozen_baseline and noise_level>=float(camera_min_sigma)
            camera_schedule.append({'step':step+1,'sigma':noise_level,'adapter_active':active})
            if step%10==0 or step==steps-1:
                write_json(output/'progress.json',{'step':step+1,'total':steps,'seconds':time.perf_counter()-started,
                    'sigma':noise_level,'camera_adapter_active':active})
            return values
        try:
            camera_context=(frozen_camera_difference(pipe.transformer,adapter,condition,static_condition,
                camera_gain=camera_guidance,text_guidance=guidance,active_min_sigma=camera_min_sigma,
                reference='teacher' if camera_residual_anchor=='teacher' else 'static')
                if camera_residual_anchor in ('frozen','teacher') else
                camera_prediction_guidance(pipe.transformer,adapter,condition,static_condition,camera_guidance,camera_min_sigma))
            with (adapter.disabled() if frozen_baseline else nullcontext()),torch.inference_mode(), \
                    camera_context as guidance_stats:
                result=pipe(image=image,prompt=prompt,negative_prompt='',height=contract.height,width=contract.width,
                    num_frames=contract.frames,num_inference_steps=steps,guidance_scale=guidance,
                    generator=torch.Generator(device='cuda').manual_seed(seed),output_type='np',
                    callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents'])
            pixels=np.asarray(result.frames[0])
            if not np.isfinite(pixels).all():raise RuntimeError('Nonfinite generated video')
            if pixels.dtype!=np.uint8:pixels=np.uint8(np.rint(np.clip(pixels,0,1)*255))
            imageio.mimwrite(output/'video.mp4',pixels,fps=contract.fps,codec='libx264',quality=8,
                ffmpeg_params=['-movflags','+faststart'])
            video=_verify_video(output/'video.mp4',contract)
            for index in sorted({0,contract.frames//3,2*contract.frames//3,contract.frames-1}):
                Image.fromarray(pixels[index]).save(output/f'frame_{index:03d}.png')
        finally:
            adapter.close()
        after=hash_module(pipe.transformer,parameters=original)
        if before!=after:raise RuntimeError('Frozen original weights changed during inference')
        record={'status':'generated_video_requires_motion_and_preservation_evaluation',
            'adapter_enabled':not frozen_baseline,'checkpoint_sha256':meta['checkpoint_sha256'],
            'adapter_component':component if not frozen_baseline else 'disabled',
            'active_adapter_parameters':active_adapter_parameters if not frozen_baseline else 0,
            'completed_updates':meta['completed_updates'],'contract':contract.record(),
            'mode':mode,'magnitude':magnitude,'prompt':prompt,'negative_prompt':'','seed':seed,
            'steps':steps,'guidance':guidance,'flow_shift':flow_shift,
            'camera_guidance':float(camera_guidance),
            'camera_residual_anchor':camera_residual_anchor,'camera_guidance_execution':guidance_stats,
            'camera_min_sigma':float(camera_min_sigma),'camera_sigma_schedule':camera_schedule,
            'camera_active_denoising_steps':sum(r['adapter_active'] for r in camera_schedule),
            'camera_guidance_reference':('frozen_positive_prediction' if camera_residual_anchor=='teacher' else
                ('same_image_static_view' if camera_residual_anchor=='frozen'
                or float(camera_guidance)!=1. else 'native_passthrough')),
            'seconds':time.perf_counter()-started,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
            'video':video,'backbone_unchanged':True,'backbone_sha256':after,
            'input_sha256':file_sha256(output/'input.png'),'wan':pipe._fpcamera_provenance}
        record['inference_source_sha256']={name:file_sha256(Path(__file__).with_name(name))
            for name in ('flow_infer.py','flow_guidance.py','flow_anchor.py','flow_control.py','flow_geometry.py','wan.py')}
        write_json(output/'run.json',record)
        return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('project-root','checkpoint','image','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',default='zoom_in');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--magnitude',type=float,default=1.)
    parser.add_argument('--prompt',default=None)
    parser.add_argument('--camera-request',type=Path,default=None)
    parser.add_argument('--steps',type=int,default=50);parser.add_argument('--frozen-baseline',action='store_true')
    parser.add_argument('--camera-guidance',type=float,default=1.)
    parser.add_argument('--camera-residual-anchor',choices=('auto','none','frozen','teacher'),default='auto')
    parser.add_argument('--camera-min-sigma',type=float,default=None,
        help='Override checkpoint camera-noise window; legacy checkpoints default to all steps')
    parser.add_argument('--component',choices=('full','guide_only','attention_only','lora_only'),default='full')
    args=parser.parse_args()
    result=generate(args.project_root,args.checkpoint,args.image,args.output,mode=args.mode,seed=args.seed,
        steps=args.steps,frozen_baseline=args.frozen_baseline,magnitude=args.magnitude,
        prompt=args.prompt,camera_request=args.camera_request,camera_guidance=args.camera_guidance,
        component=args.component,camera_min_sigma=args.camera_min_sigma,camera_residual_anchor=args.camera_residual_anchor)
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],'backbone_unchanged':result['backbone_unchanged']}),flush=True)
