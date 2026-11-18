# Reproducibility and provenance

## Frozen identity

`artifacts/flow_poc_all/latest.json` is the checkpoint sidecar. It binds the adapter checkpoint SHA-256, `resume.pt` SHA-256, exact cache index SHA-256, model name/revision, video contract, optimizer recipe, training history and six critical training source hashes. The recorded backbone hash before and after training is identical. The checkpoint parameter count is 14,261,248.

`configs/provenance/flow_delta_zoom_pilot.json` preserves the original experiment config. `artifacts/flow_poc_all/config.json` is the config saved by the actual run. `configs/train_resume_300.json` is the clean package copy with paths relative to the project. The runtime launcher resolves those to absolute paths from this folder. The changed paths do not affect the trainer identity.

## Source snapshot

The exact modules used by the training run are preserved byte for byte in `src/fpcamera/provenance/executed_source/`. Their SHA-256 values are checked against the `source_hashes` object in the step 300 sidecar. Readable package copies live directly in `src/fpcamera/`. The copies are checked for AST equivalence, so documentation and formatting changes do not alter executable syntax or behavior. Two files remain in their original formatting because the available formatter rewrites statements that delete multiple names into a different Python syntax tree shape. Section comments and the README walkthrough explain the training, cache and inference flow.

`artifacts/flow_poc_all/source_snapshot.zip` is retained from the training artifact folder. The package manifest inventories every packaged file except itself and records the checkpoint, resume, source, and dataset identities.

## Integrity checks

```powershell
python scripts/verify_frozen_package.py
```

The verifier checks every manifest digest, then checks the checkpoint against its sidecar and `latest.json`, checks the resume digest, confirms that the frozen backbone was unchanged, counts tensors from the safetensors header, checks source module hashes, and validates the cache index hash, rows, and splits. It does not load or unpickle optimizer state. The cache references and video files are all separately inventoried.

## Runtime dependencies and omitted files

The runtime versions in `requirements.txt` are those recorded by the training/inference runs. Install a matching PyTorch wheel with CUDA support for the host. The frozen multi gigabyte Wan base checkpoint and local Hugging Face cache are intentionally not copied. Credential stores, tokens, cookies, user account files and `.cache` are excluded.

This folder is a project package containing code, model, and data, not the final course report, slides or video presentation. No GitHub repository URL or team member information is invented here.





