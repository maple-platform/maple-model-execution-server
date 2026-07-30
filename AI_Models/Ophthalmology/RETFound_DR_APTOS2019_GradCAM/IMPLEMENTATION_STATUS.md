# Implementation status

## Completed

- Compared Maple `AI_Models/Manual.md` with the live repository structure.
- Confirmed the platform's image/classification metadata fields and RGB `uint8`
  image result contract.
- Reproduced RETFound's official evaluation preprocessing: bicubic resize to 256,
  center crop to 224, tensor conversion, ImageNet mean/std normalization.
- Implemented the RETFound-MAE ViT-L/16 five-class inference definition.
- Implemented last-block patch-token Grad-CAM and overlay on the model-visible crop.
- Added strict input, checkpoint structure, output shape, and probability checks.
- Added model caching, attribution, non-commercial restriction, and clinical warnings.
- Passed Python syntax compilation and static Maple metadata/folder contract checks.
- Rechecked against timm v0.9.2 `forward_head()` and preserved RETFound's required
  singleton token dimension (`[B, 1, C]`) before the classification head.
- Downloaded the official APTOS checkpoint and verified SHA-256, 5-class head,
  input size 224, global pooling, and epoch 27 using PyTorch weights-only mode.
- Executed the complete Maple `main()` path on CPU with the official checkpoint
  and a CC BY 4.0 fundus smoke-test image; output/probability contract passed.
- Added and executed the Maple runtime adapter under the local repository patch:
  `outputs/maple_repo_patch/models/RETFound_DR_APTOS2019_GradCAM/`. The runner
  saved the Grad-CAM PNG and returned JSON-safe probabilities and top prediction.

## Artifact status

No local execution-blocking artifact remains. The official checkpoint was
downloaded and verified in the development workspace but is intentionally not
committed to GitHub. A CC BY 4.0 smoke-test image is included. Formal APTOS
validation data remains a separate governed activity.

The official APTOS checkpoint must be delivered separately. Redistribution
permission must be confirmed before publishing the 3.64 GB file.

## Final local test

After installing `requirements.txt`:

```bash
python validate_package.py \
  --image sample_data/fundus_diabetic_retinopathy_ccby4.png \
  --checkpoint checkpoint/checkpoint-best.pth \
  --output validation/gradcam_overlay.png
```

Verified CPU environment: Python 3.12, PyTorch 2.5.1+cpu, torchvision 0.20.1,
timm 0.9.2, NumPy 1.26.4, OpenCV 4.9.0. Contract test result: PASS.

Smoke-test output (not clinical ground truth): Moderate was highest at
`0.56615448`; the five probabilities summed to 1 and the Grad-CAM output was
RGB `uint8` with shape `(224, 224, 3)`.

Record at minimum: checkpoint filename, SHA-256, source URL, test image provenance,
GPU/CPU, Python/PyTorch/CUDA versions, predicted class, probability sum, output
shape/dtype, and inference time.

## Remaining administrator handoff

The model runner/config are now prepared. The Maple administrator still needs to
make the 3.64 GB checkpoint available at the configured container path, confirm
the medical GPU runtime dependencies, set `docker.service_url`, register the
model, and run the container/API integration test.
