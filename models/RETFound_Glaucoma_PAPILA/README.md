# RETFound_Glaucoma_PAPILA runtime adapter

This folder connects Maple's generic runtime server to:

`AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA/inference.py`

## Files

- `config.yaml`: selects `runtime-medical` and supplies inference/checkpoint paths.
- `runner.py`: calls `main()`, saves the Grad-CAM PNG, and returns JSON-safe
  classification results. Identical structure to
  `RETFound_DR_APTOS2019_GradCAM`'s runner (single-image input), adjusted for
  the 3-class glaucoma head.

## Runner output

| Key | Meaning |
|---|---|
| `model` | Maple model name |
| `image_b64` | Base64-encoded Grad-CAM PNG |
| `output_image_role` | `gradcam_overlay` |
| `predictions` | Ordered classes 0-2 (normal, suspect, glaucoma) with probabilities |
| `top_prediction` | Highest-probability class |
| `gradcam_target` | Class used as the Grad-CAM backward target |
| `output_file` | Saved PNG path inside the runtime |

The ~3.4 GB checkpoint is delivered separately and must appear under the
configured checkpoint directory as `checkpoint-best.pth`.

**Known limitation:** batch validation against the full official PAPILA test
split found the "Glaucoma" class never wins the top-1 argmax, even for
confirmed glaucoma cases (class order is confirmed correct; the probability
signal is real but too small to dominate a 3-way argmax -- likely a
class-imbalance effect). Any UI surfacing this model's output should show
all three class probabilities, not just `top_prediction`. See
`AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA/IMPLEMENTATION_STATUS.md`
for the full validation data.

RETFound is CC BY-NC 4.0 and is limited to research/non-commercial use unless
separate permission is obtained.

## Docker execution

The host and container paths are connected by the compose mounts:

| Host path | Container path | Mode |
|---|---|---|
| `models/` | `/app/models` | read-only |
| `AI_Models/` | `/app/AI_Models` | read-only |
| `inputs/` | `/app/inputs` | read/write |
| `outputs/` | `/app/outputs` | read/write |

Place `checkpoint-best.pth` in the checkpoint directory and put the input
image under `inputs/`. Then run from the repository root:

```bash
docker compose -f docker-compose.yml -f docker-compose.runtime.yml build runtime-basic runtime-medical inference-gateway
docker compose -f docker-compose.yml -f docker-compose.runtime.yml up -d runtime-medical inference-gateway
docker compose -f docker-compose.yml -f docker-compose.runtime.yml ps
curl -f http://localhost:9021/health
curl -f http://localhost:8110/ready
```

Gateway request:

```bash
curl -sS http://localhost:8110/infer/v2 \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "RETFound_Glaucoma_PAPILA",
    "input_path": "/app/inputs/papila.jpg",
    "output_dir": "/app/outputs/RETFound_Glaucoma_PAPILA",
    "params": {"timeout": 300}
  }'
```

For runtime-only troubleshooting, bypass the Gateway:

```bash
curl -sS http://localhost:9021/run/v2 \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "RETFound_Glaucoma_PAPILA",
    "input_path": "/app/inputs/papila.jpg",
    "output_dir": "/app/outputs/RETFound_Glaucoma_PAPILA",
    "params": {}
  }'
```

Inspect failures without rebuilding:

```bash
docker compose -f docker-compose.yml -f docker-compose.runtime.yml logs --tail=200 inference-gateway runtime-medical
```
