#!/usr/bin/env python3
"""FastAPI inference server for ChestXray14 multilabel classification.

Endpoint: POST /run
  Request:  {"image_path": "<png/jpg path>", "model_path": "<ignored — pretrained weights>"}
  Response: {
    "status": "ok",
    "result_type": "image",
    "image_b64": "<Grad-CAM overlay PNG>",
    "predictions": [{"label": ..., "prob": ..., "threshold": ..., "pred": ..., "pred_name": ...}, ...],
    "top_findings": [...],
    "gradcam_target": "<label>",
    "model": "TorchXRayVision resnet50-res512-all",
    "threshold_method": "..."
  }
"""
import base64
import io
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("maple.chestxray14")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


class RunRequest(BaseModel):
    image_path: str
    model_path: str | None = None  # ignored — TorchXRayVision pretrained weights are fixed


app = FastAPI(title="ChestXray14 Multilabel Classification Inference Server")


def _ndarray_to_b64(img) -> str:
    import numpy as np
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chestxray14-inference"}


@app.post("/run")
async def run(req: RunRequest):
    logger.info("Run - image_path=%s", req.image_path)

    try:
        from inference import main
        result_image, predictions, model_output = main(req.image_path, req.model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return {
        "status": "ok",
        "result_type": "image",
        "image_b64": _ndarray_to_b64(result_image),
        "predictions": predictions,
        "top_findings": model_output.get("top_findings"),
        "gradcam_target": model_output.get("gradcam_target"),
        "model": model_output.get("model"),
        "threshold_method": model_output.get("threshold_method"),
    }
