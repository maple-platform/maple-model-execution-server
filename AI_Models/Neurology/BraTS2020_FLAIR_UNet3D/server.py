#!/usr/bin/env python3
"""FastAPI inference server for BraTS2020 FLAIR UNet3D segmentation.

Endpoint: POST /run
  Request:  {"image_path": "<nifti path>", "model_path": "<optional>"}
  Response: {"status": "ok", "result_type": "image", "images_b64": [...]}
"""
import base64
import io
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("maple.brats_flair")

_HERE = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = str(os.environ.get("MODEL_PATH") or (_HERE / "checkpoint" / "best.pt"))

sys.path.insert(0, str(_HERE))


class RunRequest(BaseModel):
    image_path: str
    model_path: str | None = None


app = FastAPI(title="BraTS2020 FLAIR UNet3D Inference Server")


def _ndarray_to_b64(img) -> str:
    import numpy as np
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "brats-flair-inference"}


@app.post("/run")
async def run(req: RunRequest):
    model_path = req.model_path or _DEFAULT_MODEL_PATH
    logger.info("Run - image_path=%s model_path=%s", req.image_path, model_path)

    try:
        from inference import main
        results = main(req.image_path, model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    images_b64 = [_ndarray_to_b64(img) for img in results]
    return {
        "status": "ok",
        "result_type": "image",
        "images_b64": images_b64,
    }
