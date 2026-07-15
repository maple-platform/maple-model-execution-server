import logging
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.model_config import load_model_config
from app.runtime_loader import resolve_runtime_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("maple.inference")


class InferRequest(BaseModel):
    model_id: str
    input_data: Any
    params: dict[str, Any] = Field(default_factory=dict)


class InferResponse(BaseModel):
    result: Any = None
    output_images: list[str] = Field(default_factory=list)
    model_output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="maple Model Execution Server — Inference Gateway", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_container_payload(input_data: Any, params: dict[str, Any]) -> dict[str, Any]:
    model_name = params.get("model_name", "")
    extra_params: dict[str, Any] = {}

    if isinstance(input_data, str):
        input_path = input_data
    elif isinstance(input_data, dict) and "roi" in input_data:
        input_path = input_data.get("image_path", "")
        extra_params["roi"] = input_data["roi"]
    elif isinstance(input_data, list):
        input_path = ""
        extra_params["data"] = input_data
    else:
        input_path = ""
        extra_params["data"] = input_data

    model_path = params.get("model_path")
    if model_path:
        extra_params["model_path"] = model_path

    extra_params.update(params.get("container_payload") or {})

    payload: dict[str, Any] = {
        "model_name": model_name,
        "input_path": input_path,
    }
    if extra_params:
        payload["params"] = extra_params
    return payload


def _normalize_container_response(model_id: str, raw: dict[str, Any]) -> InferResponse:
    # Runtime v2 containers return {"status": "ok", "model_name": "...", "result": {...}}
    # Legacy flat-response containers return a flat dict with image_b64 / result_type at top level
    runner_data = raw.get("result") if isinstance(raw.get("result"), dict) else raw

    images = runner_data.get("images_b64") or []
    if not images and runner_data.get("image_b64"):
        images = [runner_data["image_b64"]]

    model_output = {
        key: value
        for key, value in runner_data.items()
        if key not in ("status", "result_type", "predictions", "image_b64", "images_b64")
    }

    return InferResponse(
        result={
            "result_type": runner_data.get("result_type"),
            "predictions": runner_data.get("predictions"),
            "data": runner_data.get("data"),
        },
        output_images=images,
        model_output=model_output,
        metadata={
            "model_id": model_id,
            "container_status": raw.get("status"),
            "result_type": runner_data.get("result_type"),
        },
    )


def _resolve_container_url(container_url: str) -> str:
    """localhost URL을 환경변수에 정의된 Docker 서비스 URL로 교체.

    로컬 개발 환경에서 container_url 이 localhost:PORT 로 들어올 때
    RUNTIME_*_URL 환경변수가 설정되어 있으면 해당 URL 로 교체합니다.
    Docker 네트워크 내부에서는 서비스명으로 직접 통신하므로 동작하지 않습니다.
    """
    parsed = urlparse(container_url)
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        return container_url

    port_map: dict[int | None, str | None] = {
        9020: os.getenv("RUNTIME_BASIC_URL"),
        9021: os.getenv("RUNTIME_MEDICAL_URL"),
        9022: os.getenv("RUNTIME_YOLO_URL"),
        9023: os.getenv("RUNTIME_NNUNET_URL"),
    }
    replacement = port_map.get(parsed.port)
    return replacement.rstrip("/") if replacement else container_url


@app.get("/health")
async def health():
    return {"status": "ok", "service": "inference-gateway"}


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    container_url = (req.params.get("container_url") or "").rstrip("/")
    container_endpoint = req.params.get("container_endpoint", "/run")
    timeout = float(req.params.get("timeout", 120.0))

    if not container_url:
        raise HTTPException(status_code=400, detail="params.container_url is required")
    if not container_endpoint.startswith("/"):
        container_endpoint = f"/{container_endpoint}"
    container_url = _resolve_container_url(container_url)

    payload = _build_container_payload(req.input_data, req.params)
    logger.info("Infer request - model_id=%s target=%s%s", req.model_id, container_url, container_endpoint)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{container_url}{container_endpoint}", json=payload)
            resp.raise_for_status()
            raw = resp.json()
    except httpx.TimeoutException as exc:
        logger.error("Container timeout - model_id=%s: %s", req.model_id, exc)
        raise HTTPException(status_code=504, detail=f"Container timeout: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        logger.error("Container HTTP error - model_id=%s: %s", req.model_id, exc)
        raise HTTPException(status_code=502, detail=f"Container HTTP error: {exc}") from exc
    except httpx.HTTPError as exc:
        logger.error("Container request failed - model_id=%s: %s", req.model_id, exc)
        raise HTTPException(status_code=502, detail=f"Container request failed: {exc}") from exc

    if raw.get("status") != "ok":
        detail = raw.get("detail") or raw.get("message") or "Container inference failed"
        raise HTTPException(status_code=502, detail=detail)

    return _normalize_container_response(req.model_id, raw)


class InferV2Request(BaseModel):
    model_name: str
    input_path: str
    output_dir: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


@app.post("/infer/v2")
async def infer_v2(req: InferV2Request):
    """Runtime-image-based inference. model_name → config.yaml → runtime container → /run/v2."""
    try:
        config = load_model_config(req.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    runtime_name = config.get("runtime")
    if not runtime_name:
        raise HTTPException(status_code=422, detail=f"config.yaml for '{req.model_name}' missing 'runtime' field")

    runtime_url = resolve_runtime_url(runtime_name)
    if not runtime_url:
        raise HTTPException(
            status_code=503,
            detail=f"Runtime '{runtime_name}' URL not configured (set env var for this runtime)",
        )

    payload = {
        "model_name": req.model_name,
        "input_path": req.input_path,
        "output_dir": req.output_dir,
        "params": req.params,
    }
    timeout = float(req.params.get("timeout", 120.0))
    logger.info("InferV2 request - model_name=%s runtime=%s", req.model_name, runtime_name)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{runtime_url.rstrip('/')}/run/v2", json=payload)
            resp.raise_for_status()
            raw = resp.json()
    except httpx.TimeoutException as exc:
        logger.error("Runtime timeout - model_name=%s: %s", req.model_name, exc)
        raise HTTPException(status_code=504, detail=f"Runtime timeout: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        logger.error("Runtime HTTP error - model_name=%s: %s", req.model_name, exc)
        raise HTTPException(status_code=502, detail=f"Runtime HTTP error: {exc}") from exc
    except httpx.HTTPError as exc:
        logger.error("Runtime request failed - model_name=%s: %s", req.model_name, exc)
        raise HTTPException(status_code=502, detail=f"Runtime request failed: {exc}") from exc

    if raw.get("status") != "ok":
        detail = raw.get("detail") or raw.get("message") or "Runtime inference failed"
        raise HTTPException(status_code=502, detail=detail)

    return raw
