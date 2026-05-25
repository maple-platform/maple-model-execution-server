"""모델 config.yaml 로더.

/app/models/{model_name}/config.yaml 을 읽어 dict로 반환합니다.
MODELS_DIR 환경변수로 경로를 오버라이드할 수 있습니다.
"""
import os
from pathlib import Path
from typing import Any

import yaml

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))


def load_model_config(model_name: str) -> dict[str, Any]:
    config_path = MODELS_DIR / model_name / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {config_path}. "
            f"Create models/{model_name}/config.yaml or set MODELS_DIR."
        )
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format: {config_path}")
    return config


def list_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return [d.name for d in MODELS_DIR.iterdir() if (d / "config.yaml").exists()]
