"""Runtime container URL resolver.

config.yaml 의 runtime 필드를 보고 어느 컨테이너로 요청을 보낼지 결정합니다.
각 runtime 의 URL 은 환경변수로 주입됩니다.
"""
import os

_RUNTIME_ENV: dict[str, str] = {
    "runtime-basic":   "RUNTIME_BASIC_URL",
    "runtime-medical": "RUNTIME_MEDICAL_URL",
    "runtime-yolo":    "RUNTIME_YOLO_URL",
    "runtime-nnunet":  "RUNTIME_NNUNET_URL",
}


def resolve_runtime_url(runtime_name: str) -> str | None:
    """runtime 이름 → 컨테이너 URL 반환. 환경변수 미설정이면 None."""
    env_key = _RUNTIME_ENV.get(runtime_name)
    if not env_key:
        return None
    return os.getenv(env_key)


def list_runtimes() -> dict[str, str | None]:
    return {name: resolve_runtime_url(name) for name in _RUNTIME_ENV}
