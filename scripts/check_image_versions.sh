#!/usr/bin/env bash
# check_image_versions.sh
# 각 모델 Docker image 내부의 Python/PyTorch/패키지 버전을 출력합니다.
# 사용법: bash scripts/check_image_versions.sh [image_name]
#   image_name 생략 시 모든 이미지를 순서대로 검사합니다.

set -euo pipefail

IMAGES=(
  "maple/inference-server:latest"
  "maple/brats-t1:latest"
  "maple/brats-t1ce:latest"
  "maple/brats-t2:latest"
  "maple/brats-flair:latest"
  "maple/chestxray14:latest"
  "maple/rsna-pneumonia:latest"
)

CHECK_SCRIPT=$(cat <<'PY'
import sys, subprocess
print("=" * 60)
print("python:", sys.version)

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("cudnn:", torch.backends.cudnn.version())
except Exception as e:
    print("torch: not installed -", e)

packages = [
    "torchvision",
    "ultralytics",
    "nnunetv2",
    "monai",
    "timm",
    "torchxrayvision",
    "cv2",
    "SimpleITK",
    "nibabel",
    "pydicom",
    "skimage",
    "scipy",
    "PIL",
    "fastapi",
    "uvicorn",
    "httpx",
]
for p in packages:
    try:
        mod = __import__(p)
        ver = getattr(mod, "__version__", None) or getattr(mod, "version", "version_unknown")
        print(f"{p}: {ver}")
    except Exception:
        print(f"{p}: not installed")
PY
)

check_image() {
  local image="$1"

  # 이미지가 로컬에 있는지 확인
  if ! docker image inspect "$image" &>/dev/null; then
    echo "⚠️  Image not found locally: $image (skipping)"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📦  Image: $image"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  docker run --rm "$image" python - <<HEREDOC
$CHECK_SCRIPT
HEREDOC
}

if [ $# -eq 1 ]; then
  check_image "$1"
else
  for img in "${IMAGES[@]}"; do
    check_image "$img"
  done
fi

echo ""
echo "✅  Done."
