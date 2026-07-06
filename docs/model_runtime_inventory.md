# Model Runtime Inventory

> 작성일: 2026-05-25  
> 목적: Docker image 통합 전 각 모델의 실행 환경 및 의존성 현황 파악  
> 범위: 이 레포에 실제 존재하는 모델만 포함 (부재 모델은 별도 섹션에 기록)

---

## 1. 모델 목록 및 존재 여부

| Model | Docker Image | 레포 내 존재 | 비고 |
|-------|-------------|-------------|------|
| maple-inference (Gateway) | `maple/inference-server` | ✅ | `main.py` + `Dockerfile` |
| BraTS2020 T1 UNet3D | `maple/brats-t1` | ✅ | `AI_Models/Neurology/BraTS2020_T1_UNet3D/` |
| BraTS2020 T1ce UNet3D | `maple/brats-t1ce` | ✅ | `AI_Models/Neurology/BraTS2020_T1ce_UNet3D/` |
| BraTS2020 T2 UNet3D | `maple/brats-t2` | ✅ | `AI_Models/Neurology/BraTS2020_T2_UNet3D/` |
| BraTS2020 FLAIR UNet3D | `maple/brats-flair` | ✅ | `AI_Models/Neurology/BraTS2020_FLAIR_UNet3D/` |
| ChestXray14 Multilabel | `maple/chestxray14` | ✅ | `AI_Models/Pulmonology/ChestXray14_Multilabel_Classification/` |
| RSNA Pneumonia YOLO | `maple/rsna-pneumonia` | ✅ | `AI_Models/Pulmonology/RSNA_Pneumonia_YOLO26x/` |
| nnUNet SMWI Segmentation | `maple/nnunet-smwi` | ❌ | 이 레포에 없음 (별도 레포 추정) |
| BME Classification | `maple/bme-classifier` | ❌ | 이 레포에 없음 |
| SI Joints Detection | `maple/si-joint-detector` | ❌ | 이 레포에 없음 |
| Gateway (현재) | `maple/inference-server` | ✅ | maple-inference (포트 8110) |

---

## 2. 모델별 버전 정보 (Inventory)

### 2-A. maple/inference-server (Gateway)

| 항목 | 값 |
|------|----|
| Dockerfile | `./Dockerfile` |
| Base Image | `python:3.11-slim` |
| Python | 3.11 |
| PyTorch | 없음 |
| CUDA | 없음 (CPU only) |
| 주요 프레임워크 | FastAPI 0.121.3, uvicorn 0.34.1, httpx 0.27.0 |
| Weight in Image | 해당 없음 |
| Runner | `main.py` |
| Input | HTTP JSON (`POST /infer`) |
| Output | HTTP JSON (`InferResponse`) |
| 특이사항 | 순수 라우팅 게이트웨이. GPU 불필요 |

---

### 2-B. maple/brats-t1 / brats-t1ce / brats-t2 / brats-flair (공통)

> 4개 모델이 동일한 requirements.txt, 동일한 Dockerfile 구조를 사용.  
> 차이점: 모델 폴더명, checkpoint/best.pt 가중치, inference.py 내 로그 이름.

| 항목 | 값 |
|------|----|
| Dockerfile | `AI_Models/Neurology/BraTS2020_{MODALITY}_UNet3D/Dockerfile` |
| Base Image | `nvcr.io/nvidia/pytorch:25.12-py3` |
| Python | 3.12 (NV container 기준) |
| PyTorch (requirements.txt 명시) | `2.12.0+cu130` |
| PyTorch (실제 컨테이너) | `2.10.0a0+b4e4ee81d3.nv25.12` (NV 제공, requirements.txt와 **불일치**) |
| CUDA (requirements.txt 명시) | 13.0 |
| CUDA (NV container 실제) | ~13.1 (B200 최적화) |
| cuDNN | 9.x (NV container 기준) |
| 주요 프레임워크 | MONAI 1.5.2 |
| 기타 의존성 | nibabel 5.4.2, numpy 2.4.6, scipy, scikit-image, Pillow |
| 선택적 의존성 | vedo (3D 렌더링, try/except로 graceful degradation) |
| torchvision | 미설치 (불필요) |
| Weight in Image | ✅ (`COPY . .`로 `checkpoint/best.pt` 포함) |
| Weight Path (컨테이너 내) | `/app/checkpoint/best.pt` |
| Weight Path (volume) | `/AI_Models/Neurology/BraTS2020_{MODALITY}_UNet3D/checkpoint/best.pt` |
| Runner | `server.py` → `inference.py::main()` |
| Input | NIfTI (`.nii.gz`) 파일 경로 |
| Output | `list[np.ndarray]` — 3D 렌더링 + axial overlay 이미지 (최대 5장) |
| 추론 명령 | `uvicorn server:app --host 0.0.0.0 --port 9000` |
| 특이사항 | requirements.txt의 torch 버전과 NV container 내 실제 torch 버전이 다름. 현재 Dockerfile에서 requirements.txt를 설치하지 않아 NV container 기본 torch 사용 중 |

---

### 2-C. maple/chestxray14

| 항목 | 값 |
|------|----|
| Dockerfile | `AI_Models/Pulmonology/ChestXray14_Multilabel_Classification/Dockerfile` |
| Base Image | `nvcr.io/nvidia/pytorch:25.12-py3` |
| Python | 3.12 |
| PyTorch | `2.10.0a0+b4e4ee81d3.nv25.12` (NV container 제공, requirements.txt에서 주석 처리) |
| torchvision | NV container 제공 (`0.25.0a0+ca221243`) |
| CUDA | 13.1 (B200 환경 기준) |
| cuDNN | 9.x |
| 주요 프레임워크 | TorchXRayVision 1.4.0 (`--no-deps` 설치) |
| 기타 의존성 | numpy 2.1.0, Pillow 12.0.0, imageio, scikit-image, opencv-python-headless 4.10.0.84 |
| Weight in Image | ❌ (런타임에 TorchXRayVision이 자동 다운로드: `resnet50-res512-all`) |
| Weight Path | 런타임 캐시 (`~/.cache/torch/hub/`) |
| Runner | `server.py` → `inference.py::main()` |
| Input | PNG/JPG 흉부 X-ray 파일 경로 |
| Output | `(np.ndarray, list[dict], dict)` — Grad-CAM overlay + 14개 레이블 분류 확률 |
| 추론 명령 | `uvicorn server:app --host 0.0.0.0 --port 9000` |
| 특이사항 | 오프라인 환경에서는 가중치 사전 캐싱 필요. `--no-deps`로 설치하여 NV container torch와 충돌 방지 |

---

### 2-D. maple/rsna-pneumonia

| 항목 | 값 |
|------|----|
| Dockerfile | `AI_Models/Pulmonology/RSNA_Pneumonia_YOLO26x/Dockerfile` |
| Base Image | `python:3.12-slim` |
| Python | 3.12 |
| PyTorch | ultralytics 내 torch 의존성 (CPU/GPU 자동) |
| CUDA | 선택적 (CPU 동작 가능) |
| 주요 프레임워크 | ultralytics 8.4.51 (YOLOv8/v12 계열) |
| 기타 의존성 | pydicom 3.0.2, numpy 2.4.6, Pillow 12.2.0, opencv-python-headless 4.10.0.84 |
| Weight in Image | ✅ (`COPY . .`로 `checkpoint/best.pt` 포함) |
| Weight Path (컨테이너 내) | `/app/checkpoint/best.pt` |
| Weight Path (volume) | `/AI_Models/Pulmonology/RSNA_Pneumonia_YOLO26x/checkpoint/best.pt` |
| Runner | `server.py` → `inference.py::main()` |
| Input | DICOM 흉부 X-ray 파일 경로 |
| Output | `(np.ndarray, list[dict])` — bbox overlay 이미지 + 탐지 결과 목록 |
| 추론 명령 | `uvicorn server:app --host 0.0.0.0 --port 9000` |
| 특이사항 | GPU 없어도 동작. DICOM은 비압축 포맷이라 pylibjpeg 불필요 |

---

### 2-E. 부재 모델 (이 레포에 없음)

| Model | 마지막 알려진 Image | 비고 |
|-------|-------------------|------|
| SI Joints Detection | `maple/si-joint-detector` | YOLOv12, DICOM 입력, CUDA 필요 |
| BME Classification | `maple/bme-classifier` | GradCAM++, ConvNeXt-Large, CUDA 12.1, torch 2.5.1+cu121 |
| ParkinsonGait_ML | `maple/parkinson-gait` | ExtraTrees, CPU only, scikit-learn |
| nnUNet SMWI | `maple/nnunet-smwi` | nnunetv2, CUDA 12.8, vtk-osmesa |

---

## 3. 현재 Image 크기 (Docker Desktop 기준)

| Image | Size | 비고 |
|-------|------|------|
| `maple/inference-server` | 227 MB | python:3.11-slim 기반 |
| `maple/rsna-pneumonia` | 9.59 GB | ultralytics + torch |
| `maple/brats-t1` | 30.47 GB | NV PyTorch 25.12 기반 |
| `maple/brats-t1ce` | 30.47 GB | 레이어 공유 (실제 추가 점유 최소) |
| `maple/brats-t2` | 30.47 GB | 레이어 공유 |
| `maple/brats-flair` | 30.47 GB | 레이어 공유 |
| `maple/chestxray14` | 30.56 GB | NV PyTorch 25.12 기반 |

> **실제 디스크 사용량**: Docker는 동일 base image 레이어를 공유 저장.  
> brats-t1/t1ce/t2/flair + chestxray14 는 `nvcr.io/nvidia/pytorch:25.12-py3` (~30GB)를 공유하므로,  
> 실제 추가 점유는 각 모델별 pip 설치 레이어 + checkpoint 파일 크기.

---

## 4. Suggested Runtime Groups

| Group | 해당 모델 | Base Image | Python | PyTorch | CUDA | 주요 Framework |
|-------|----------|-----------|--------|---------|------|---------------|
| `runtime-nvpytorch2512-monai` | brats-t1, brats-t1ce, brats-t2, brats-flair | `nvcr.io/nvidia/pytorch:25.12-py3` | 3.12 | 2.10.0a0.nv25.12 | 13.1 | MONAI 1.5.2 |
| `runtime-nvpytorch2512-xray` | chestxray14 | `nvcr.io/nvidia/pytorch:25.12-py3` | 3.12 | 2.10.0a0.nv25.12 | 13.1 | TorchXRayVision 1.4.0 |
| `runtime-py312-cpu-yolo` | rsna-pneumonia | `python:3.12-slim` | 3.12 | (ultralytics 내 포함) | optional | ultralytics 8.4.51 |
| `runtime-py311-slim-gateway` | maple-inference | `python:3.11-slim` | 3.11 | 없음 | 없음 | FastAPI |

> **monai와 torchxrayvision 분리 이유**:  
> - MONAI는 nibabel/scipy/scikit-image와 함께 3D 의료 영상 처리 특화  
> - TorchXRayVision은 2D CXR 분류 특화, `--no-deps` 설치 필요  
> - 동일 컨테이너에 공존 가능하나 목적이 달라 variant로 분리 권장  
>  
> **YOLO 분리 이유**:  
> - ultralytics는 자체 torch 의존성 관리. NV container와 충돌 가능성  
> - CPU 동작 가능하므로 GPU base image 불필요 → 경량 이미지로 유지
