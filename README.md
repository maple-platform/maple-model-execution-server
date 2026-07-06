# maple-platform AI Inference Server

maple-platform 백엔드에서 분리된 AI 추론 실행 서버입니다.  
FastAPI 게이트웨이가 요청을 받아 runtime 컨테이너로 라우팅하고, 결과를 정규화해 반환합니다.

---

## 아키텍처 개요

```
maple Backend
     │
     ├─ POST /infer      (legacy)  → 개별 모델 컨테이너 → POST /run
     │
     └─ POST /infer/v2   (신규)    → runtime 컨테이너   → POST /run/v2
                                          │
                                   config.yaml 읽기
                                          │
                                   runner.py 동적 로드
                                          │
                                   predict(input_path, output_dir, config)
```

| 엔드포인트 | 방식 | 용도 |
|-----------|------|------|
| `POST /infer` | container_url 직접 지정 | 기존 호환 |
| `POST /infer/v2` | model_name → config.yaml → runtime | 신규 모델 추가 |

---

## 디렉터리 구조

```
maple-model-execution-server/
│
├── main.py                        # 게이트웨이 (/infer + /infer/v2)
├── Dockerfile                     # 게이트웨이 컨테이너
├── requirements.txt
├── docker-compose.yml             # 게이트웨이 서비스
├── docker-compose.runtime.yml     # runtime 컨테이너 4종
│
├── app/
│   ├── model_config.py            # config.yaml 로더
│   ├── runtime_loader.py          # runtime 이름 → URL 매핑
│   ├── runner_loader.py           # runner.py 동적 로드
│   └── runtime_server.py          # runtime 컨테이너용 FastAPI 서버
│
├── docker/
│   ├── Dockerfile.runtime-basic   # NV PyTorch 25.12 + 공통 의료 라이브러리
│   ├── Dockerfile.runtime-medical # runtime-basic + MONAI + TorchXRayVision
│   ├── Dockerfile.runtime-yolo    # python:3.12-slim + ultralytics
│   └── Dockerfile.runtime-nnunet  # runtime-basic + nnunetv2
│
├── models/                        # 모델별 config + runner (volume mount)
│   ├── BraTS2020_T1_UNet3D/
│   ├── BraTS2020_T1ce_UNet3D/
│   ├── BraTS2020_T2_UNet3D/
│   ├── BraTS2020_FLAIR_UNet3D/
│   ├── ChestXray14_Multilabel_Classification/
│   ├── RSNA_Pneumonia_YOLO26x/
│   └── example_model/             # 템플릿
│
├── AI_Models/                     # 기존 inference.py + 모델 소스 (volume mount)
│   ├── Neurology/
│   │   ├── BraTS2020_T1_UNet3D/
│   │   ├── BraTS2020_T1ce_UNet3D/
│   │   ├── BraTS2020_T2_UNet3D/
│   │   └── BraTS2020_FLAIR_UNet3D/
│   └── Pulmonology/
│       ├── ChestXray14_Multilabel_Classification/
│       └── RSNA_Pneumonia_YOLO26x/
│
├── inputs/                        # 추론 입력 파일 (volume mount, gitignore)
├── outputs/                       # 추론 출력 파일 (volume mount, gitignore)
└── docs/
    ├── runtime_architecture.md
    ├── runtime_migration_plan.md
    └── model_runtime_inventory.md
```

---

## 빠른 시작

```bash
# 1단계: runtime-basic 먼저 빌드 (medical, nnunet이 의존)
docker compose -f docker-compose.runtime.yml build runtime-basic

# 2단계: 전체 빌드 및 실행
docker compose -f docker-compose.yml -f docker-compose.runtime.yml up -d --build
```

### 헬스 체크

```bash
curl http://localhost:8110/health
# {"status": "ok", "service": "maple-inference-server"}

# runtime 컨테이너 헬스 체크 (등록된 모델 목록 포함)
curl http://localhost:9021/health
# {"status": "ok", "service": "maple-runtime-server", "models": [...]}
```

---

## 등록된 모델

| model_name | runtime | 입력 | 출력 |
|------------|---------|------|------|
| `BraTS2020_T1_UNet3D` | runtime-medical | NIfTI (.nii.gz) | 분할 오버레이 이미지 (WT/TC/ET) |
| `BraTS2020_T1ce_UNet3D` | runtime-medical | NIfTI (.nii.gz) | 분할 오버레이 이미지 (WT/TC/ET) |
| `BraTS2020_T2_UNet3D` | runtime-medical | NIfTI (.nii.gz) | 분할 오버레이 이미지 (WT/TC/ET) |
| `BraTS2020_FLAIR_UNet3D` | runtime-medical | NIfTI (.nii.gz) | 분할 오버레이 이미지 (WT/TC/ET) |
| `ChestXray14_Multilabel_Classification` | runtime-medical | PNG/JPG (흉부 X-ray) | Grad-CAM 오버레이 + 14개 레이블 예측 |
| `RSNA_Pneumonia_YOLO26x` | runtime-yolo | DICOM (.dcm) | 폐렴 opacity bbox 오버레이 |

---

## Runtime 컨테이너

| 컨테이너 | 포트 | Base Image | 탑재 라이브러리 |
|---------|------|-----------|----------------|
| `runtime-basic` | 9020 | nvcr.io/nvidia/pytorch:25.12-py3 | nibabel, scipy, scikit-image |
| `runtime-medical` | 9021 | runtime-basic | + MONAI 1.5.2, TorchXRayVision 1.4.0 |
| `runtime-yolo` | 9022 | python:3.12-slim | ultralytics, pydicom, opencv |
| `runtime-nnunet` | 9023 | runtime-basic | + nnunetv2 |

---

## API

### `GET /health`

```json
{"status": "ok", "service": "maple-inference-server"}
```

---

### `POST /infer/v2` — 신규 runtime 기반 추론

**Request**

```json
{
  "model_name": "BraTS2020_T1_UNet3D",
  "input_path": "/app/inputs/sample.nii.gz",
  "output_dir": "/app/outputs/BraTS2020_T1_UNet3D",
  "params": {}
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `model_name` | `str` | `models/` 디렉터리 이름과 일치 |
| `input_path` | `str` | 컨테이너 내부 입력 파일 경로 |
| `output_dir` | `str` | 출력 디렉터리 (미지정 시 `/app/outputs/{model_name}`) |
| `params` | `dict` | config 오버라이드 (선택) |

**Response**

```json
{
  "status": "ok",
  "model_name": "BraTS2020_T1_UNet3D",
  "result": {
    "images_b64": ["<base64 PNG>", "..."],
    "labels": ["all_regions", "WT", "TC", "ET"],
    "output_files": ["/app/outputs/BraTS2020_T1_UNet3D/sample_all_regions.png"]
  }
}
```

**에러 응답**

| HTTP | 발생 조건 |
|------|----------|
| `404` | model_name에 해당하는 config.yaml 없음 |
| `503` | runtime 컨테이너 URL 환경변수 미설정 |
| `502` | runtime 컨테이너 오류 |
| `504` | 타임아웃 |

---

### `POST /infer` — Legacy 직접 라우팅

기존 호환용. `params.container_url`에 컨테이너 URL을 직접 지정합니다.

```json
{
  "model_id": "brats-t1",
  "input_data": "/app/inputs/sample.nii.gz",
  "params": {
    "container_url": "http://localhost:9021",
    "model_path": "/app/AI_Models/Neurology/BraTS2020_T1_UNet3D/checkpoint/best.pt",
    "timeout": 120.0
  }
}
```

---

## 환경 변수

### 게이트웨이

| 변수 | 설명 |
|------|------|
| `RUNTIME_BASIC_URL` | `http://runtime-basic:8000` |
| `RUNTIME_MEDICAL_URL` | `http://runtime-medical:8000` |
| `RUNTIME_YOLO_URL` | `http://runtime-yolo:8000` |
| `RUNTIME_NNUNET_URL` | `http://runtime-nnunet:8000` |
| `MODELS_DIR` | config.yaml 탐색 경로 (기본값: `/app/models`) |

### Runtime 컨테이너

| 변수 | 기본값 |
|------|--------|
| `MODELS_DIR` | `/app/models` |
| `INPUTS_DIR` | `/app/inputs` |
| `OUTPUTS_DIR` | `/app/outputs` |

---

## 새 모델 추가 가이드

이미지 재빌드 없이 `models/` 디렉터리에 파일 2개만 추가하면 됩니다.

### 1. `models/<model_name>/config.yaml`

```yaml
model_name: my_model
execution_mode: external_runtime
runtime: runtime-medical          # runtime-basic | runtime-medical | runtime-yolo | runtime-nnunet
runner_path: /app/models/my_model/runner.py
inference_path: /app/AI_Models/.../inference.py
model_path: /app/AI_Models/.../checkpoint/best.pt
```

### 2. `models/<model_name>/runner.py`

```python
def predict(input_path: str, output_dir: str, config: dict) -> dict:
    # 1. config["inference_path"] 에서 inference.py 동적 로드
    # 2. 추론 실행
    # 3. 결과를 output_dir 에 저장
    # 4. JSON-serializable dict 반환
    return {"images_b64": [...], "output_files": [...]}
```

### 3. 추론 요청

```bash
curl -X POST http://localhost:8110/infer/v2 \
  -H "Content-Type: application/json" \
  -d '{"model_name": "BraTS2020_T1_UNet3D", "input_path": "/app/inputs/sample.nii.gz"}'
```

자세한 내용은 [docs/runtime_architecture.md](docs/runtime_architecture.md) 참고.

---

## 의존성

### 게이트웨이

```
fastapi==0.121.3
uvicorn==0.34.1
pydantic==2.11.3
httpx==0.27.0
pyyaml==6.0.2
```

### Runtime 컨테이너별 주요 패키지

| Runtime | 주요 패키지 |
|---------|------------|
| runtime-basic | nibabel 5.4.2, scipy, scikit-image, Pillow |
| runtime-medical | + monai 1.5.2, torchxrayvision 1.4.0 |
| runtime-yolo | ultralytics 8.4.51, pydicom, opencv-python-headless |
| runtime-nnunet | + nnunetv2 |
