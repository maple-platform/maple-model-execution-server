# maple-model-execution-server

maple-platform 백엔드에서 분리된 **모델 실행 서버(model execution server)** 입니다.  
모델 추론을 담당하는 스택 전체 — **추론 게이트웨이(inference gateway) + runtime 컨테이너 + 모델 자산** — 를 한 저장소로 묶어 관리합니다.

> **이 저장소의 역할**: 단순히 Dockerfile을 올려두는 것이 아니라, **빌드된 이미지들을 상시 실행되는 서비스로 띄우는 것**입니다.
> 라우팅 서버는 runtime 컨테이너를 직접 호출하지 않고, 이 저장소가 띄우는 **추론 게이트웨이(`inference-gateway`, :8110)** 에 요청을 보냅니다.
> 게이트웨이가 `model_name → config.yaml → runtime 컨테이너`로 라우팅하고 응답을 정규화합니다.

### 구성 계층

| 계층 | 실체 | 역할 |
|------|------|------|
| **추론 게이트웨이** | `inference-gateway` (:8110) · [main.py](main.py) | 요청을 받아 runtime 컨테이너로 라우팅하고 응답을 정규화하는 상주 서비스 |
| **runtime 컨테이너** | `runtime-*` (:9020~9023) | 모델을 실제로 실행 (GPU 추론) |
| **모델 자산** | [models/](models/) · [AI_Models/](AI_Models/) | 모델별 `config.yaml` · `runner.py` · 소스 · 가중치 |

---

## 아키텍처 개요

```
라우팅 서버 (maple-routing-server)
     │
     ├─ POST /infer      (legacy)  → 개별 모델 컨테이너 → POST /run
     │
     └─ POST /infer/v2   (신규)    → 추론 게이트웨이(:8110) → runtime 컨테이너 → POST /run/v2
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

## 시스템 구성 (배포 토폴로지)

두 대의 GPU 서버 + 로컬 클라이언트로 구성됩니다.
**서버 간 통신은 인터넷망을 경유하지 않고 사설망(`ens224`)의 사설 IP를 사용**합니다.

```
Client (로컬)
   │
   ▼
┌──────────────────────────────────────────────────┐
│ A100 서버                                          │  ← 이 저장소가 배포되는 서버
│   ├─ 라우팅 서버 (maple-routing-server)    :8100   │
│   ├─ 추론 게이트웨이 (이 저장소)           :8110   │
│   ├─ runtime 컨테이너                      :9020~9023 │
│   └─ MongoDB                               :27017 (localhost) │
└──────────────────────────────────────────────────┘
          │  사설망 (인터넷 미경유) → Agent :8101
          ▼
┌──────────────────────────────────────────────────┐
│ H100 서버                                          │
│   └─ Agent                                 :8101   │
└──────────────────────────────────────────────────┘
```

| 서버 | 역할 | 포트 |
|------|------|------|
| A100 서버 | 라우팅 + 모델 실행 + MongoDB | 8100 / 8110 / 9020~9023 / 27017 |
| H100 서버 | Agent | 8101 |

> 역할 매핑은 **현재 A100 서버에서 실제 구동 중인 프로세스 기준**입니다. 배치가 바뀌면 이 표만 갱신하세요.

**통신 규칙**

- 서버 간(라우팅 → Agent)은 **H100 서버의 사설 IP** 사용: 라우팅 서버 `.env`의 `AGENT_URL = http://<H100 사설 IP>:8101`
- 서버 내(라우팅 → 모델 실행, → MongoDB)는 A100 서버에 함께 있으므로 `localhost` 유지
  (`MAPLE_INFERENCE_URL=http://localhost:8110`, `MONGO_URI=mongodb://localhost:27017`)
- 사설 IP 확인: `ifconfig | grep ens224 -A1`

---

## 디렉터리 구조

```
maple-model-execution-server/
│
├── main.py                        # 추론 게이트웨이 (/infer + /infer/v2)
├── Dockerfile                     # 게이트웨이 컨테이너
├── requirements.txt
├── docker-compose.yml             # 게이트웨이 서비스 (inference-gateway)
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
├── AI_Models/                     # inference.py + 모델 소스, 진료과별 구조 (volume mount)
│   ├── Manual.md                  # 연구원용 모델 제출 가이드
│   ├── Neurology/                 # 신경과
│   │   ├── BraTS2020_T1_UNet3D/
│   │   ├── BraTS2020_T1ce_UNet3D/
│   │   ├── BraTS2020_T2_UNet3D/
│   │   └── BraTS2020_FLAIR_UNet3D/
│   ├── Pulmonology/               # 호흡기내과
│   │   ├── ChestXray14_Multilabel_Classification/
│   │   └── RSNA_Pneumonia_YOLO26x/
│   ├── Ophthalmology/             # 안과
│   ├── Pathology/                 # 병리과
│   ├── Cardiology/                # 심장내과
│   ├── Dermatology/               # 피부과
│   ├── Gastroenterology/          # 소화기내과
│   ├── Orthopedics/               # 정형외과
│   └── Obstetrics/                # 산부인과
│
├── inputs/                        # 추론 입력 파일 (volume mount, 내용물 gitignore)
├── outputs/                       # 추론 출력 파일 (volume mount, 내용물 gitignore)
└── docs/
    ├── runtime_architecture.md
    ├── runtime_migration_plan.md
    └── model_runtime_inventory.md
```

---

## 빠른 시작

### 요구 사항 (A100 서버)

- NVIDIA GPU + 드라이버 (검증 환경: **A100-SXM4-80GB**, driver 535.183.06)
- Docker 29.x + Compose v2, **nvidia-container-toolkit** (runtime `nvidia` 등록 필요)
- 디스크 여유 ≥ 45GB (runtime-basic/medical/nnunet 이미지가 각 ~30GB, NV PyTorch base 공유)
- `docker` 그룹 권한 (`sudo usermod -aG docker $USER` 후 재로그인)

### 빌드 및 실행

```bash
# 1단계: runtime-basic 먼저 빌드 (medical, nnunet이 의존)
docker compose -f docker-compose.runtime.yml build runtime-basic

# 2단계: 전체 빌드 및 실행 (게이트웨이 + runtime 컨테이너)
docker compose -f docker-compose.yml -f docker-compose.runtime.yml up -d --build
```

> `runtime-nnunet`을 사용하는 모델이 아직 없으면 빌드에서 제외해 디스크/시간을 아낄 수 있습니다.

### 헬스 체크

```bash
curl http://localhost:8110/health
# {"status": "ok", "service": "inference-gateway"}

# runtime 컨테이너 헬스 체크 (등록된 모델 목록 포함)
curl http://localhost:9021/health
# {"status": "ok", "service": "maple-runtime-server", "models": [...]}
```

---

## 등록된 모델

| 진료과 | model_name | runtime | 입력 | result_type |
|--------|------------|---------|------|-------------|
| Neurology | `BraTS2020_T1_UNet3D` | runtime-medical | `.nii.gz` | `segmentation_overlay`, `3d_overlay` |
| Neurology | `BraTS2020_T1ce_UNet3D` | runtime-medical | `.nii.gz` | `segmentation_overlay`, `3d_overlay` |
| Neurology | `BraTS2020_T2_UNet3D` | runtime-medical | `.nii.gz` | `segmentation_overlay`, `3d_overlay` |
| Neurology | `BraTS2020_FLAIR_UNet3D` | runtime-medical | `.nii.gz` | `segmentation_overlay`, `3d_overlay` |
| Pulmonology | `ChestXray14_Multilabel_Classification` | runtime-medical | `.png/.jpg` | `gradcam_overlay`, `classification_probabilities` |
| Pulmonology | `RSNA_Pneumonia_YOLO26x` | runtime-yolo | `.dcm` | `bbox_overlay`, `detection_predictions` |

> **검증 상태 (A100, 2026-07):** 6개 모델 전부 `POST /infer/v2` end-to-end 통과 — 유효 PNG 출력 확인.
> BraTS 4종은 `images_b64`(4장)+`output_files`, ChestXray14/RSNA는 `image_b64`(단수)+`output_file`로 반환합니다.

---

## 컨테이너 구성

| 컨테이너 | 포트 | Base Image | 이미지 크기 | 탑재 라이브러리 |
|---------|------|-----------|:----------:|----------------|
| 추론 게이트웨이 (`inference-gateway`) | 8110 | python:3.11-slim | 232MB | FastAPI, httpx |
| `runtime-basic` | 9020 | nvcr.io/nvidia/pytorch:25.12-py3 | 30.3GB | nibabel, scipy, scikit-image |
| `runtime-medical` | 9021 | runtime-basic | 30.9GB | + MONAI 1.5.2, TorchXRayVision 1.4.0, opencv-python-headless |
| `runtime-yolo` | 9022 | python:3.12-slim | 9.35GB | ultralytics, pydicom, opencv |
| `runtime-nnunet` | 9023 | runtime-basic | 30.9GB | + nnunetv2 |

> 이미지 크기는 A100 서버 빌드 기준(GPU runtime 3종은 NV PyTorch base 레이어를 공유하므로 실제 디스크 점유는 합산보다 작음).

---

## API

### `GET /health`

```json
{"status": "ok", "service": "inference-gateway"}
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

### 추론 게이트웨이

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

새 모델을 `AI_Models/{진료과}/{모델명}/`에 추가할 때는 [AI_Models/Manual.md](AI_Models/Manual.md)를 참고하세요.  
런타임 아키텍처 상세는 [docs/runtime_architecture.md](docs/runtime_architecture.md) 참고.

---

## 의존성

### 추론 게이트웨이

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
