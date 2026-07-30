# Ophthalmology models — sample data & quick test guide

Four models are currently submitted under this department. Each folder is a
self-contained Maple package (see `../Manual.md`); this file is just a quick
index of what sample images exist and the exact `main()` call to try each
model once its checkpoint is in place.

**Checkpoints are not in this repo** (see each model's `checkpoint/README.md`
for the official download link, verified size, and hash). Everything below
assumes you've already placed the checkpoint under `checkpoint/` in that
model's folder.

| Model | Disease / body part | Input | Sample images |
|---|---|---|---|
| `RETFound_DR_APTOS2019_GradCAM` | Diabetic retinopathy (posterior) | 1 fundus photo | `sample_data/fundus_diabetic_retinopathy_ccby4.png` |
| `DeepSeeNet_AMD_SimplifiedScore` | AMD (posterior) | 2 fundus photos (left + right eye) | `sample_data/left_eye.jpg`, `right_eye.jpg` |
| `RETFound_Glaucoma_PAPILA` | Glaucoma (posterior) | 1 fundus photo | `sample_data/papila_test_cglaucoma_RET102OD.jpg`, `papila_test_anormal_RET199OD.jpg` |
| `DeepLensNet_Cataract_Severity` | Cataract (anterior) | up to 3 anterior-segment photos | `sample_data/sample_01_ns_slitlamp.jpg`, `sample_01_cortical_retro.jpg`, `sample_01_psc_retro.jpg` |

## 1. RETFound_DR_APTOS2019_GradCAM

Single image path in, `(gradcam_overlay, predictions[5])` out.

```python
import sys
sys.path.insert(0, "AI_Models/Ophthalmology/RETFound_DR_APTOS2019_GradCAM")
from inference import main

overlay, predictions = main(
    "AI_Models/Ophthalmology/RETFound_DR_APTOS2019_GradCAM/sample_data/fundus_diabetic_retinopathy_ccby4.png",
    "AI_Models/Ophthalmology/RETFound_DR_APTOS2019_GradCAM/checkpoint/checkpoint-best.pth",
)
```

## 2. DeepSeeNet_AMD_SimplifiedScore

Both eyes required (dict input) -- see the model's own `README.md` for the
"Glaucoma"-style limitation note that does **not** apply here; this one is
well-behaved.

```python
import sys
sys.path.insert(0, "AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore")
from inference import main

result_df = main(
    {
        "left_eye": "AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/sample_data/left_eye.jpg",
        "right_eye": "AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/sample_data/right_eye.jpg",
    },
    "AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/checkpoint",
)
```

## 3. RETFound_Glaucoma_PAPILA

Single image path in, `(gradcam_overlay, predictions[3])` out. ⚠️ Read
`RETFound_Glaucoma_PAPILA/README.md`'s "Known limitation" section before
interpreting the output -- the top-1 label almost never says "Glaucoma"; the
per-class probabilities are the meaningful signal.

```python
import sys
sys.path.insert(0, "AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA")
from inference import main

overlay, predictions = main(
    "AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA/sample_data/papila_test_cglaucoma_RET102OD.jpg",
    "AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA/checkpoint/checkpoint-best.pth",
)
```

## 4. DeepLensNet_Cataract_Severity

Any subset of the three photo keys (dict input); all three are provided in
`sample_data/`.

```python
import sys
sys.path.insert(0, "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity")
from inference import main

result_df = main(
    {
        "ns_image": "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/sample_data/sample_01_ns_slitlamp.jpg",
        "cortical_image": "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/sample_data/sample_01_cortical_retro.jpg",
        "psc_image": "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/sample_data/sample_01_psc_retro.jpg",
    },
    "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/checkpoint",
)
```

## Docker 및 Gateway 실행

### 지원 상태

모델이 `meta.json`과 `models/*/config.yaml`에 등록되어 있다는 사실과 현재
공용 Docker 이미지에서 실제로 실행 가능하다는 것은 구분해야 합니다.

| 모델 | 등록 Runtime | 현재 공용 Docker 실행 | 이유 |
|---|---|---|---|
| `RETFound_DR_APTOS2019_GradCAM` | `runtime-medical` | 지원 | 현재 PyTorch/timm 계열 |
| `RETFound_Glaucoma_PAPILA` | `runtime-medical` | 지원 대상 | 현재 PyTorch/timm 계열, 대용량 checkpoint 별도 필요 |
| `DeepSeeNet_AMD_SimplifiedScore` | `runtime-medical` | 미지원 | Python 3.6, TF 1.15/Keras 2.2 전용 이미지 필요 |
| `DeepLensNet_Cataract_Severity` | `runtime-medical` | 미지원 | Python 3.8, TF 2.3/Keras 2.4 전용 이미지 필요 |

DeepSeeNet과 DeepLensNet을 현재 `runtime-medical` 이미지에 억지로 함께
설치하면 TensorFlow, Keras, NumPy 및 Python 버전 충돌이 발생한다. 두
모델은 각각 별도의 legacy TensorFlow Runtime을 만들거나, 모델을 현대
프레임워크로 검증된 방식으로 변환한 후 공용 Runtime에 포함해야 한다.

### 공통 볼륨 구조

두 compose 파일을 함께 사용하면 다음 경로가 컨테이너에 마운트된다.

| 호스트 | 컨테이너 | 용도 |
|---|---|---|
| `./models` | `/app/models` | Runtime adapter와 `config.yaml` |
| `./AI_Models` | `/app/AI_Models` | inference 코드와 checkpoint |
| `./inputs` | `/app/inputs` | 입력 이미지와 manifest |
| `./outputs` | `/app/outputs` | JSON 및 Grad-CAM 결과 |

checkpoint는 이미지에 복사하지 않고 호스트의 각 모델
`checkpoint/` 폴더에 둔다. `AI_Models`는 read-only mount이므로 실행 중
컨테이너가 checkpoint를 내려받거나 수정하도록 구성하면 안 된다.

### 컨테이너 시작과 상태 점검

저장소 루트에서 실행한다.

```bash
docker compose -f docker-compose.yml -f docker-compose.runtime.yml build
docker compose -f docker-compose.yml -f docker-compose.runtime.yml up -d
docker compose -f docker-compose.yml -f docker-compose.runtime.yml ps
```

서비스별 포트는 Gateway `8110`, basic `9020`, medical `9021`, YOLO
`9022`, nnUNet `9023`이다.

```bash
curl -f http://localhost:8110/health
curl -f http://localhost:8110/ready
curl -f http://localhost:9021/health
curl -sS http://localhost:9021/models
```

`/health`는 프로세스 생존 여부이고 `/ready`는 모델 설정과 모든 Runtime
연결을 함께 검사한다. 배포 트래픽을 열기 전에는 `/ready`가 HTTP 200과
`"status": "ready"`를 반환하는지 확인한다.

### 단일 이미지 모델 요청

이미지를 `inputs/papila.jpg`에 놓은 경우:

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

### 복수 이미지 모델 요청 규약

DeepSeeNet과 DeepLensNet의 `meta.json.required_data`는 임상 원본 파일이
JPG/PNG임을 나타낸다. 하지만 현재 Runtime API는 `input_path` 하나만
받으므로 adapter는 임시로 JSON manifest 경로를 요구한다.

DeepSeeNet:

```json
{
  "left_eye": "left_eye.jpg",
  "right_eye": "right_eye.jpg"
}
```

DeepLensNet:

```json
{
  "ns_image": "ns.jpg",
  "cortical_image": "cortical.jpg",
  "psc_image": "psc.jpg"
}
```

manifest 내부 상대 경로는 manifest가 있는 폴더를 기준으로 해석된다. 따라서
이미지와 manifest를 같은 `inputs/<exam>/` 폴더에 두면 호스트와 컨테이너
경로를 모두 유지할 수 있다. 이 방식은 로컬 검증용 임시 규약이며, 정식
서비스 전에 업로드 API가 `exam_id`, laterality, view type을 포함한 복수
파일 계약을 제공해야 한다.

### 장애 확인

```bash
docker compose -f docker-compose.yml -f docker-compose.runtime.yml logs --tail=200 inference-gateway runtime-medical
```

주요 확인 항목:

- checkpoint 파일명과 컨테이너 내부 경로가 `config.yaml`과 일치하는가
- 요청의 `input_path`가 호스트 경로가 아닌 `/app/inputs/...`인가
- manifest의 좌·우안 및 촬영 방식이 올바르게 매핑됐는가
- `/ready`의 `errors`에 잘못된 모델 설정이나 Runtime URL이 있는가
- GPU 메모리 부족 또는 checkpoint 로딩 실패가 발생했는가

## Sample image provenance at a glance

| File(s) | Source | License |
|---|---|---|
| `fundus_diabetic_retinopathy_ccby4.png` | Wikimedia Commons (smoke-test only, not an APTOS case) | CC BY 4.0 |
| `left_eye.jpg`, `right_eye.jpg` | ncbi-nlp/DeepSeeNet's own demo images | Public domain (US Government Work) |
| `papila_test_*.jpg` | Real PAPILA test-set images, own ground truth | CC BY 4.0 |
| `sample_01_*.jpg` | ncbi/deeplensnet's own test images (subject 1) | No formal license; NCBI research-use-only notice |

Each model's own `sample_data/README.md` has the full attribution and, where
applicable, the expected/reference output for that exact sample.

## Requirements note

Each model pins a **different** Python/framework stack (see each
`requirements.txt`) -- they are not designed to run in the same virtual
environment:

| Model | Python | Framework |
|---|---|---|
| `RETFound_DR_APTOS2019_GradCAM` | 3.11+ | PyTorch 2.5.1 / timm |
| `DeepSeeNet_AMD_SimplifiedScore` | 3.6 | TensorFlow 1.15.5 / Keras 2.2.4 |
| `RETFound_Glaucoma_PAPILA` | 3.11+ | PyTorch 2.5.1 / timm |
| `DeepLensNet_Cataract_Severity` | 3.8 | TensorFlow 2.3.1 / Keras 2.4.3 |
