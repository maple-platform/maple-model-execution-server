# Runtime Migration Plan

> 작성일: 2026-05-25  
> 현재 단계: 1차 Inventory 완료 / 리팩토링 미착수

---

## 1. 현재 구조의 문제점

### 1-1. Image 용량 과다
- BraTS 4개 모델이 각각 `nvcr.io/nvidia/pytorch:25.12-py3` (~30GB)를 base로 사용
- Docker가 레이어를 공유하므로 실제 디스크 점유는 적으나, Docker Hub push/pull 시 중복 전송 발생
- 모델별 image가 아니라 runtime base image + 경량 model layer로 분리하면 관리 효율 향상

### 1-2. 모델마다 개별 Dockerfile 관리
- 현재 7개 서비스 = 7개 Dockerfile
- BraTS 4개는 내용이 거의 동일 (copy-paste 상태)
- 런타임 환경이 바뀌면 모든 Dockerfile을 개별 수정해야 함

### 1-3. requirements.txt vs 실제 컨테이너 버전 불일치
- BraTS requirements.txt: `torch==2.12.0+cu130`
- 실제 컨테이너 (NV 25.12 base): `torch 2.10.0a0+b4e4ee81d3.nv25.12`
- 현재 Dockerfile에서 requirements.txt의 torch를 재설치하지 않아 NV 기본 torch를 사용 중
- 모델이 torch 2.12 기능에 의존하는 경우 예상치 못한 동작 가능

### 1-4. 가중치 파일이 image에 포함됨
- BraTS, RSNA 모델은 `COPY . .`로 checkpoint를 image에 bake-in
- 동시에 docker-compose volume mount로도 전달 → 중복
- 가중치 업데이트 시 image 재빌드 필요

### 1-5. ChestXray14 오프라인 취약
- 가중치를 런타임에 인터넷에서 다운로드
- 네트워크 없는 환경에서 첫 실행 실패

---

## 2. 모델별 Dependency/Version Inventory 요약

| Model | Base Image | Torch | CUDA | 주요 Framework | Weight 위치 |
|-------|-----------|-------|------|---------------|------------|
| maple-inference | python:3.11-slim | 없음 | 없음 | FastAPI | 해당없음 |
| brats-t1/t1ce/t2/flair | nvcr.io/nvidia/pytorch:25.12-py3 | 2.10.0a0.nv25.12 | 13.1 | MONAI 1.5.2 | image 내 + volume |
| chestxray14 | nvcr.io/nvidia/pytorch:25.12-py3 | 2.10.0a0.nv25.12 | 13.1 | TorchXRayVision 1.4.0 | 런타임 다운로드 |
| rsna-pneumonia | python:3.12-slim | ultralytics 내장 | optional | ultralytics 8.4.51 | image 내 + volume |

> 상세 내용은 [model_runtime_inventory.md](./model_runtime_inventory.md) 참조

---

## 3. 추천 Runtime Group

```
runtime-nvpytorch2512-monai   (BraTS 4종)
  └─ base: nvcr.io/nvidia/pytorch:25.12-py3
  └─ add: MONAI 1.5.2, nibabel, scipy, scikit-image, Pillow

runtime-nvpytorch2512-xray    (ChestXray14)
  └─ base: nvcr.io/nvidia/pytorch:25.12-py3
  └─ add: TorchXRayVision 1.4.0 (--no-deps), opencv, scikit-image

runtime-py312-cpu-yolo        (RSNA Pneumonia)
  └─ base: python:3.12-slim
  └─ add: ultralytics, pydicom, opencv

runtime-py311-slim-gateway    (Gateway)
  └─ base: python:3.11-slim
  └─ add: FastAPI, uvicorn, httpx
```

**monai와 torchxrayvision 분리 이유**:
- 공존 가능하나 목적과 의존성 트리가 달라 variant로 유지
- 향후 버전 업 시 충돌 리스크 최소화

**YOLO 분리 이유**:
- ultralytics가 자체 torch 버전을 관리
- CPU 동작 가능 → NV 30GB 이미지 불필요

---

## 4. Weight/Config/Runner 외부화 전략

### 현재
```
image = base + pip packages + checkpoint/*.pt + inference.py + server.py
```

### 목표
```
image = base + pip packages  (runtime image, 재사용 가능)
volume = checkpoint/*.pt     (모델 가중치, image 외부)
volume = AI_Models/*/        (inference.py, server.py)
```

### 구체적 방법
1. Dockerfile에서 `COPY . .` → `COPY server.py inference.py ./` + `COPY src/ ./src/`로 변경
2. checkpoint는 COPY 하지 않고 volume mount만 사용
3. `MODEL_PATH` 환경변수로 가중치 경로 주입 (이미 구현됨)
4. ChestXray14는 pre-cache script로 가중치 사전 다운로드

---

## 5. 다음 단계에서 만들 파일 목록

```
Dockerfile.runtime-nvpytorch2512-monai    # BraTS 공용 runtime base
Dockerfile.runtime-nvpytorch2512-xray     # ChestXray14 runtime base
Dockerfile.runtime-py312-cpu-yolo         # RSNA runtime base
Dockerfile.runtime-py311-slim-gateway     # Gateway (현재와 동일)

AI_Models/Neurology/BraTS2020_T1_UNet3D/Dockerfile      # runtime base FROM 사용
AI_Models/Neurology/BraTS2020_T1ce_UNet3D/Dockerfile    # runtime base FROM 사용
AI_Models/Neurology/BraTS2020_T2_UNet3D/Dockerfile      # runtime base FROM 사용
AI_Models/Neurology/BraTS2020_FLAIR_UNet3D/Dockerfile   # runtime base FROM 사용
AI_Models/Radiology/ChestXray14_Multilabel_Classification/Dockerfile
AI_Models/Radiology/RSNA_Pneumonia_YOLO26x/Dockerfile

scripts/precache_chestxray14_weights.sh   # TorchXRayVision 가중치 사전 다운로드
```

---

## 6. Migration Risk

| 리스크 | 수준 | 설명 |
|--------|------|------|
| BraTS torch 버전 불일치 | **HIGH** | requirements.txt(2.12.0+cu130) vs NV container(2.10.0a0.nv25.12). 모델 정확도/동작에 영향 가능 |
| MONAI 버전 호환성 | MEDIUM | MONAI 1.5.2가 NV container의 numpy 2.4.6과 호환되는지 확인 필요 |
| ChestXray14 오프라인 | MEDIUM | 첫 실행 시 인터넷 필요. pre-cache 전략으로 해결 가능 |
| 가중치 외부화 후 경로 변경 | LOW | `MODEL_PATH` 환경변수 이미 사용 중이라 docker-compose 수정만 필요 |
| ultralytics torch 충돌 | LOW | python:3.12-slim 기반으로 격리되어 있어 충돌 없음 |

---

## 7. Backward Compatibility 전략

- `docker-compose.yml` 서비스명/포트 유지 (외부 인터페이스 변경 없음)
- `POST /infer` API 스펙 유지
- 컨테이너 내부 `POST /run` 인터페이스 유지
- 가중치 외부화는 volume mount 방식이므로 기존 `MODEL_PATH` 환경변수 그대로 사용
- runtime base image 교체 후 `scripts/check_image_versions.sh`로 버전 검증 필수

---

## 8. 우선순위 제안

1. **즉시**: BraTS torch 버전 불일치 확인 — `scripts/check_image_versions.sh maple/brats-t1:latest`로 실제 버전 확인
2. **단기**: 가중치 외부화 (COPY → volume only) — image 재빌드 시간 단축
3. **중기**: runtime base image 분리 (`Dockerfile.runtime-*`) — Dockerfile 중복 제거
4. **장기**: ChestXray14 pre-cache 전략 — 오프라인 배포 대응
