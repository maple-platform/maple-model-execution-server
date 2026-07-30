# 안과 AI 모델 통합 개요

## 1. 목적

본 문서는 MAPLE Model Execution Server에 새로 통합된 안과 AI 모델의
기능, 시스템 구성, 검증 결과, 운영 제약 및 향후 개선 과제를 보고서 형태로
정리한다. 대상 모델은 다음 세 가지다.

1. `DeepSeeNet_AMD_SimplifiedScore`
2. `DeepLensNet_Cataract_Severity`
3. `RETFound_Glaucoma_PAPILA`

기존 `RETFound_DR_APTOS2019_GradCAM`과 함께 안저 및 전안부 영상 기반의
당뇨망막병증, 황반변성, 녹내장, 백내장 분석 범위를 구성한다.

## 2. 모델별 기능

| 모델 | 임상 대상 | 입력 | 출력 | 프레임워크 |
|---|---|---|---|---|
| DeepSeeNet | 연령관련 황반변성 | 동일 환자의 좌·우 안저 사진 | AREDS 단순 중증도 점수 0–5와 눈별 위험인자 | TensorFlow 1.15 / Keras 2.2 |
| DeepLensNet | 백내장 | 세극등 및 역조명 사진 최대 3장 | 핵경화 등급, 피질 및 후낭하 혼탁 비율 | TensorFlow 2.3 / Keras 2.4 |
| RETFound PAPILA | 녹내장 | 단일 컬러 안저 사진 | 정상·녹내장 의증·녹내장 확률과 Grad-CAM | PyTorch / timm |

DeepSeeNet은 양안을 함께 평가하는 환자 단위 모델이다. 드루젠 크기,
색소 이상, 진행성 AMD 여부를 눈별로 추정한 뒤 AREDS 산식으로 환자
점수를 계산한다. DeepLensNet은 촬영 방식이 다른 최대 세 장의 전안부
사진을 각각의 회귀 모델에 전달한다. RETFound PAPILA는 단일 안저 사진에
대한 3분류 확률과 모델의 주목 영역을 시각화한다.

## 3. MAPLE 통합 구조

각 모델은 연구 코드와 실행 adapter를 분리한다.

```text
AI_Models/Ophthalmology/<model>/
├── inference.py
├── meta.json
├── requirements.txt
├── checkpoint/
└── sample_data/

models/<model>/
├── config.yaml
├── runner.py
└── README.md
```

`meta.json`은 모델 탐색과 UI에 필요한 임상 메타데이터를 제공한다.
`config.yaml`은 Gateway가 사용할 Runtime, runner, inference 및 checkpoint
경로를 정의한다. `runner.py`는 연구 코드의 반환값을 플랫폼 공통 JSON
응답으로 변환한다.

요청 흐름은 다음과 같다.

```text
Client
  → Inference Gateway :8110
  → model config 검증 및 Runtime 선택
  → runtime-medical :8000
  → model runner
  → research inference
  → 표준 JSON/이미지 응답
```

Gateway의 `GET /ready`는 전체 모델 설정과 Runtime health endpoint를
검사한다. 설정 누락이나 접근 불가능한 경로가 있으면 HTTP 503과 구조화된
오류 목록을 반환하므로 배포 전 점검 지점으로 사용할 수 있다.

## 4. 입력 데이터 계약

RETFound PAPILA는 기존 플랫폼 계약과 동일하게 단일 이미지 경로를
사용한다. 반면 DeepSeeNet과 DeepLensNet은 한 검사에 여러 영상이
필요하다. 현재 Gateway 요청에는 `input_path`가 하나뿐이므로 두 adapter는
임시 JSON manifest를 사용한다.

DeepSeeNet manifest:

```json
{
  "left_eye": "left_eye.jpg",
  "right_eye": "right_eye.jpg"
}
```

DeepLensNet manifest:

```json
{
  "ns_image": "ns.jpg",
  "cortical_image": "cortical.jpg",
  "psc_image": "psc.jpg"
}
```

이 구조는 로컬 실행과 API 연결을 가능하게 하지만 정식 플랫폼 계약은
아니다. `meta.json.required_data`는 JPG/PNG 원본 형식을 나타내는 반면,
실제 `input_path`는 JSON을 가리킨다는 차이가 있다. 향후 복수 파일 업로드
API에는 최소한 다음 정보가 필요하다.

- 검사 또는 환자 단위 그룹 식별자
- 좌안·우안 구분
- 세극등·전방 역조명·후방 역조명 촬영 방식
- 파일별 무결성 및 허용 확장자
- 누락 가능한 view와 필수 view 규칙

## 5. Docker 배포 현황

세 모델의 metadata는 `runtime-medical`을 가리키며 service URL은
`http://runtime-medical:8000`으로 동기화되어 있다. 그러나 등록 상태와
실행 가능 상태는 동일하지 않다.

RETFound PAPILA는 현재 공용 PyTorch 기반 의료 Runtime과 같은 계열이다.
checkpoint를 올바른 위치에 배치하고 Runtime의 실제 timm/PyTorch
호환성을 smoke test한 뒤 운영할 수 있다.

DeepSeeNet과 DeepLensNet은 서로 다른 legacy TensorFlow 환경을 요구한다.
DeepSeeNet은 Python 3.6과 TensorFlow 1.15, DeepLensNet은 Python 3.8과
TensorFlow 2.3을 기준으로 검증됐다. 현재 공용 Runtime에 두 환경을 함께
설치하면 Python wheel 부재와 TensorFlow/Keras/NumPy 충돌이 발생하므로
운영 배포 대상으로 간주할 수 없다.

권장 배포안은 다음 두 가지다.

1. 모델별 전용 legacy Runtime 이미지를 만들고 Gateway의 지원 Runtime
   목록에 등록한다.
2. 모델을 ONNX 또는 현대 TensorFlow/PyTorch로 변환한 뒤 원본과의 수치
   동등성을 검증하여 공용 Runtime에 통합한다.

첫 번째 방법은 재현성이 높지만 이미지 수와 운영 복잡도가 늘어난다. 두
번째 방법은 운영 효율이 높지만 변환 오차와 전처리 차이에 대한 재검증이
필수다.

## 6. 검증 결과와 해석상 주의점

### DeepSeeNet

공식 데모 영상과 비교했을 때 드루젠 및 진행성 AMD 결과는 일치했다.
좌안 색소 이상은 확률이 약 48.9% 대 51.1%로 경계에 매우 가까워 기준
실행과 반대쪽 클래스로 결정됐다. 이는 작은 전처리 또는 라이브러리 차이가
경계 사례의 최종 점수를 바꿀 수 있음을 의미한다.

### DeepLensNet

공식 테스트 영상에 대한 세 회귀 점수는 upstream 결과와 약 0.3% 이내로
일치했다. 회귀 결과는 진단명이 아니라 연속형 정량값이며 촬영 방식이
잘못 매핑되면 서로 다른 회귀기에 입력되므로 임상적으로 무의미한 결과가
생길 수 있다.

### RETFound PAPILA

공식 test split 98장에 대한 로컬 검증에서 `Glaucoma` 클래스는 확진
녹내장 영상을 포함해 top-1이 되지 않았다. 클래스 순서는 확인됐고 실제
녹내장군에서 녹내장 클래스 평균 확률이 더 높았지만 절대 크기가 다른
클래스를 넘지 못했다. 따라서 top-1 결과를 선별 판정처럼 노출해서는 안
되며 세 확률 전체와 calibration 제한을 함께 표시해야 한다. Grad-CAM도
병변 분할이나 인과적 근거가 아니라 모델 주목 영역의 보조 시각화다.

## 7. 라이선스 및 임상 거버넌스

| 모델 | 사용 조건 |
|---|---|
| DeepSeeNet | 미국 정부 저작물 기반이나 NCBI 고지상 연구·비상업 용도 |
| DeepLensNet | upstream LICENSE 부재, NCBI 연구·비상업 용도 고지 |
| RETFound PAPILA | RETFound CC BY-NC 4.0, PAPILA 데이터 CC BY 4.0 |

세 모델 모두 의료기기로 승인된 자율 진단 시스템이 아니다. 상용 사용,
환자 진료 의사결정 또는 자동 선별에 사용하기 전에 별도의 라이선스 확인,
외부 검증, 편향 평가, calibration, 임상 책임자 승인 및 감사 가능한 결과
기록이 필요하다.

## 8. 운영 위험과 개선 과제

우선순위가 높은 과제는 다음과 같다.

1. DeepSeeNet과 DeepLensNet용 실행 Runtime을 구현하거나 현대
   프레임워크로 변환한다.
2. 복수 이미지 검사에 대한 플랫폼 공통 입력 스키마를 정의한다.
3. 좌·우안 및 촬영 view를 서버에서 검증할 수 있는 metadata 계약을
   추가한다.
4. RETFound PAPILA에 대해 class calibration 또는 재학습을 검토하고,
   UI가 전체 확률과 제한사항을 강제로 표시하도록 한다.
5. checkpoint의 SHA-256, 파일 크기, 출처 및 설치 이력을 배포 단계에서
   자동 검증한다.
6. 모델별 대표 sample을 사용하는 container smoke test를 CI에 추가한다.
7. 데이터 분포, 인종·연령·장비별 성능과 실패 사례를 별도 임상 검증
   보고서로 관리한다.

## 9. 결론

이번 통합으로 MAPLE의 안과 분석 범위는 단일 안저 분류를 넘어 양안 기반
AMD 중증도, 다중 전안부 영상 기반 백내장 정량화, 녹내장 확률 및 설명
시각화까지 확장됐다. 동시에 복수 이미지 입력 계약, legacy TensorFlow
격리, 확률 calibration이라는 운영 과제가 확인됐다.

현재 단계에서는 RETFound PAPILA가 공용 Runtime 통합에 가장 가깝고,
DeepSeeNet과 DeepLensNet은 연구 코드 및 adapter 검증은 완료됐으나 전용
실행 환경이 필요한 상태다. 운영 전환은 단순 등록 여부가 아니라 framework
재현성, 입력 무결성, 임상적 출력 해석 및 라이선스 요건을 함께 충족하는
방식으로 진행해야 한다.
