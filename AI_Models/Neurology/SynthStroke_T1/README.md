# SynthStroke_T1

## 개요
- **진료과**: Neurology
- **모델 타입**: standalone
- **task_type**: segmentation
- **result_type**: `["segmentation_overlay", "3d_overlay", "detection_predictions"]`
- **output_image_role**: segmentation_overlay
- **runtime**: runtime-medical
- **업스트림**: [liamchalcroft/SynthStroke](https://github.com/liamchalcroft/SynthStroke) (MIT), MELBA 2025

## 대상 질환
- 만성/허혈성 뇌졸중 병변(stroke lesion) 분할

## 입력 데이터
| 항목 | 타입 | 설명 |
|------|------|------|
| input_data | str (파일 경로) | 단일 T1 강조 NIfTI (`.nii.gz` / `.nii`) |

## 출력
- `main(input_data, model_path)` 은 `(images, labels, predictions)` 3-tuple 반환.
- **images**: `list[np.ndarray (H,W,3) uint8 RGB]`, 순서는 **labels** 와 일치.

| 인덱스 | label | 내용 |
|--------|-------|------|
| `[0]` | `3d_overlay` | 뇌 표면 + 병변 3D 렌더링 (vedo 없으면 생략되어 이 항목이 빠짐) |
| `[-1]` | `lesion_overlay` | 병변이 가장 많은 axial slice의 병변 오버레이(빨강) |

> vedo/vtk-osmesa 가 런타임에 없으면 `3d_overlay` 는 자동 생략되고 `lesion_overlay` 만 반환됩니다. runner 가 반환 길이에 맞춰 labels 를 그대로 사용하므로 안전합니다.

- **predictions**: `list[dict]` — connected-component 기반 병변별 통계.
  ```json
  {"lesion_id": 1, "volume_ml": 12.34, "voxels": 12340,
   "centroid_vox": [z, y, x], "bbox_vox": [z0,y0,x0,z1,y1,x1]}
  ```
  1mm isotropic 리샘플 후 계산되므로 voxel 1개 = 0.001 mL. 이 값은 seg→예측 파이프라인
  (예: ICH Score 의 혈종 부피 입력 등)의 원천으로 사용할 수 있습니다.

## 모델 파일
| 파일 | 설명 |
|------|------|
| `checkpoint/model.safetensors` | SynthStroke 가중치 (HF `liamchalcroft/synthstroke-synth-plus`) |
| `checkpoint/config.json` | ModelConfig (in/out channels, UNet 아키텍처) |

가중치는 git 과 별개로 서버에서 다운로드합니다 (아래 참고).

## 모델 상세
| 항목 | 값 |
|------|----|
| 아키텍처 | MONAI 3D UNet (channels 32-64-128-256-320-320, strides 2×5, num_res_units 1, INSTANCE norm) |
| 입력 채널 | 1 (T1) |
| 출력 클래스 | synth-plus: 6-class (배경/GM/WM/부분용적/CSF/**Stroke=5**) · baseline: 2-class(**Stroke=1**) |
| 추론 | sliding-window(patch 128, gaussian, overlap 0.5) + TTA(flip 8종 평균) |
| 전처리 | RAS 정렬 → 1mm 등방 리샘플 → HistogramNormalize → 채널별 NormalizeIntensity |

## 전처리
업스트림 `SynthStrokeModel.preprocess_image` 를 그대로 복제 (huggingface_hub 의존 제거).

## 실행 예시
```python
from inference import main

images, labels, predictions = main(
    input_data="sample_data/sample_01.nii.gz",
    model_path="checkpoint",   # 폴더 또는 checkpoint/model.safetensors
)
```

## 가중치 확보 (서버에서)
```bash
cd AI_Models/Neurology/SynthStroke_T1/checkpoint
huggingface-cli download liamchalcroft/synthstroke-synth-plus \
    model.safetensors config.json --local-dir .
# 대안(2-class, 더 단순): liamchalcroft/synthstroke-baseline
```

## 비고
- self-contained: `huggingface_hub` 없이 `safetensors` + `config.json` 만으로 로드 → 서버 인터넷 불필요.
- `result_type` 에 `detection_predictions` 를 segmentation 과 함께 넣은 신규 조합 — 관리자 합의 권장(NEUROL 작업노트 §4.3).
