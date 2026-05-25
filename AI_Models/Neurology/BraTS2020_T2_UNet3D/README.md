# BraTS2020 Brain Tumor Segmentation — T2

## 개요
- **진료과**: Neurology / Radiology
- **모달리티**: T2
- **모델 타입**: standalone
- **task_type**: segmentation
- **result_type**: image
- **output_image_role**: segmentation_overlay

## 대상 질환
- 뇌종양 (glioma, glioblastoma)
- BraTS2021 Training Data 기반 학습

## 입력 데이터
| 항목 | 타입 | 설명 |
|------|------|------|
| input_data | nifti | T2 MRI NIfTI 파일 (.nii.gz) |

## 출력
- **반환**: `list[np.ndarray (H, W, 3) uint8 RGB]`

| 인덱스 | 설명 |
|--------|------|
| `[0]` | 3D isosurface 렌더링 (뇌 + 종양) |
| `[1]` | Axial overlay — 3개 영역 합성 |
| `[2]` | WT (Whole Tumor) — 빨강 |
| `[3]` | TC (Tumor Core) — 초록 |
| `[4]` | ET (Enhancing Tumor) — 파랑 |

## 모델 파일
| 파일 | 설명 |
|------|------|
| checkpoint/best.pt | T2 모달리티 학습 가중치 |

## 모델 상세
| 항목 | 값 |
|------|----|
| 아키텍처 | MONAI UNet3D (channels: 16-32-64-128-256) |
| 입력 채널 | 1 (단일 모달리티) |
| 출력 채널 | 3 (WT / TC / ET) |
| Threshold | 0.5 |

## 실행 예시
```python
from inference import main

results = main(
    input_data="sample_data/sample_01_t2.nii.gz",
    model_path="checkpoint/best.pt"
)
```
