# RSNA_Pneumonia_Detection

## 개요
- **진료과**: Radiology
- **모델 타입**: standalone
- **task_type**: bbox detection
- **result_type**: image
- **output_image_role**: bbox_overlay

## 대상 질환
- 폐렴성 폐 혼탁 (pneumonia-related lung opacity)
- RSNA Pneumonia Detection Challenge (Stage 2) 데이터 기반 학습

## 입력 데이터
| 항목 | 타입 | 설명 |
|------|------|------|
| input_data | dicom | 흉부 정면 X-ray DICOM 파일 경로 |

## 출력
- **출력 타입**: image
- **output_image_role**: bbox_overlay
- **반환**: `(result_image, predictions)`
  - `result_image`: `np.ndarray (H, W, 3)` uint8 RGB — bbox overlay 이미지. 탐지 없으면 원본 반환
  - `predictions`: `list[dict]` — 탐지된 box별 정보

`predictions` 항목:
```python
{
    "x1": int, "y1": int, "x2": int, "y2": int,  # xyxy 좌표
    "conf": float,       # confidence score
    "pred": 1,
    "pred_name": "pneumonia_opacity"
}
```

## 모델 파일
| 파일 | 설명 |
|------|------|
| checkpoint/best.pt | YOLO26x fine-tuned weights (RSNA Stage 2) |

## 모델 상세
| 항목 | 값 |
|------|----|
| 아키텍처 | YOLO26x (ultralytics) |
| 학습 데이터 | RSNA Pneumonia Detection Challenge Stage 2 |
| 이미지 크기 | 1024×1024 |
| Confidence threshold | 0.15 (Youden index 기준) |
| mAP50 (test) | 0.414 |
| Precision / Recall | 0.499 / 0.468 |

## 전처리
1. DICOM → pixel_array 추출
2. p1–p99 percentile clipping → uint8 정규화
3. MONOCHROME1 반전 (해당 시)
4. Histogram equalization (PIL ImageOps.equalize)
5. Grayscale → RGB 3채널 복제

## 후처리
- NMS: YOLO 내장 (iou=0.5)
- Dedup: overlap fraction ≥ 0.5인 하위 confidence box 제거

## 실행 예시
```python
from inference import main

result_image, predictions = main(
    input_data="sample_data/sample_01.dcm",
    model_path="checkpoint/best.pt"
)
```

## 비고
- RSNA DICOM은 비압축 포맷으로 pylibjpeg 불필요
- GPU 없이 CPU만으로 동작 가능 (추론 속도 저하)
- `opencv-python-headless` 사용 (컨테이너 환경)
