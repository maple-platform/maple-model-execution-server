# CAIDE_Dementia

결정론적 위험 스코어 (Kivipelto 2006). 중년 위험인자 → 20년 치매 위험.

- **runtime**: runtime-basic (CPU) · **required_data**: `["csv"]` · **result_type**: `text`
- **입력 컬럼**: `age, education_years, sex, sbp, bmi, cholesterol_mmol, physically_active(0/1)`
- **출력 컬럼**: `pred, pred_name, caide_score(0-15), dementia_risk_20yr_pct`
- **파이프라인 연결**: WMH seg(wmh_seg 등)의 소혈관 부담 지표와 상호 보완.
- **가중치**: 없음. 점수표는 `checkpoint/coefficients.json`. (APOE 포함 시 0-18)

> 총콜레스테롤 단위 mmol/L (mg/dL ÷ 38.67). 배포 전 원 논문 대비 점수 임계값 검증 권장.

## 실행 예시

```python
import pandas as pd
from inference import main

result = main(
    pd.read_csv("sample_data/sample_01.csv"),
    "checkpoint/coefficients.json",
)
print(result)
```
