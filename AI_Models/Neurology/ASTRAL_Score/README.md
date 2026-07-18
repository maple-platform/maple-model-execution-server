# ASTRAL_Score

결정론적 정수 스코어 (Ntaios 2012). 급성 허혈뇌졸중 90일 불량예후(mRS>2) 위험.

- **runtime**: runtime-basic (CPU) · **required_data**: `["csv"]` · **result_type**: `text`
- **입력 컬럼**: `age, nihss, onset_to_admission_hours, visual_field_defect(0/1), glucose_mmol, decreased_consciousness(0/1)`
- **출력 컬럼**: `pred, pred_name, astral_score`
- **점수**: age 0.2/년(=1/5년) + NIHSS(1/점) + 발병>3h(2) + 시야결손(2) + 혈당>7.3 or <3.7mmol(1) + 의식저하(3)
- **가중치**: 없음. 점수표는 `checkpoint/coefficients.json`.

> 혈당 단위는 mmol/L (mg/dL ÷ 18). risk_bands 는 표시용 근사 구간이며, 검증된 정량 앵커는 **ASTRAL≈31 → 불량예후 ~50%**. 배포 전 원 논문 대비 검증 권장.

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
