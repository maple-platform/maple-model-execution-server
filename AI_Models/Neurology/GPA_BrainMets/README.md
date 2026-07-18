# GPA_BrainMets

결정론적 예후 지수 (Sperduto 2008). 뇌전이 환자 중앙 생존 예측.

- **runtime**: runtime-basic (CPU) · **required_data**: `["csv"]` · **result_type**: `text`
- **입력 컬럼**: `age, kps, num_cns_mets, extracranial_mets(0/1)`
- **출력 컬럼**: `pred, pred_name, gpa_score(0-4), median_survival_months`
- **파이프라인 연결**: `num_cns_mets` 를 뇌전이 detection 모델(AURORA 등) 결과에서 주입 가능.
- **가중치**: 없음. 점수표는 `checkpoint/coefficients.json`.

```python
import pandas as pd
from inference import main
main(pd.read_csv("sample_data/sample_01.csv"), "checkpoint/coefficients.json")
```
> 원 논문(basic GPA) 대비 검증 권장. 질환별 DS-GPA 는 별도 계수 필요.
