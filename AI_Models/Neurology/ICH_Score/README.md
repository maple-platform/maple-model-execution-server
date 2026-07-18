# ICH_Score

결정론적 임상 스코어 (Hemphill 2001). ICH 30일 사망 위험.

- **runtime**: runtime-basic (CPU) · **required_data**: `["csv"]` · **result_type**: `text`
- **입력 컬럼**: `gcs, age, ich_volume_ml, ivh(0/1), infratentorial(0/1)`
- **출력 컬럼**: `pred, pred_name, ich_score(0-6), mortality_30day_pct`
- **파이프라인 연결**: `ich_volume_ml`, `ivh` 를 출혈 seg 모델(DeepBleed 등) 결과에서 주입 가능.
- **가중치**: 학습 가중치 없음. 점수표를 `checkpoint/coefficients.json` 으로 보관(감사 가능).

```python
import pandas as pd
from inference import main
main(pd.read_csv("sample_data/sample_01.csv"), "checkpoint/coefficients.json")
```
> 임상 배포 전 원 논문 대비 점수/사망률 값 검증 권장. 의사결정 보조용.
