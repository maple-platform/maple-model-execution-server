# ECG_FM17

Official ECG-FM fine-tuned checkpoint with the publisher's 17-label aggregation rules.

## Input

The CSV contains a `time` column in seconds and the twelve standard lead columns `I, II, III, aVR, aVL, aVF, V1` through `V6`. The platform supplies the parsed CSV as a pandas DataFrame to `main(input_data, model_path)`.

## Output

Returns a pandas DataFrame containing `pred`, `pred_name`, and `prob`. Rows are ordered by model score where applicable.

## Local smoke test

```bash
python inference.py
```

The test input is `sample_data/sample_ecg.csv`. Large checkpoint files and sample data are delivered separately from Git according to the platform manual.

## Limitations

Research use only; not a medical device. Scores are not calibrated clinical probabilities and no deployment threshold is claimed. Validate preprocessing, labels, calibration, and population shift on the target clinical cohort.
