# ECGLight_MI

Official ECGLight Arsenal classifier for myocardial infarction versus normal segmented ECG beats.

## Input

The CSV contains `subject_id`, `timestamp`, and the twelve standard lead columns; rows are pre-segmented ECG beats. The platform supplies the parsed CSV as a pandas DataFrame to `main(input_data, model_path)`.

## Output

Returns a pandas DataFrame containing `pred`, `pred_name`, and `prob`. Rows are ordered by model score where applicable. The official classifier is a Python pickle; load it only after trusting the publisher artifact.

## Local smoke test

```bash
python inference.py
```

The test input is `sample_data/sample_segmented_ecg.csv`. Large checkpoint files and sample data are delivered separately from Git according to the platform manual.

## Limitations

Research use only; not a medical device. Scores are not calibrated clinical probabilities and no deployment threshold is claimed. Validate preprocessing, labels, calibration, and population shift on the target clinical cohort.
