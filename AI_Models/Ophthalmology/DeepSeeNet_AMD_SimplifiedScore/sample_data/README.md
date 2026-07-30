# Sample input

`left_eye.jpg` and `right_eye.jpg` are the upstream repository's own demo
images (`data/left_eye.jpg`, `data/right_eye.jpg`).

Source: https://github.com/ncbi-nlp/DeepSeeNet (public domain, "United States
Government Work"; see `../NOTICE.md`).

Expected result for this pair, from the upstream README's own documented CLI
run (`python examples/predict_simplified_score.py data/left_eye.jpg
data/right_eye.jpg`):

```
Risk factors: {'pigment': (0, 0), 'advanced_amd': (0, 0), 'drusen': (2, 2)}
The simplified score: 2
```

See `../IMPLEMENTATION_STATUS.md` for this package's own local validation run
against this exact pair, including a near-tied `pigment_left` probability
(0.489 vs 0.511) that flips the score to 3 under this environment's floating
point behavior -- documented there rather than silently reconciled.
