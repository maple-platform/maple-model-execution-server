# NLM_TB_Site_External_Ensemble

Trained on Shenzhen (['convnext_tiny_seed877', 'densenet121_seed733', 'resnet50_seed611']) and evaluated only on the independent Montgomery site. See `checkpoint/metrics.json`. This predicts radiographic TB-consistent abnormality, not microbiologically confirmed active TB. Grad-CAM is model attention, not lesion localization.
