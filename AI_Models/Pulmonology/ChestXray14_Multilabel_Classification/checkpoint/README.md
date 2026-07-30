# Checkpoint

No local checkpoint file is included for this model.

This project does not perform new training. It uses the public TorchXRayVision pretrained model loaded at runtime with:

```python
xrv.models.ResNet(weights="resnet50-res512-all")
```

For offline or network-restricted deployment, the platform administrator should pre-cache or bundle the TorchXRayVision pretrained weight in the runtime environment.
