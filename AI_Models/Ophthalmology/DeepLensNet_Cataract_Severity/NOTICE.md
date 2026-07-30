# Attribution notice

This package contains an inference-only adaptation derived from:

- DeepLensNet, ncbi/deeplensnet: https://github.com/ncbi/deeplensnet
- Paper: Keenan, T.D., Chen, Q., Agron, E., Tham, Y.C., Goh, J.H.L., et al.,
  Lu, Z. and Chew, E.Y., *DeepLensNet: Deep Learning Automated Diagnosis and
  Quantitative Classification of Cataract Type and Severity*, Ophthalmology
  (2022). https://www.sciencedirect.com/science/article/pii/S0161642021009672

## License status

The upstream repository does **not** publish a LICENSE file. Its README states
only: *"The performance characteristics of this product have not been
evaluated by the Food and Drug Administration and is not intended for
commercial use or purposes beyond research use only."* This is a usage
restriction, not a copyright license grant -- treat this package as
research/non-commercial use only, and confirm redistribution terms with NCBI
before any commercial or clinical deployment. The work was supported in part
by the NIH National Eye Institute and NCBI/NLM intramural research programs.

Local changes: extracted the InceptionV3-based preprocessing and per-axis
`.h5` inference from `model_classify.py`'s CSV-batch script into a single-exam
Maple `main()` entrypoint; no changes to the model architecture or weights.
No endorsement by the original authors is implied.

## Clinical limitation

This is not a medical device and must not be used for autonomous diagnosis or
patient-care decisions. NCBI's own disclaimer: *"The information produced on
this website is not intended for direct diagnostic use or medical
decision-making without review and oversight by a clinical professional."*
