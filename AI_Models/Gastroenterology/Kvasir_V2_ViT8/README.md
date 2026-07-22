# Kvasir-V2 ViT8

Eight-class GI endoscopy classifier using `mmuratarat/kvasir-v2-classifier`.

- Checkpoint directory: `checkpoint/`
- Input: upper/lower GI endoscopy PNG/JPG/JPEG image
- Classes: dyed lifted polyp, dyed resection margin, esophagitis, normal cecum, normal pylorus, normal Z-line, polyp, ulcerative colitis
- Output: DataFrame with the top finding and all softmax probabilities

Research use only; probabilities are not calibrated clinical probabilities.
