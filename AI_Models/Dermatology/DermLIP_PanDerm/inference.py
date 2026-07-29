#!/usr/bin/env python3
"""Maple entry point for DermLIP-PanDerm zero-shot inference with Grad-CAM."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
import open_clip

LABELS = [
    "acne", "allergic contact dermatitis", "drug rash", "eczema", "folliculitis",
    "herpes simplex", "herpes zoster", "impetigo", "insect bite",
    "pigmented purpuric eruption", "psoriasis", "tinea", "urticaria",
]
_CACHE = {}


def _load(model_path: str):
    key = str(Path(model_path).resolve())
    if key not in _CACHE:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "PanDerm-base-w-PubMed-256",
            pretrained=str(Path(key) / "open_clip_model.safetensors"),
            pretrained_hf=False, device="cpu",
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
            image_interpolation="bicubic", image_resize_mode="shortest",
        )
        _CACHE[key] = (model.eval(), preprocess, open_clip.get_tokenizer("PanDerm-base-w-PubMed-256"))
    return _CACHE[key]


def _overlay(image, activation, gradient):
    activation, gradient = activation[0], gradient[0]
    if int((len(activation)-1)**.5)**2 == len(activation)-1:
        activation, gradient = activation[1:], gradient[1:]
    side = int(len(activation)**.5)
    weights = gradient.reshape(side,side,-1).mean((0,1),keepdim=True)
    cam = torch.relu((activation.reshape(side,side,-1)*weights).sum(-1)).detach().numpy()
    cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
    heat = np.asarray(Image.fromarray(np.uint8(cam*255)).resize(image.size,Image.Resampling.BILINEAR),dtype=np.float32)/255
    base=np.asarray(image,dtype=np.float32);color=np.zeros_like(base);color[...,0]=np.clip(heat*2,0,1)*255;color[...,1]=np.clip(1-np.abs(heat-.6)/.6,0,1)*220
    return np.clip(.55*base+.45*color,0,255).astype(np.uint8)


def main(input_data, model_path: str):
    image=Image.open(str(input_data)).convert("RGB");model,preprocess,tokenizer=_load(model_path)
    holder={}
    def capture(_m,_i,o):
        holder["a"]=o;o.register_hook(lambda g:holder.__setitem__("g",g))
    hook=model.visual.blocks[-1].norm1.register_forward_hook(capture)
    image_features=model.encode_image(preprocess(image).unsqueeze(0));text_features=model.encode_text(tokenizer([f"This is a skin image of {x}" for x in LABELS]))
    image_features=image_features/image_features.norm(dim=-1,keepdim=True);text_features=text_features/text_features.norm(dim=-1,keepdim=True)
    logits=100*image_features@text_features.T;top=int(logits[0].argmax());model.zero_grad(set_to_none=True);logits[0,top].backward();hook.remove()
    probabilities=torch.softmax(logits.detach(),1)[0].numpy();ranking=np.argsort(probabilities)[::-1]
    predictions=[{"pred":int(i),"pred_name":LABELS[i],"prob":float(probabilities[i])} for i in ranking]
    return _overlay(image,holder["a"],holder["g"]),predictions


if __name__ == "__main__":
    result,predictions=main(BASE_DIR/"sample_data/sample_scin.png",str(BASE_DIR/"checkpoint"))
    print(result.shape,result.dtype,predictions[0])
