#!/usr/bin/env python3
"""Maple entry point for Derm Foundation + SCIN classification and occlusion map."""
from __future__ import annotations

import subprocess,sys,tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from PIL import Image,ImageDraw

BASE_DIR=Path(__file__).resolve().parent


def _embedding(args):
    model_dir,image_path,output_path=args
    subprocess.run([sys.executable,str(BASE_DIR/"extract_embedding_once.py"),str(model_dir),str(image_path),str(output_path)],check=True)
    return np.load(output_path)


def _overlay(image,cam):
    cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    heat=np.asarray(Image.fromarray(np.uint8(cam*255)).resize(image.size,Image.Resampling.BILINEAR),dtype=np.float32)/255
    base=np.asarray(image,dtype=np.float32);color=np.zeros_like(base);color[...,0]=np.clip(heat*2,0,1)*255;color[...,1]=np.clip(1-np.abs(heat-.6)/.6,0,1)*220
    return np.clip(.55*base+.45*color,0,255).astype(np.uint8)


def main(input_data,model_path:str):
    model_root=Path(model_path);image=Image.open(str(input_data)).convert("RGB");grid=3
    classifier=joblib.load(model_root/"classifier/scin_common13_classifier.joblib")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp=Path(tmp_name);paths=[]
        original=tmp/"original.png";image.save(original);paths.append(original)
        for row in range(grid):
            for col in range(grid):
                altered=image.copy();draw=ImageDraw.Draw(altered)
                x0=round(col*image.width/grid);x1=round((col+1)*image.width/grid);y0=round(row*image.height/grid);y1=round((row+1)*image.height/grid)
                color=tuple(np.asarray(image)[y0:y1,x0:x1].reshape(-1,3).mean(0).astype(int));draw.rectangle((x0,y0,x1,y1),fill=color)
                path=tmp/f"occ_{row}_{col}.png";altered.save(path);paths.append(path)
        args=[(model_root/"foundation",path,tmp/f"emb_{i}.npy") for i,path in enumerate(paths)]
        with ThreadPoolExecutor(max_workers=4) as pool: embeddings=np.stack(list(pool.map(_embedding,args)))
    probabilities=classifier.predict_proba(embeddings);baseline=probabilities[0];target=int(np.argmax(baseline));cam=(baseline[target]-probabilities[1:,target]).reshape(grid,grid)
    ranking=np.argsort(baseline)[::-1];labels=[str(x) for x in classifier.classes_]
    predictions=[{"pred":int(i),"pred_name":labels[i],"prob":float(baseline[i])} for i in ranking]
    return _overlay(image,cam),predictions


if __name__=="__main__":
    result,predictions=main(BASE_DIR/"sample_data/sample_scin.png",str(BASE_DIR/"checkpoint"))
    print(result.shape,result.dtype,predictions[0])
