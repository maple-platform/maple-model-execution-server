"""Maple entry point for ECG-FM 17-label inference."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

BASE_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE_DIR/"src"))
from fairseq_signals.models import build_model_from_checkpoint
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
AGG={"Poor data quality":"max","Sinus rhythm":"mean","Premature ventricular contraction":"max","Tachycardia":"mean","Ventricular tachycardia":"max","Supraventricular tachycardia with aberrancy":"max","Bradycardia":"mean","Infarction":"mean","Atrioventricular block":"mean","Right bundle branch block":"mean","Left bundle branch block":"mean","Electronic pacemaker":"max","Atrial fibrillation":"mean","Atrial flutter":"mean","Accessory pathway conduction":"mean","1st degree atrioventricular block":"mean","Bifascicular block":"mean"}
_CACHE={}

def _segments(frame):
    columns={str(x).lower():x for x in frame.columns}
    missing=[x for x in LEADS if x.lower() not in columns]
    if missing: raise ValueError(f"Missing ECG leads: {missing}")
    fs=1/float(np.median(np.diff(frame["time"]))) if "time" in frame else 500
    x=resample_poly(frame[[columns[x.lower()] for x in LEADS]].to_numpy(np.float32).T,500,round(fs),axis=1)
    x=(x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-8)
    usable=(x.shape[1]//2500)*2500
    if not usable: raise ValueError("ECG-FM requires at least five seconds of ECG")
    return torch.from_numpy(x[:,:usable].reshape(12,-1,2500).transpose(1,0,2).copy()).float()

def main(input_data,model_path:str):
    key=str(Path(model_path).resolve())
    if key not in _CACHE: _CACHE[key]=build_model_from_checkpoint(checkpoint_path=str(Path(key)/"mimic_iv_ecg_finetuned.pt")).eval()
    labels=pd.read_csv(BASE_DIR/"ref/label_def.csv")["name"].tolist()
    with torch.inference_mode(): pred=torch.sigmoid(_CACHE[key](source=_segments(input_data))["out"]).numpy()
    probs=np.array([pred[:,i].max() if AGG[name]=="max" else pred[:,i].mean() for i,name in enumerate(labels)])
    order=np.argsort(probs)[::-1]
    return pd.DataFrame([{"pred":int(probs[i]>=0.5),"pred_name":labels[i],"prob":float(probs[i])} for i in order])

if __name__=="__main__":
    print(main(pd.read_csv(BASE_DIR/"sample_data/sample_ecg.csv"),str(BASE_DIR/"checkpoint")).head())
