"""Maple entry point for DeepECG-SL 77-label inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

BASE_DIR = Path(__file__).resolve().parent
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
_CACHE = {}

def _signal(frame):
    columns={str(x).lower():x for x in frame.columns}
    missing=[x for x in LEADS if x.lower() not in columns]
    if missing: raise ValueError(f"Missing ECG leads: {missing}")
    fs=500.0
    if "time" in frame.columns and len(frame)>1:
        fs=1.0/float(np.median(np.diff(frame["time"].to_numpy(dtype=float))))
    x=frame[[columns[x.lower()] for x in LEADS]].to_numpy(dtype=np.float32).T
    return resample_poly(x,500,round(fs),axis=1).astype(np.float32)

def main(input_data, model_path: str):
    key=str(Path(model_path).resolve())
    if key not in _CACHE: _CACHE[key]=torch.jit.load(str(Path(key)/"efficientnet_deepecg_unscaled.pt"),map_location="cpu").eval()
    x=_signal(input_data)
    if x.shape[1] != 5000: raise ValueError(f"Expected a 10-second ECG; received {x.shape[1]} samples at 500 Hz")
    with torch.inference_mode(): probs=torch.sigmoid(_CACHE[key](torch.from_numpy(x/0.0048).unsqueeze(0))).squeeze().numpy()
    labels=json.loads((BASE_DIR/"ref/labels.json").read_text())
    order=np.argsort(probs)[::-1]
    return pd.DataFrame([{"pred":int(probs[i]>=0.5),"pred_name":labels[i],"prob":float(probs[i])} for i in order])

if __name__=="__main__":
    print(main(pd.read_csv(BASE_DIR/"sample_data/sample_ecg.csv"),str(BASE_DIR/"checkpoint")).head())
