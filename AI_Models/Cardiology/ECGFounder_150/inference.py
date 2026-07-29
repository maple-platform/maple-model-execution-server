"""Maple entry point for ECGFounder 150-label inference."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

BASE_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE_DIR/"src"))
from net1d import Net1D
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
_CACHE={}

def _preprocess(frame):
    columns={str(x).lower():x for x in frame.columns}
    missing=[x for x in LEADS if x.lower() not in columns]
    if missing: raise ValueError(f"Missing ECG leads: {missing}")
    signal=frame[[columns[x.lower()] for x in LEADS]].to_numpy(dtype=np.float32).T
    signal=(signal-signal.mean())/(signal.std()+1e-8)
    old=np.linspace(0,1,signal.shape[1]);new=np.linspace(0,1,5000)
    return np.stack([interp1d(old,x)(new) for x in signal]).astype(np.float32)

def _load(path):
    key=str(Path(path).resolve())
    if key not in _CACHE:
        model=Net1D(in_channels=12,base_filters=64,ratio=1,filter_list=[64,160,160,400,400,1024,1024],m_blocks_list=[2,2,2,3,3,4,4],kernel_size=16,stride=2,groups_width=16,verbose=False,use_bn=False,use_do=False,n_classes=150)
        ckpt=torch.load(Path(key)/"12_lead_ECGFounder.pth",map_location="cpu",weights_only=False)
        model.load_state_dict(ckpt["state_dict"],strict=True);_CACHE[key]=model.eval()
    return _CACHE[key]

def main(input_data,model_path:str):
    with torch.inference_mode(): probs=torch.sigmoid(_load(model_path)(torch.from_numpy(_preprocess(input_data)).unsqueeze(0)))[0].numpy()
    labels=[x for x in (BASE_DIR/"ref/tasks.txt").read_text().splitlines() if x.strip()]
    order=np.argsort(probs)[::-1]
    return pd.DataFrame([{"pred":int(probs[i]>=0.5),"pred_name":labels[i],"prob":float(probs[i])} for i in order])

if __name__=="__main__":
    print(main(pd.read_csv(BASE_DIR/"sample_data/sample_ecg.csv"),str(BASE_DIR/"checkpoint")).head())
