"""Maple entry point for the MediSense dual-head ECG Transformer."""
from pathlib import Path
import _codecs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample_poly

BASE_DIR=Path(__file__).resolve().parent
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
DIAG=["Normal","STEMI","ST depression","LBBB","RBBB","LVH"]
RHYTHM=["Normal rhythm","Atrial fibrillation","Bradycardia","Tachycardia"]
_CACHE={}

class ECGTransformer(nn.Module):
    def __init__(self):
        super().__init__();self.pos_embed=nn.Parameter(torch.zeros(1,25,256))
        self.conv_layers=nn.Sequential(nn.Conv1d(12,64,7,padding=3),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,256,7,padding=3),nn.BatchNorm1d(256),nn.ReLU(),nn.MaxPool1d(4))
        self.patch_embed=nn.Linear(256*50,256)
        layer=nn.TransformerEncoderLayer(256,8,1024,.1,batch_first=True);self.transformer=nn.TransformerEncoder(layer,6)
        self.diagnostic_head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,6))
        self.rhythm_head=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,4))
    def forward(self,x):
        x=self.conv_layers(x);x=x.unfold(2,50,50).permute(0,2,1,3).reshape(x.size(0),25,256*50)
        x=self.transformer(self.patch_embed(x)+self.pos_embed).mean(1)
        return self.diagnostic_head(x),self.rhythm_head(x)

def _load(path):
    key=str(Path(path).resolve())
    if key not in _CACHE:
        reconstruct=np.core.multiarray._reconstruct;scalar=np.core.multiarray.scalar
        reconstruct.__module__=scalar.__module__="numpy._core.multiarray"
        torch.serialization.add_safe_globals([reconstruct,scalar,np.ndarray,np.dtype,type(np.dtype(np.float64)),_codecs.encode])
        ckpt=torch.load(Path(key)/"best_model.pth",map_location="cpu",weights_only=True)
        model=ECGTransformer();model.load_state_dict(ckpt["model_state_dict"],strict=True);_CACHE[key]=model.eval()
    return _CACHE[key]

def main(input_data,model_path:str):
    columns={str(x).lower():x for x in input_data.columns}
    missing=[x for x in LEADS if x.lower() not in columns]
    if missing: raise ValueError(f"Missing ECG leads: {missing}")
    fs=1/float(np.median(np.diff(input_data["time"]))) if "time" in input_data else 500
    x=resample_poly(input_data[[columns[x.lower()] for x in LEADS]].to_numpy(np.float32).T,500,round(fs),axis=1)
    if x.shape[1] != 5000: raise ValueError("MediSense requires a 10-second ECG")
    x=(x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-6)
    with torch.inference_mode(): d,r=_load(model_path)(torch.from_numpy(x).unsqueeze(0));dp=torch.softmax(d,1)[0].numpy();rp=torch.softmax(r,1)[0].numpy()
    rows=[]
    for head,labels,probs in [("rhythm",RHYTHM,rp),("diagnostic",DIAG,dp)]:
        top=int(np.argmax(probs));rows.extend({"pred":int(i==top),"pred_name":name,"prob":float(probs[i]),"head":head} for i,name in enumerate(labels))
    return pd.DataFrame(rows).sort_values(["head","prob"],ascending=[True,False],ignore_index=True)

if __name__=="__main__":
    print(main(pd.read_csv(BASE_DIR/"sample_data/sample_ecg.csv"),str(BASE_DIR/"checkpoint")))
