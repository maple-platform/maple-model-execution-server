"""Maple entry point for five-year incident AF risk."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

BASE_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE_DIR/"src"))
from fairseq_signals.utils import checkpoint_utils
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
_CACHE={}

def main(input_data,model_path:str):
    columns={str(x).lower():x for x in input_data.columns}
    missing=[x for x in LEADS if x.lower() not in columns]
    if missing: raise ValueError(f"Missing ECG leads: {missing}")
    fs=1/float(np.median(np.diff(input_data["time"]))) if "time" in input_data else 250
    x=resample_poly(input_data[[columns[x.lower()] for x in LEADS]].to_numpy(np.float32).T,250,round(fs),axis=1)/0.0048
    if x.shape[1]!=2500: raise ValueError("WCR_AFIB_5Y requires a 10-second ECG")
    key=str(Path(model_path).resolve())
    if key not in _CACHE:
        model,_,_=checkpoint_utils.load_model_and_task(str(Path(key)/"wcr_afib_5y.pt"),arg_overrides={"model_path":str(Path(key)/"base_ssl.pt")},suffix="");_CACHE[key]=model.eval()
    with torch.inference_mode(): out=_CACHE[key](source=torch.from_numpy(x).unsqueeze(0),padding_mask=None);p=torch.sigmoid(_CACHE[key].get_logits(out)).reshape(-1)[0].item()
    return pd.DataFrame([{"pred":int(p>=0.5),"pred_name":"Incident AF within 5 years","prob":p},{"pred":int(p<0.5),"pred_name":"No incident AF within 5 years","prob":1-p}]).sort_values("prob",ascending=False,ignore_index=True)

if __name__=="__main__":
    print(main(pd.read_csv(BASE_DIR/"sample_data/sample_ecg.csv"),str(BASE_DIR/"checkpoint")))
