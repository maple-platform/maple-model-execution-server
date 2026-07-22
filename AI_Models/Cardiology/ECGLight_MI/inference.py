"""Maple entry point for ECGLight MI-vs-normal classification."""
from pathlib import Path
import json,pickle,sys
import numpy as np
import pandas as pd

BASE_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE_DIR/"src"))
from classification_runner import preprocess_dataframe_for_inference,run_pretrained_inference
_CACHE={}

def _load(model_path):
    key=str(Path(model_path).resolve())
    if key not in _CACHE:
        metadata=json.loads((Path(key)/"model_metadata.json").read_text())
        with (Path(key)/metadata["model_file"]).open("rb") as f: model=pickle.load(f)
        _CACHE[key]=(model,metadata)
    return _CACHE[key]

def main(input_data,model_path:str):
    model,metadata=_load(model_path)
    x,ids,_=preprocess_dataframe_for_inference(input_data,metadata)
    result=run_pretrained_inference(model,x,metadata,ids)
    probs=np.asarray(result["probabilities"],dtype=float).mean(0);labels=result["class_labels"];top=int(np.argmax(probs))
    return pd.DataFrame([{"pred":int(i==top),"pred_name":str(name),"prob":float(probs[i])} for i,name in sorted(enumerate(labels),key=lambda x:probs[x[0]],reverse=True)])

if __name__=="__main__":
    print(main(pd.read_csv(BASE_DIR/"sample_data/sample_segmented_ecg.csv"),str(BASE_DIR/"checkpoint")))
