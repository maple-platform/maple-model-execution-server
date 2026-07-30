from __future__ import annotations
import base64,importlib.util,io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
def predict(input_path:str,output_dir:str,config:dict)->dict:
 name=str(config['model_name']);module_name='_maple_nlm_tb'
 if module_name in sys.modules:m=sys.modules[module_name]
 else:
  spec=importlib.util.spec_from_file_location(module_name,str(config['inference_path']));m=importlib.util.module_from_spec(spec);sys.modules[module_name]=m;spec.loader.exec_module(m)
 image,predictions=m.main(input_path,str(config['model_path']));buf=io.BytesIO();Image.fromarray(image.astype(np.uint8),'RGB').save(buf,format='PNG');raw=buf.getvalue();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);stem=Path(input_path).stem;ip=out/f'{stem}_gradcam.png';jp=out/f'{stem}_prediction.json';payload={'model':name,'input':Path(input_path).name,'image_count':1,'labels':['gradcam_overlay'],'predictions':predictions};ip.write_bytes(raw);jp.write_text(json.dumps(payload,indent=2)+'\n');return {**payload,'images_b64':[base64.b64encode(raw).decode()],'output_files':[str(ip),str(jp)]}
