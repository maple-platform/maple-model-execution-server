"""Maple inference for the Shenzhen-trained, Montgomery-tested TB ensemble."""
from __future__ import annotations
import json,threading
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torchvision import models
_MODELS=None;_CONFIG=None;_DEVICE=None;_LOCK=threading.Lock();_RUN=threading.Lock()
def _build(architecture):
 if architecture=='resnet50':m=models.resnet50(weights=None);m.fc=nn.Linear(m.fc.in_features,1)
 elif architecture=='densenet121':m=models.densenet121(weights=None);m.classifier=nn.Linear(m.classifier.in_features,1)
 elif architecture=='convnext_tiny':m=models.convnext_tiny(weights=None);m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,1)
 else:raise ValueError(architecture)
 return m
def _layer(model,architecture):
 if architecture=='resnet50':return model.layer4[-1],'layer4[-1]'
 if architecture=='densenet121':return model.features.denseblock4,'features.denseblock4'
 if architecture=='convnext_tiny':return model.features[-1],'features[-1]'
 raise ValueError(architecture)
def _load(model_path):
 global _MODELS,_CONFIG,_DEVICE
 root=Path(model_path)
 if _MODELS is None:
  with _LOCK:
   if _MODELS is None:
    config=json.loads((root/'model_config.json').read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');loaded=[]
    for member in config['members']:
     checkpoint=torch.load(root/member['file'],map_location='cpu',weights_only=False);model=_build(member['architecture']);model.load_state_dict(checkpoint['state_dict'],strict=True);loaded.append((model.to(device).eval(),member['architecture']))
    _MODELS,_CONFIG,_DEVICE=loaded,config,device
 return _MODELS,_CONFIG,_DEVICE
def _preprocess(input_data):
 path=Path(input_data)
 if not path.is_file() or path.suffix.lower() not in {'.png','.jpg','.jpeg'}:raise ValueError('Expected PNG/JPG PA chest radiograph')
 image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
 if image is None:raise ValueError(f'Could not decode {path}')
 foreground=image>max(3,int(np.percentile(image,2)));ys,xs=np.where(foreground)
 if len(xs):image=image[max(0,ys.min()-16):min(image.shape[0],ys.max()+17),max(0,xs.min()-16):min(image.shape[1],xs.max()+17)]
 side=max(image.shape);canvas=np.zeros((side,side),np.uint8);y0=(side-image.shape[0])//2;x0=(side-image.shape[1])//2;canvas[y0:y0+image.shape[0],x0:x0+image.shape[1]]=image
 image=cv2.resize(canvas,(384,384),interpolation=cv2.INTER_AREA);image=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(image)
 x=np.repeat(image[None],3,axis=0).astype('float32')/255;x=(x-np.asarray([.485,.456,.406],dtype='float32')[:,None,None])/np.asarray([.229,.224,.225],dtype='float32')[:,None,None]
 return torch.from_numpy(x).unsqueeze(0),image
def main(input_data,model_path):
 members,config,device=_load(model_path);batch,base=_preprocess(input_data);batch=batch.to(device);logits=[];captures=[];handles=[]
 with _RUN:
  for model,architecture in members:
   activation=[];gradient=[]
   def hook(_m,_i,out,a=activation,g=gradient):a.append(out);out.register_hook(lambda grad:g.append(grad))
   layer,name=_layer(model,architecture);handles.append(layer.register_forward_hook(hook));model.zero_grad(set_to_none=True);logit=model(batch).flatten()[0];logits.append(logit);captures.append((activation,gradient,name))
  ensemble=torch.stack(logits).mean();prob=float(torch.sigmoid(ensemble.detach()).cpu());positive=prob>=float(config['threshold']);cams=[]
  for index,(logit,(activation,gradient,_)) in enumerate(zip(logits,captures)):
   (logit if positive else -logit).backward(retain_graph=index<len(logits)-1);a=activation[-1].detach();g=gradient[-1].detach();cam=torch.relu((g.mean((2,3),keepdim=True)*a).sum(1))[0];cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8);cams.append(cv2.resize(cam.float().cpu().numpy(),(384,384)))
  for handle in handles:handle.remove()
 cam=np.mean(cams,axis=0);available=float(cam.max())>float(cam.min())+1e-8;heat=cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(cam,0,1)*255),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB);rgb=np.repeat(base[...,None],3,axis=2);overlay=cv2.addWeighted(rgb,.65,heat,.35,0) if available else rgb
 pred=int(positive);result=[{'label':'TB-consistent abnormality','probability':prob,'threshold':float(config['threshold']),'pred':pred,'pred_name':'TB-consistent abnormality' if pred else 'No TB-consistent abnormality','model_id':config['model_id'],'ensemble_members':len(members),'external_validation':config['external_test'],'gradcam_available':bool(available),'gradcam_target':'TB-consistent abnormality' if positive else 'No TB-consistent abnormality','gradcam_score':'positive_logit' if positive else 'negative_logit','gradcam_layer':'ensemble architecture-specific final feature stages','interpretation_warning':'Screening output and model attention only; not microbiologic TB diagnosis or lesion localization.','image_roles':['gradcam_overlay']}]
 return overlay.astype(np.uint8),result
