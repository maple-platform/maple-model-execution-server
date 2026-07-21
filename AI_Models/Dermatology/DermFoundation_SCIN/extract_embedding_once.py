#!/usr/bin/env python3
import os,sys
from io import BytesIO
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
import numpy as np
import tensorflow as tf
from PIL import Image

model_dir=Path(sys.argv[1]);image=Image.open(sys.argv[2]).convert("RGB")
buffer=BytesIO();image.save(buffer,"PNG")
example=tf.train.Example(features=tf.train.Features(feature={"image/encoded":tf.train.Feature(bytes_list=tf.train.BytesList(value=[buffer.getvalue()]))})).SerializeToString()
model=tf.saved_model.load(str(model_dir))
embedding=model.signatures["serving_default"](inputs=tf.constant([example]))["embedding"].numpy()[0]
np.save(sys.argv[3],embedding)
