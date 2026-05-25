# BraTS2020 T2 UNet3D — Python 3.12 / CUDA (NV PyTorch base)
FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi==0.121.3 \
        uvicorn==0.34.1 \
        pydantic==2.11.3 \
        Pillow \
        scipy \
        scikit-image \
        monai==1.5.2 \
        nibabel==5.4.2

COPY . .

EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
