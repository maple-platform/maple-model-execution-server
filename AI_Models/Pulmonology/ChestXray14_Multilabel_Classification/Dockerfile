# ChestXray14 Multilabel Classification
# Base: NV PyTorch container (system torch/torchvision provided)
# requirements.txt 주석 참고: torch/torchvision은 컨테이너 base가 제공
FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Server dependencies
RUN pip install --no-cache-dir \
        fastapi==0.121.3 \
        uvicorn==0.34.1 \
        pydantic==2.11.3

# Model dependencies — torchxrayvision은 --no-deps로 설치해 base torch와 충돌 방지
RUN pip install --no-cache-dir --no-deps torchxrayvision==1.4.0
RUN pip install --no-cache-dir \
        numpy==2.1.0 \
        Pillow==12.0.0 \
        imageio \
        scikit-image \
        opencv-python-headless==4.10.0.84

COPY . .

EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
