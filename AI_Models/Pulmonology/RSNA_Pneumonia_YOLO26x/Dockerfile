# RSNA Pneumonia YOLO — Python 3.12 / GPU 선택적 (CPU 동작 가능)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Server dependencies
RUN pip install --no-cache-dir \
        fastapi==0.121.3 \
        uvicorn==0.34.1 \
        pydantic==2.11.3 \
        Pillow

# Model dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
