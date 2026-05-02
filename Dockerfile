FROM python:3.11-slim

ENV TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git espeak-ng ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchaudio

RUN pip install --no-cache-dir \
    neutts \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

WORKDIR /app
COPY .services/NeuTTSTTS/server.py ./server.py
COPY tools/neutts_samples/jo.wav ./samples/jo.wav
COPY tools/neutts_samples/jo.txt ./samples/jo.txt

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]