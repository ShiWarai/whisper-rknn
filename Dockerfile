# Whisper RKNN HTTP API — build on linux/arm64 (RK3588) with NPU.
# Before build: third_party/rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl + third_party/librknnrt.so
# (same major version as your .rknn models).

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/whisper-rknn

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY third_party/ /tmp/rknn_bundle/

RUN set -e; \
    pip install --no-cache-dir /tmp/rknn_bundle/rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl; \
    cp /tmp/rknn_bundle/librknnrt.so /usr/lib/librknnrt.so; \
    python -c "import rknnlite"; \
    rm -rf /tmp/rknn_bundle

COPY app/ ./app/

ENV WHISPER_ENCODER=/models/encoder.rknn \
    WHISPER_DECODER=/models/decoder.rknn \
    WHISPER_TOKENS=/models/tokens.txt \
    PORT=8080 \
    HOST=0.0.0.0

EXPOSE 8080

# HOST/PORT читаются в api_server.py (не хардкодить в uvicorn CLI)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; p=os.environ.get('PORT','8080'); urllib.request.urlopen('http://127.0.0.1:%s/health' % p, timeout=5)"

CMD ["python", "-m", "app.api_server"]
