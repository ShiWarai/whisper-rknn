# Whisper RKNN HTTP API — build on linux/arm64 (RK3588) with NPU.
# Before build: third_party/rknn_toolkit_lite2-2.3.2-*-aarch64.whl + third_party/librknnrt.so
# Audio: PyAV in-process (prebuilt wheel with bundled libav). apt ffmpeg — только CLI fallback.

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/whisper-rknn

COPY requirements.txt .
RUN pip install -U pip \
    && pip install -r requirements.txt

COPY third_party/librknnrt.so third_party/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl /tmp/rknn_bundle/

RUN set -e; \
    pip install /tmp/rknn_bundle/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl; \
    cp /tmp/rknn_bundle/librknnrt.so /usr/lib/librknnrt.so; \
    python -c "import rknnlite; import av; print('av', av.__version__)"; \
    rm -rf /tmp/rknn_bundle

COPY app/ ./app/
COPY scripts/docker-entrypoint.sh scripts/download_models.sh ./scripts/
RUN chmod +x ./scripts/docker-entrypoint.sh ./scripts/download_models.sh

ENV WHISPER_ENCODER=/models/encoder.rknn \
    WHISPER_DECODER=/models/decoder.onnx \
    WHISPER_DECODER_BACKEND=onnx \
    WHISPER_TOKENS=/models/tokens.txt \
    WHISPER_MODEL_PROFILE=turbo \
    PORT=8080 \
    HOST=0.0.0.0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; p=os.environ.get('PORT','8080'); urllib.request.urlopen('http://127.0.0.1:%s/health' % p, timeout=5)"

CMD ["./scripts/docker-entrypoint.sh"]
