#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/whisper-rknn}"
MODELS_DIR="${MODELS_DIR:-/models}"

if [[ -n "${WHISPER_DOWNLOAD_MODELS:-}" && "${WHISPER_DOWNLOAD_MODELS}" != "0" ]]; then
  echo "WHISPER_DOWNLOAD_MODELS=${WHISPER_DOWNLOAD_MODELS}: checking turbo in ${MODELS_DIR}"
  MODELS_DIR="${MODELS_DIR}" \
    WHISPER_DOWNLOAD_MODELS="${WHISPER_DOWNLOAD_MODELS}" \
    WHISPER_MODEL_URLS="${WHISPER_MODEL_URLS:-}" \
    "${APP_DIR}/scripts/download_models.sh"

  export WHISPER_ENCODER="${MODELS_DIR}/encoder.rknn"
  export WHISPER_DECODER="${WHISPER_DECODER:-${MODELS_DIR}/decoder.onnx}"
  export WHISPER_DECODER_BACKEND="${WHISPER_DECODER_BACKEND:-onnx}"
  export WHISPER_TOKENS="${MODELS_DIR}/tokens.txt"
  export WHISPER_MODELS_DIR="${MODELS_DIR}"
  export WHISPER_MODEL_PROFILE="${WHISPER_MODEL_PROFILE:-turbo}"
fi

exec python -m app.api_server "$@"
