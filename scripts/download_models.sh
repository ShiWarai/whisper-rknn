#!/usr/bin/env bash
# Download Whisper turbo RKNN + ONNX decoder weights into MODELS_DIR.
#
# Presets (WHISPER_DOWNLOAD_MODELS or first argument):
#   turbo | 1 — ShiWarai/sherpa-rknn-whisper-turbo on Hugging Face
#     (encoder.rknn, decoder.onnx, tokens.txt)
#
# Custom URLs (WHISPER_MODEL_URLS), comma- or newline-separated:
#   local_filename=https://host/path/file.rknn
#
# Only requested files are downloaded and verified.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
HF_BASE="${WHISPER_HF_BASE:-https://huggingface.co/ShiWarai}"
HF_TURBO_REPO="${WHISPER_HF_TURBO_REPO:-sherpa-rknn-whisper-turbo}"

mkdir -p "${MODELS_DIR}"

declare -a WANT_LOCAL=()
declare -a WANT_URL=()

add_turbo() {
  WANT_LOCAL+=("encoder.rknn" "decoder.onnx" "tokens.txt" "silero_vad.onnx")
  WANT_URL+=(
    "${HF_BASE}/${HF_TURBO_REPO}/resolve/main/encoder.rknn"
    "${HF_BASE}/${HF_TURBO_REPO}/resolve/main/decoder.onnx"
    "${HF_BASE}/${HF_TURBO_REPO}/resolve/main/tokens.txt"
    "${WHISPER_VAD_MODEL_URL:-https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx}"
  )
}

parse_selection() {
  local raw="${1:-}"
  raw="${raw// /}"
  if [[ -z "${raw}" || "${raw}" == "0" ]]; then
    return 0
  fi
  if [[ "${raw}" == "urls" || "${raw}" == "custom" ]]; then
    return 0
  fi
  case "${raw}" in
    1 | turbo) add_turbo ;;
    *)
      echo "error: unknown model preset '${raw}' (use turbo or 1)" >&2
      exit 1
      ;;
  esac
}

parse_custom_urls() {
  local raw="${WHISPER_MODEL_URLS:-}"
  [[ -z "${raw}" ]] && return 0
  raw="${raw//$'\n'/,}"
  local IFS=',' entry local_name url
  for entry in ${raw}; do
    entry="${entry#"${entry%%[![:space:]]*}"}"
    entry="${entry%"${entry##*[![:space:]]}"}"
    [[ -z "${entry}" ]] && continue
    if [[ "${entry}" != *"="* ]]; then
      echo "error: WHISPER_MODEL_URLS entry must be local_name=URL, got: ${entry}" >&2
      exit 1
    fi
    local_name="${entry%%=*}"
    url="${entry#*=}"
    local_name="${local_name#"${local_name%%[![:space:]]*}"}"
    url="${url#"${url%%[![:space:]]*}"}"
    if [[ -z "${local_name}" || -z "${url}" ]]; then
      echo "error: invalid WHISPER_MODEL_URLS entry: ${entry}" >&2
      exit 1
    fi
    WANT_LOCAL+=("${local_name}")
    WANT_URL+=("${url}")
  done
}

download_if_missing() {
  local local_name="$1"
  local url="$2"
  local dest="${MODELS_DIR}/${local_name}"

  if [[ -f "${dest}" ]]; then
    echo "skip (exists): ${local_name}"
    return 0
  fi

  local tmp="${dest}.part"
  echo "download: ${local_name} <- ${url}"
  curl -fL --retry 3 --retry-delay 5 -o "${tmp}" "${url}"
  mv "${tmp}" "${dest}"
}

selection="${WHISPER_DOWNLOAD_MODELS:-}"
if [[ $# -gt 0 ]]; then
  selection="$1"
fi

parse_selection "${selection}"
parse_custom_urls

if [[ ${#WANT_LOCAL[@]} -eq 0 ]]; then
  echo "error: nothing to download (set WHISPER_DOWNLOAD_MODELS=turbo and/or WHISPER_MODEL_URLS)" >&2
  exit 1
fi

for i in "${!WANT_LOCAL[@]}"; do
  download_if_missing "${WANT_LOCAL[$i]}" "${WANT_URL[$i]}"
done

for f in "${WANT_LOCAL[@]}"; do
  if [[ ! -f "${MODELS_DIR}/${f}" ]]; then
    echo "error: missing ${MODELS_DIR}/${f}" >&2
    exit 1
  fi
done

echo "models ready in ${MODELS_DIR}: ${WANT_LOCAL[*]}"
