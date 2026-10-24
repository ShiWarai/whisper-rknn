#!/usr/bin/env bash
# Скачать веса Whisper turbo RKNN + ONNX decoder в MODELS_DIR.
#
# Пресеты (WHISPER_DOWNLOAD_MODELS или первый аргумент):
#   turbo | 1 — ShiWarai/sherpa-rknn-whisper-turbo на Hugging Face
#     (encoder.rknn, decoder.onnx, tokens.txt)
#
# Свои URL (WHISPER_MODEL_URLS), через запятую или с новой строки:
#   local_filename=https://host/path/file.rknn
#
# Скачиваются и проверяются только запрошенные файлы.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
HF_BASE="${WHISPER_HF_BASE:-https://huggingface.co/ShiWarai}"
HF_TURBO_REPO="${WHISPER_HF_TURBO_REPO:-sherpa-rknn-whisper-turbo}"

mkdir -p "${MODELS_DIR}"

DOWNLOAD_LOG_INTERVAL_SEC="${DOWNLOAD_LOG_INTERVAL_SEC:-10}"

fmt_size() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec-i --suffix=B "${bytes}" 2>/dev/null || echo "${bytes} B"
  else
    echo "${bytes} B"
  fi
}

remote_content_length() {
  local url="$1"
  curl -fsIL --retry 3 --retry-delay 2 "${url}" 2>/dev/null \
    | awk -F': ' 'tolower($1) ~ /^content-length$/ { len=$2 } END { gsub(/\r/, "", len); print len }'
}

download_with_progress() {
  local tmp="$1"
  local url="$2"
  local label="$3"

  rm -f "${tmp}"
  local expected
  expected="$(remote_content_length "${url}" || true)"

  if [[ -n "${expected}" && "${expected}" -gt 0 ]]; then
    echo "download: ${label} size $(fmt_size "${expected}")"
  fi

  curl -fL --retry 3 --retry-delay 5 --silent --show-error -o "${tmp}" "${url}" &
  local curl_pid=$!
  local last_log=0

  while kill -0 "${curl_pid}" 2>/dev/null; do
    if [[ -f "${tmp}" ]]; then
      local now size
      now="$(date +%s)"
      size="$(stat -c%s "${tmp}" 2>/dev/null || echo 0)"
      if (( now - last_log >= DOWNLOAD_LOG_INTERVAL_SEC )) || [[ "${last_log}" -eq 0 ]]; then
        if [[ -n "${expected}" && "${expected}" -gt 0 ]]; then
          local pct=$(( size * 100 / expected ))
          echo "download: ${label} $(fmt_size "${size}") / $(fmt_size "${expected}") (${pct}%)"
        else
          echo "download: ${label} $(fmt_size "${size}")"
        fi
        last_log="${now}"
      fi
    fi
    sleep 2
  done

  if ! wait "${curl_pid}"; then
    rm -f "${tmp}"
    return 1
  fi

  local final
  final="$(stat -c%s "${tmp}")"
  echo "download: ${label} complete $(fmt_size "${final}")"
}

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
  download_with_progress "${tmp}" "${url}" "${local_name}"
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
