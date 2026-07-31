#!/usr/bin/env bash
# Smoke HTTP API: /health и короткая транскрипция.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${1:-}"
SAMPLE="${2:-${ROOT}/samples/audio/stepan_whisper.ogg}"
TIMEOUT_SEC="${SMOKE_TIMEOUT_SEC:-180}"
API_KEY_HEADER=()

if [[ -z "${HOST}" ]]; then
  echo "usage: $0 <host:port> [audio.ogg]" >&2
  exit 2
fi

if [[ ! -f "${SAMPLE}" ]]; then
  echo "sample audio not found: ${SAMPLE}" >&2
  exit 2
fi

if [[ -n "${WHISPER_API_KEY:-}" ]]; then
  API_KEY_HEADER=(-H "Authorization: Bearer ${WHISPER_API_KEY}")
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  API_KEY_HEADER=(-H "Authorization: Bearer ${OPENAI_API_KEY}")
fi

echo "==> health ${HOST}"
health_json="$(curl -fsS --max-time 15 "${API_KEY_HEADER[@]}" "http://${HOST}/health")"
echo "${health_json}"
if ! grep -q '"status"[[:space:]]*:[[:space:]]*"ok"' <<<"${health_json}"; then
  echo "health check failed" >&2
  exit 1
fi

echo "==> transcribe ${SAMPLE}"
start_ts=$(date +%s)
response="$(curl -fsS --max-time "${TIMEOUT_SEC}" "${API_KEY_HEADER[@]}" \
  -F "file=@${SAMPLE}" \
  -F "language=ru" \
  -F "response_format=json" \
  "http://${HOST}/v1/audio/transcriptions")"
elapsed=$(( $(date +%s) - start_ts ))
echo "${response}" | head -c 240
echo
if ! grep -q '"text"' <<<"${response}"; then
  echo "transcription response has no text field" >&2
  exit 1
fi
text_len=$(python3 - <<'PY' "${response}"
import json, sys
print(len(json.loads(sys.argv[1]).get("text", "").strip()))
PY
)
if [[ "${text_len}" -lt 8 ]]; then
  echo "transcription text too short (${text_len} chars)" >&2
  exit 1
fi
echo "OK smoke (${elapsed}s, ${text_len} chars)"
