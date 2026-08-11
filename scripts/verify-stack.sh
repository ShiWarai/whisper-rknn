#!/usr/bin/env bash
# Сборка и smoke-проверка monolith (local) и distributed стеков.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

NETWORK="${WHISPER_DOCKER_NETWORK:-whisper_rknn_default}"
SAMPLE="${SMOKE_AUDIO:-${ROOT}/samples/audio/stepan_whisper.ogg}"
COMPOSE_LOCAL=(docker compose -f docker-compose.yml)
COMPOSE_DIST=(docker compose -f docker-compose.distributed.yml)
WAIT_SEC="${VERIFY_WAIT_SEC:-240}"

usage() {
  cat <<'EOF'
usage: ./scripts/verify-stack.sh <command>

Команды:
  test          dev-образ: ruff + pytest (как CI, без NPU)
  build         prod-образ whisper-rknn-api:latest
  local         поднять monolith + smoke (WHISPER_RUNTIME=local)
  distributed   gateway + encoder-0 + decoder-0 (monolith останавливается)
  all           test + build + local + distributed

Переменные:
  SMOKE_AUDIO           путь к ogg/wav (по умолчанию samples/audio/stepan_whisper.ogg)
  VERIFY_WAIT_SEC       ожидание healthcheck (по умолчанию 240)
  WHISPER_DOCKER_NETWORK  docker network (по умолчанию whisper_rknn_default)
EOF
}

ensure_network() {
  if ! docker network inspect "${NETWORK}" >/dev/null 2>&1; then
    echo "==> create network ${NETWORK}"
    docker network create "${NETWORK}"
  fi
}

wait_worker_running() {
  local container="$1"
  local deadline=$(( $(date +%s) + WAIT_SEC ))
  echo "==> wait worker: ${container} (max ${WAIT_SEC}s)"
  while (( $(date +%s) < deadline )); do
    running="$(docker inspect -f '{{.State.Status}}' "${container}" 2>/dev/null || echo missing)"
    if [[ "${running}" == "running" ]] && docker logs "${container}" 2>&1 | tail -30 | grep -q 'listening on'; then
      echo "ready: ${container}"
      return 0
    fi
    sleep 5
  done
  echo "timeout waiting for ${container}" >&2
  docker logs --tail 80 "${container}" >&2 || true
  return 1
}

wait_healthy() {
  local container="$1"
  local deadline=$(( $(date +%s) + WAIT_SEC ))
  echo "==> wait healthy: ${container} (max ${WAIT_SEC}s)"
  while (( $(date +%s) < deadline )); do
    status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container}" 2>/dev/null || echo missing)"
    running="$(docker inspect -f '{{.State.Status}}' "${container}" 2>/dev/null || echo missing)"
    if [[ "${status}" == "healthy" ]]; then
      echo "healthy: ${container}"
      return 0
    fi
    if [[ "${status}" == "none" && "${running}" == "running" ]]; then
      echo "running (no healthcheck): ${container}"
      return 0
    fi
    sleep 5
  done
  echo "timeout waiting for ${container} (health=${status:-?} status=${running:-?})" >&2
  docker logs --tail 80 "${container}" >&2 || true
  return 1
}

container_hostport() {
  local container="$1"
  local ip
  ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${container}")"
  local port
  port="$(docker inspect -f '{{range $k, $v := .Config.Env}}{{println $v}}{{end}}' "${container}" | sed -n 's/^PORT=//p' | head -1)"
  port="${port:-8080}"
  echo "${ip}:${port}"
}

cmd_test() {
  echo "==> dev tests (ruff + pytest)"
  docker compose -f docker-compose.dev.yml build dev
  docker compose -f docker-compose.dev.yml run --rm -T dev ruff check .
  docker compose -f docker-compose.dev.yml run --rm dev pytest tests/ -q --tb=short
}

cmd_build() {
  echo "==> build prod image"
  ensure_network
  "${COMPOSE_LOCAL[@]}" build
}

cmd_local() {
  ensure_network
  if [[ ! -f "${SAMPLE}" ]]; then
    echo "sample missing: ${SAMPLE}" >&2
    exit 2
  fi
  echo "==> up monolith (WHISPER_ROLE=all, WHISPER_RUNTIME=local)"
  WHISPER_ROLE=all WHISPER_RUNTIME=local "${COMPOSE_LOCAL[@]}" up -d --build
  wait_healthy whisper-rknn-api
  host="$(container_hostport whisper-rknn-api)"
  "${ROOT}/scripts/smoke-http.sh" "${host}" "${SAMPLE}"
}

cmd_distributed() {
  ensure_network
  if [[ ! -f "${SAMPLE}" ]]; then
    echo "sample missing: ${SAMPLE}" >&2
    exit 2
  fi
  if docker ps --format '{{.Names}}' | grep -qx 'whisper-rknn-api'; then
    echo "==> stop monolith whisper-rknn-api (освободить NPU для encoder worker)"
    docker stop whisper-rknn-api >/dev/null
  fi
  echo "==> reset distributed stack"
  "${COMPOSE_DIST[@]}" down --remove-orphans >/dev/null 2>&1 || true
  echo "==> up distributed stack (gateway + encoder-0 + decoder-0)"
  "${COMPOSE_DIST[@]}" up -d --build
  wait_healthy whisper-rknn-gateway
  for c in whisper-rknn-encoder-0 whisper-rknn-decoder-0; do
    wait_worker_running "${c}"
  done
  host="$(container_hostport whisper-rknn-gateway)"
  "${ROOT}/scripts/smoke-http.sh" "${host}" "${SAMPLE}"
}

cmd_all() {
  cmd_test
  cmd_build
  cmd_local
  cmd_distributed
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    test) cmd_test ;;
    build) cmd_build ;;
    local) cmd_local ;;
    distributed|dist) cmd_distributed ;;
    all) cmd_all ;;
    -h|--help|help|"") usage ;;
    *) echo "unknown command: ${cmd}" >&2; usage; exit 2 ;;
  esac
}

main "$@"
