#!/usr/bin/env bash
# Совместимость: делегирует в единый entrypoint.
set -euo pipefail
exec "$(dirname "$0")/docker-entrypoint.sh" "$@"
