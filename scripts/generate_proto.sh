#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/app/core/grpc_gen"
PYTHON="${PYTHON:-python3}"

mkdir -p "${OUT}"
touch "${OUT}/__init__.py"
mkdir -p "${OUT}/whisper_rknn/v1"
touch "${OUT}/whisper_rknn/__init__.py"
touch "${OUT}/whisper_rknn/v1/__init__.py"

"${PYTHON}" -m grpc_tools.protoc \
  -I"${ROOT}/proto" \
  --python_out="${OUT}" \
  --grpc_python_out="${OUT}" \
  "${ROOT}/proto/whisper_rknn/v1/worker.proto"

# grpc_tools генерирует абсолютные импорты; правим под layout пакета.
GRPC_FILE="${OUT}/whisper_rknn/v1/worker_pb2_grpc.py"
if [[ -f "${GRPC_FILE}" ]]; then
  sed -i \
    -e 's/^import worker_pb2 as worker__pb2$/from app.core.grpc_gen.whisper_rknn.v1 import worker_pb2 as worker__pb2/' \
    -e 's/^from whisper_rknn\.v1 import worker_pb2 as /from app.core.grpc_gen.whisper_rknn.v1 import worker_pb2 as /' \
    "${GRPC_FILE}"
fi

echo "gRPC stubs сгенерированы в ${OUT}"
