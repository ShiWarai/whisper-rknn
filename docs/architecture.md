# Архитектура whisper-rknn

Проект поддерживает два режима развёртывания:

| Режим | Compose | `WHISPER_RUNTIME` | Описание |
|-------|---------|-------------------|----------|
| **monolith** | `docker-compose.yml` | `local` | Один контейнер: HTTP API + NPU encode + CPU decode |
| **distributed** | `docker-compose.distributed.yml` / k3s | `distributed` | Gateway (тот же HTTP API) + encoder/decoder gRPC workers |

Внешний контракт в обоих режимах — **один** OpenAI-compatible REST entrypoint: `app.api_server` (`POST /v1/audio/transcriptions`).

## Единый пайплайн

Local и distributed отличаются только транспортом encode→decode на чанк. Вся логика VAD/mel/stitch/SSE-reorder — в `app/pipeline/`:

| Модуль | Назначение |
|--------|------------|
| `pipeline/chunks.py` | `plan_utterance_chunks`, `utterance_mels` |
| `pipeline/utterance.py` | `run_utterance_pipeline`, `utterance_stream` |
| `pipeline/stream.py` | `emit_chunks_in_order` (SSE reorder) |
| `pipeline/transport.py` | `LocalChunkTransport`, `GrpcChunkTransport` |
| `runtime/backend.py` | `LocalBackend`, `GrpcBackend` — тонкие обёртки над pipeline |

## Распределённый пайплайн

```
Client (HTTP)
    │
    ▼
┌─────────────┐   PCM 16kHz    ┌──────────────────────────────┐
│  api_server │ ─────────────► │ VAD + mel (CPU, in-process)  │
│  (gateway)  │                └──────────────┬───────────────┘
└──────┬──────┘                               │ mel ~1.5 MiB/chunk
       │ gRPC EncodeThenDecode                ▼
       │                          ┌───────────────────────┐
       └────────────────────────► │ encoder worker × N  │  NPU pool на узел
                                  └───────────┬───────────┘
                                              │ cross_kv (~60 MiB)
                                              │ напрямую в decoder
                                              ▼
                                  ┌───────────────────────┐
                                  │ decoder worker × M    │  CPU ONNX
                                  └───────────┬───────────┘
                                              │ text (без cross_kv в gateway)
                                              ▼
                                    stitch → HTTP / SSE
```

**N** и **M** задаются списками `ENCODER_TARGETS` / `DECODER_TARGETS` на gateway (least-inflight). Один encoder-инстанс = один узел с полным NPU pool; decoder-инстансы масштабируют CPU decode (на одном узле или на разных).

Gateway **не** десериализует `cross_kv`: encoder-воркер получает `decode_target` и сам вызывает `DecodeService.Decode`.

### Сервисы и роли

| `WHISPER_ROLE` | Модуль | Что грузится |
|----------------|--------|--------------|
| `all` / `local` | `app.api_server` | encoder.rknn + decoder.onnx + tokens + VAD |
| `gateway` | `app.api_server` (`WHISPER_RUNTIME=distributed`) | VAD + профиль из env; без RKNN/ONNX |
| `encoder` | `app.encode_worker` | `encoder.rknn` — пул на все dedicated NPU cores (`rknn_dup_context`) |
| `decoder` | `app.decode_worker` | `decoder.onnx` + `tokens.txt`; профиль из env |

Роль задаётся в `scripts/docker-entrypoint.sh`.

### Параметры воркеров (encoder / decoder)

Gateway (только VAD + mel) эти переменные **не использует**.

| Переменная | Где | Назначение |
|------------|-----|------------|
| `WHISPER_NPU_CORE_MASK` | encoder, monolith | Какие NPU cores задействует encoder pool (`0`, `0_1`, `0_1_2`, …) |
| `WHISPER_ENCODER_WORKERS` / `WHISPER_ENCODER_MAX_WORKERS` | encoder, monolith | Число encoder pool воркеров (в пределах разрешённых NPU cores) |
| `WHISPER_CPU_AFFINITY` | encoder, decoder, monolith | `sched_setaffinity`: `4,5,6` или `4-7` — P-cores и т.п. |
| `WHISPER_ONNX_INTRA_OP_THREADS` | decoder, monolith | Потоки ONNX Runtime (`0` = все видимые CPU процесса) |

Дополнительно на уровне k8s/Docker: `cpuset`, `resources.limits.cpu` — ограничивают видимые CPU; приложение подстраивает ONNX под `sched_getaffinity`.

### gRPC контракт

Protobuf: [`proto/whisper_rknn/v1/worker.proto`](../proto/whisper_rknn/v1/worker.proto)

- `EncodeService.Encode`: mel → cross_kv (отладка)
- `EncodeService.EncodeThenDecode`: mel + `decode_target` → text (основной путь)
- `DecodeService.Decode`: cross_kv → text/segments
- `Health`: inflight, MemAvailable, npu_core

Сгенерированные stubs: `app/core/grpc_gen/`. Регенерация:

```bash
PYTHON=.venv/bin/python ./scripts/generate_proto.sh
```

Балансировка воркеров: least-inflight в `app/core/grpc_client.py`. Headless k8s Services резолвятся через `expand_targets()`.

## Рекомендация: одна машина (RK3588)

На **одной плате** используйте **monolith** — `docker compose up -d`. Один контейнер, encoder pool на все NPU cores, decode in-process. Бенчмарк ~107 с аудио: monolith ~68 с wall, distributed на той же машине ~74 с (лишний gRPC).

Distributed имеет смысл при **горизонтальном масштабировании** (несколько узлов, рост throughput), не как замена monolith на одном SoC.

## Monolith (один узел)

```bash
docker network create whisper_rknn_default
docker compose up -d --build
```

## Распределённый Compose (smoke на одной плате)

Минимальный стек для проверки gRPC-пути — **не** полная топология кластера:

```bash
docker compose -f docker-compose.distributed.yml up -d --build
```

| Сервис в compose | Назначение |
|------------------|------------|
| `gateway` | HTTP API, `WHISPER_RUNTIME=distributed` |
| `encoder-0` | один encoder-воркер (NPU pool на все ядра узла) |
| `decoder-0` | один decoder-воркер |

Имена с `-0` — соглашение для k8s headless Services; в compose достаточно одной реплики каждого типа, чтобы убедиться, что пайплайн работает.

Переменные gateway: `ENCODER_TARGETS`, `DECODER_TARGETS`, `WHISPER_RUNTIME=distributed`, `WHISPER_MODEL_PROFILE`.

## k3s

Те же роли и gRPC-контракты. Масштабирование — на стороне кластера:

| Компонент | Масштабирование |
|-----------|-----------------|
| gateway | 1 HTTP Service (или несколько за ingress) |
| encoder-* | **по одному на узел** с NPU (каждый — полный NPU pool) |
| decoder-* | **M реплик** на CPU (один узел или несколько) |

Gateway:

```text
ENCODER_TARGETS=encoder-0:50051,encoder-1:50051,encoder-2:50051
DECODER_TARGETS=decoder-0:50052,decoder-1:50052,decoder-2:50052
```

Headless Services резолвятся в несколько pod'ов через `expand_targets()` в `app/core/grpc_client.py`. Манифесты k3s в репозитории не хранятся — настраиваются на сервере.
