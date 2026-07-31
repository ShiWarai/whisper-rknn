# whisper-rknn

[![Deploy](https://github.com/ShiWarai/whisper-rknn/actions/workflows/deploy.yml/badge.svg)](https://github.com/ShiWarai/whisper-rknn/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux%2Farm64-orange.svg)
![Docker](https://img.shields.io/badge/docker-GHCR-blue.svg)

Распознавание **Whisper turbo** на **Rockchip RK3588**: **NPU encoder** (`encoder.rknn`) + **CPU ONNX decoder** (`decoder.onnx`) через контейнер с **REST API**. Единственный поддерживаемый способ запуска — Docker.

Репозиторий: [github.com/ShiWarai/whisper-rknn](https://github.com/ShiWarai/whisper-rknn)

## Стек технологий

| Категория | Технологии |
|-----------|------------|
| Inference | RKNN Toolkit Lite 2 (encoder NPU), ONNX Runtime CPU (decoder), numpy mel |
| API | FastAPI, Uvicorn |
| Аудио | PyAV (libav in-process), soundfile/kaldi_native_fbank |
| Инфраструктура | Docker, Docker Compose, GHCR |
| CI | GitHub Actions, ruff, pytest |
| Платформа | **linux/arm64** (RK3588 NPU) |

## Оглавление

| Раздел | Содержание |
|--------|------------|
| [Быстрый старт](#быстрый-старт) | Запуск за 3 шага |
| [Установка и запуск](#установка-и-запуск) | Локальная сборка, prod/prerelease из GHCR |
| [Модели](#модели) | Веса, third_party, автозагрузка |
| [API](#api) | Эндпоинты |
| [Архитектура](#архитектура) | Monolith vs distributed, k3s |
| [Структура проекта](#структура-проекта) | Дерево каталогов |
| [Тестирование](#тестирование) | ruff, pytest в dev-контейнере |
| [CI/CD](#cicd) | Пайплайны, GHCR, релизы, Telegram |
| [Лицензия](#лицензия) | MIT + Rockchip SDK |

---

## Быстрый старт

**На одной RK3588** используйте monolith (`docker-compose.yml`) — один контейнер, NPU pool на все ядра, без gRPC-оверхеда. Distributed нужен для кластера или отладки gRPC, не для ускорения на одной плате.

1. Создайте Docker-сеть (один раз):

   ```bash
   docker network create whisper_rknn_default
   ```

2. Настройте `.env`:

   ```bash
   cp .env.example .env
   # WHISPER_MODELS_DIR, WHISPER_LANGUAGE=ru, PORT=9003 — см. .env.example
   ```

3. Соберите и запустите:

   ```bash
   docker compose build
   docker compose up -d
   ```

Другие контейнеры (например Telegram-бот) подключайте к `whisper_rknn_default` и обращайтесь к **`whisper-rknn-api:${PORT}`** (у нас в `.env` — `9003`).

### Distributed (кластер / отладка gRPC)

Не для ускорения на одной плате — monolith быстрее. Нужен для k3s или smoke-теста:

```bash
docker compose -f docker-compose.distributed.yml up -d --build
./scripts/verify-stack.sh distributed
```

HTTP: `whisper-rknn-gateway:${PORT}`. См. [docs/architecture.md](docs/architecture.md).

---

## Установка и запуск

На **RK3588** (`linux/arm64`): `third_party/`, устройства NPU в `docker-compose.yml`, каталог моделей на хосте. Подробности — [docs/models.md](docs/models.md).

### Локальная сборка (разработка на плате)

```bash
docker compose build
docker compose up -d
```

Собирает образ `whisper-rknn-api:latest` из `Dockerfile` на этой машине.

### Образ из GHCR (без локальной сборки)

| Overlay | Тег GHCR | Когда использовать |
|---------|----------|-------------------|
| `docker-compose.prod.yml` | `:main` | **Продакшен** — стабильная версия после merge в `main` |
| `docker-compose.prerelease.yml` | `:prerelease` | **Тестовый стенд** — свежий код из `dev` до релиза в `main` |

```bash
# prod — стабильный релиз
docker pull ghcr.io/shiwarai/whisper-rknn:main
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# prerelease — кандидат на релиз (ветка dev)
docker pull ghcr.io/shiwarai/whisper-rknn:prerelease
docker compose -f docker-compose.yml -f docker-compose.prerelease.yml up -d
```

**Чем отличается prerelease от prod:** это не другой продукт и не другой compose-файл по сути — оба overlay только подменяют `image` (убирают локальный `build`). Разница в **теге образа** и **пайплайне публикации**:

- `:prerelease` — собирается из `dev` (коммит с `[prerelease]` или ручной workflow). Для проверки фич на стенде до merge в `main`.
- `:main` — публикуется автоматически после успешного Deploy на ветке `main`. То, что крутится в проде.

Подробнее: [docs/cicd.md](docs/cicd.md).

Переменные окружения: шаблон [`.env.example`](.env.example), справочник в [docs/models.md](docs/models.md#переменные-окружения) и [docs/api.md](docs/api.md#переменные-окружения).

---

## Модели

Файлы `encoder.rknn`, `decoder.onnx`, `tokens.txt` в `WHISPER_MODELS_DIR`; автозагрузка turbo, язык, RAM, toolchain: **[docs/models.md](docs/models.md)**. Runtime NPU: [third_party/README.md](third_party/README.md).

---

## API

Контракт совместим с [hwdsl2/docker-whisper](https://github.com/hwdsl2/docker-whisper) (OpenAI Audio API). NPU — этот сервис, CPU — `hwdsl2/whisper-server`; меняется только контейнер, пути те же.

| Метод | Путь |
|-------|------|
| GET | `/health` |
| GET | `/v1/models` |
| POST | `/v1/audio/transcriptions` |
| POST | `/v1/audio/translations` |

Проверка после старта:

```bash
docker run --rm --network whisper_rknn_default curlimages/curl:latest \
  -s http://whisper-rknn-api:9003/health
```

Параметры (`file`, `language`, `response_format`, `stream`, auth, streaming SSE, примеры curl, ошибки, chunking): **[docs/api.md](docs/api.md)**.

Для бота: `OPENAI_BASE_URL=http://whisper-rknn-api:9003/v1` (порт из `.env`). В distributed режиме — `whisper-rknn-gateway:${PORT}`.

---

## Архитектура

| Режим | Compose | Когда |
|-------|---------|-------|
| **Monolith** | `docker-compose.yml` | **Одна RK3588** — рекомендуемый режим |
| Distributed | `docker-compose.distributed.yml` | Кластер k3s или smoke gRPC |
| k3s | на сервере | Несколько узлов: encoder на узел + M decoder + gateway |

Полное описание пайплайна, gRPC контракта и переменных: **[docs/architecture.md](docs/architecture.md)**.

---

## Структура проекта

```
whisper-rknn/
├── app/
│   ├── api_server.py         # единый OpenAI-compatible FastAPI (local + distributed)
│   ├── pipeline/             # общий VAD/mel/stitch/SSE pipeline
│   ├── runtime/              # LocalBackend / GrpcBackend
│   ├── gateway/              # совместимость: python -m app.gateway → api_server
│   ├── encode_worker/        # gRPC NPU encoder (pool на все ядра)
│   ├── decode_worker/        # gRPC CPU ONNX decoder
│   ├── core/                 # grpc_client, tensor_codec, model_config
│   ├── openai_response.py    # response_format / SSE helpers
│   ├── system_memory.py      # MemAvailable RAM forecast
│   ├── decode.py             # RKNN encoder, chunking, decode loop
│   ├── encode_pool.py        # parallel NPU encoder workers (monolith)
│   ├── speech_cut.py         # Silero VAD spans in RAM
│   ├── onnx_decoder.py       # CPU ONNX decoder (hybrid)
│   ├── audio_features.py     # mel-спектрограмма (numpy + knf)
│   ├── whisper_languages.py  # language token ids без openai-whisper
│   └── assets/
│       └── mel_filters.npz   # Whisper mel filters (80/128)
├── proto/whisper_rknn/v1/    # gRPC worker.proto
├── docker-compose.distributed.yml
├── third_party/
│   ├── librknnrt.so, rknn wheel
├── tests/
├── docs/
│   ├── api.md
│   ├── architecture.md
│   ├── models.md
│   └── cicd.md
├── .github/workflows/
│   ├── deploy.yml
│   └── publish.yml
├── Dockerfile
├── Dockerfile.dev
├── docker-compose.yml
├── docker-compose.prod.yml
├── docker-compose.prerelease.yml
├── scripts/
│   ├── docker-entrypoint.sh
│   ├── verify-stack.sh      # test/build/smoke local+distributed
│   ├── smoke-http.sh        # /health + транскрипция
│   └── download_models.sh
├── samples/                  # локальные записи (gitignore)
│   ├── audio/
│   └── vad_out/
├── docker-compose.dev.yml
└── requirements.txt
```

---

## Тестирование

### Unit / lint (как CI, без NPU)

```bash
./scripts/verify-stack.sh test
```

Или вручную:

```bash
docker compose -f docker-compose.dev.yml build dev
docker compose -f docker-compose.dev.yml run --rm -T dev ruff check .
docker compose -f docker-compose.dev.yml run --rm dev pytest tests/ -v --tb=short
```

### Сборка и smoke на RK3588 (оба runtime)

Один скрипт проверяет **monolith** (`WHISPER_RUNTIME=local`) и **distributed** (`WHISPER_RUNTIME=distributed`):

```bash
chmod +x scripts/verify-stack.sh scripts/smoke-http.sh

# только сборка prod-образа
./scripts/verify-stack.sh build

# monolith: docker compose up + /health + транскрипция
./scripts/verify-stack.sh local

# distributed: gateway + encoder-0 + decoder-0 (monolith останавливается)
./scripts/verify-stack.sh distributed

# всё подряд (CI-like test + build + оба smoke)
./scripts/verify-stack.sh all
```

По умолчанию аудио: `samples/audio/stepan_whisper.ogg` (локально, в git не коммитится). Переопределение: `SMOKE_AUDIO=/path/to/file.ogg`.

Подробнее (как в CI, кэш GHA): [docs/cicd.md](docs/cicd.md#локальные-тесты-как-в-ci).

---

## CI/CD

Workflows, GHCR, prerelease `[prerelease]`, Telegram: **[docs/cicd.md](docs/cicd.md)**.

---

## Лицензия

MIT — см. [LICENSE](LICENSE).

Код приложения — MIT. Бинарники Rockchip в `third_party/` — по условиям Rockchip RKNN SDK. Системный `ffmpeg` в образе — по лицензии Debian (GPL/LGPL компоненты libav).

_Проект создан с использованием нейросетей._
