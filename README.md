# whisper-rknn

[![Deploy](https://github.com/ShiWarai/whisper-rknn/actions/workflows/deploy.yml/badge.svg)](https://github.com/ShiWarai/whisper-rknn/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux%2Farm64-orange.svg)
![Docker](https://img.shields.io/badge/docker-GHCR-blue.svg)

Распознавание **Whisper turbo** на **Rockchip RK3588**: **NPU encoder** (`encoder.rknn`) + **CPU ONNX decoder** (`decoder.onnx`) через контейнер с **REST API**. Полный RKNN (`decoder.rknn`) остаётся как fallback. Единственный поддерживаемый способ запуска — Docker.

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
| [Модели и third_party](#модели-и-third_party) | `.rknn`, `decoder.onnx`, SDK |
| [API](#api) | Эндпоинты |
| [Структура проекта](#структура-проекта) | Дерево каталогов |
| [Тестирование](#тестирование) | ruff, pytest в dev-контейнере |
| [CI/CD](#cicd) | Пайплайны, GHCR, релизы, Telegram |
| [Лицензия](#лицензия) | MIT + Rockchip SDK |

---

## Быстрый старт

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

---

## Установка и запуск

### Локальная сборка (RK3588)

**RAM:** профиль `turbo` (~1.7 GiB весов + пик запроса) рассчитан на платы с **≥8 GiB** и достаточным `MemAvailable`. На 2–4 GiB используйте `base`/`small`. Сервис отказывается загружать модель при старте и возвращает **507** на `/v1/audio/transcriptions`, если прогноз не влезает в `MemAvailable` (см. `WHISPER_MAX_AUDIO_SECONDS`).

Требуется `third_party/` с `librknnrt.so` и wheel rknnlite. См. [third_party/README.md](third_party/README.md).

На хосте нужны устройства NPU (как в [video-descriptor-rkllm](https://github.com/ShiWarai/video-descriptor-rkllm)): `/dev/mpp_service`, `/dev/rga`, `/dev/dri`, `/dev/dma_heap` — проброшены в `docker-compose.yml`.

```bash
docker compose build
docker compose up -d
```

Сервис: `privileged: true`, `platform: linux/arm64`, порты на хост **не** публикуются. Размер образа ~**350 MB** (slim Python + PyAV wheel + apt ffmpeg, без PyTorch).

### Автозагрузка turbo

Как в [video-descriptor-rkllm](https://github.com/ShiWarai/video-descriptor-rkllm) — при старте контейнера:

```bash
# в .env
WHISPER_DOWNLOAD_MODELS=turbo
WHISPER_MODEL_PROFILE=turbo
WHISPER_LANGUAGE=ru
WHISPER_MODELS_DIR=/mnt/nvme0/models/whisper-rknn-turbo
```

Источник: [HF ShiWarai/sherpa-rknn-whisper-turbo](https://huggingface.co/ShiWarai/sherpa-rknn-whisper-turbo) (`encoder.rknn`, `decoder.rknn`, `decoder.onnx`, `tokens.txt` — ~2.4 GB, скачивается один раз в volume).

### Продакшен (образ из GHCR)

```bash
docker pull ghcr.io/shiwarai/whisper-rknn:main
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Prerelease (тестовый стенд)

```bash
docker pull ghcr.io/shiwarai/whisper-rknn:prerelease
docker compose -f docker-compose.yml -f docker-compose.prerelease.yml up -d
```

Подробнее: [docs/cicd.md](docs/cicd.md).

---

## Модели и third_party

| Путь | Назначение |
|------|------------|
| `third_party/librknnrt.so` | Runtime для NPU; копируется в `/usr/lib` при сборке образа |
| `third_party/rknn_toolkit_lite2-2.3.2-…-aarch64.whl` | Python-пакет rknnlite (версия зашита в `Dockerfile`) |
| `app/assets/mel_filters.npz` | Mel-фильтры Whisper (turbo 128 / base 80 mel), без `openai-whisper` |
| Каталог моделей на хосте | `encoder.rknn`, `decoder.onnx` (по умолчанию), `decoder.rknn` (fallback), `tokens.txt` |

Версия **`librknnrt.so`** должна совпадать с toolchain, которым собраны `.rknn`. Подробнее: [docs/models.md](docs/models.md).

---

## API

Контракт совместим с [hwdsl2/docker-whisper](https://github.com/hwdsl2/docker-whisper) (OpenAI Audio API). Один клиент может переключаться между NPU и CPU сменой контейнера.

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | `{ "status": "ok", "model": "..." }` или `"loading"` |
| GET | `/v1/models` | Список активной модели |
| POST | `/v1/audio/transcriptions` | Транскрипция (multipart OpenAI) |
| POST | `/v1/audio/translations` | Перевод аудио → английский |

При заданном `WHISPER_API_KEY` / `OPENAI_API_KEY`: заголовок `Authorization: Bearer <key>`.

Пример:

```bash
docker run --rm --network whisper_rknn_default curlimages/curl:latest \
  -s -F "file=@/path/to/voice.ogg" \
  -F "model=whisper-1" \
  -F "language=ru" \
  http://whisper-rknn-api:9003/v1/audio/transcriptions
```

CPU-аналог: `hwdsl2/whisper-server` с тем же путём `/v1/audio/transcriptions`.

Полное описание: [docs/api.md](docs/api.md).

### Переменные окружения

| Переменная | По умолчанию | Смысл |
|------------|--------------|-------|
| `WHISPER_ENCODER` | `/models/encoder.rknn` | Encoder RKNN (NPU) |
| `WHISPER_DECODER` | `/models/decoder.onnx` | Decoder (ONNX на CPU по умолчанию) |
| `WHISPER_DECODER_BACKEND` | `onnx` | `onnx` / `rknn` / `auto` |
| `WHISPER_TOKENS` | `/models/tokens.txt` | Токены |
| `WHISPER_DOWNLOAD_MODELS` | `0` | `turbo` или `1` — скачать turbo с HF при старте |
| `WHISPER_MODEL_URLS` | — | Свои URL: `file.rknn=https://...` |
| `WHISPER_MODEL_PROFILE` | `turbo` | Профиль декодера (для generic-имён `encoder.rknn`) |
| `WHISPER_LANGUAGE` | `ru` | Язык распознавания: код Whisper (`ru`, `en`, `uk`, …). **Обязательно** задать под ваше аудио — иначе turbo может «галлюцинировать» на английском |
| `WHISPER_NPU_CORE_MASK` | `0_1_2` | Ядра NPU: `0`, `0_1`, `0_1_2` (все), `all`, `auto` |
| `WHISPER_MODELS_DIR` | — | Путь на хосте (volume в compose) |
| `LIBRKNNRT_SO` | — | Опциональный override пути к `.so` |
| `FFMPEG_BIN` | из `PATH` | Fallback CLI ffmpeg (основной путь — PyAV) |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | Прослушивание внутри контейнера (переопределяется в `.env`) |
| `MAX_UPLOAD_MB` | `25` | Лимит тела `POST /v1/audio/transcriptions` |
| `WHISPER_MAX_AUDIO_SECONDS` | `600` | Потолок длительности для оценки RAM запроса; при нехватке MemAvailable — HTTP 507 |
| `WHISPER_API_KEY` | — | Bearer-ключ для API (alias: `OPENAI_API_KEY`); пусто = без auth |
| `WHISPER_CHUNK_SECONDS` | окно модели (~30) | Длина куска ≤ окна 3000 mel; для ГС длиннее окна |
| `WHISPER_CHUNK_OVERLAP_SECONDS` | `2` | Перекрытие соседних окон (сэмплы внутри тех же 30 с); `0` — встык |
| `WHISPER_MIN_TAIL_SECONDS` | `8` | Короткий хвост не гоняется отдельным окном |
| `WHISPER_MAX_DECODE_TOKENS` | `0` | `0`/`auto` = до EOT в пределах `n_text_ctx` (448); число — мягкий потолок |
| `WHISPER_TRUNCATE_RETRY_SECONDS` | `10` | При обрыве без EOT — переслушать хвост в следующем окне |
| `WHISPER_MAX_NGRAM_REPEAT` | `6` | Остановка при зацикливании n-грамм в декодере |

---

## Структура проекта

```
whisper-rknn/
├── app/
│   ├── api_server.py         # OpenAI-compatible FastAPI
│   ├── openai_response.py    # response_format / SSE helpers
│   ├── system_memory.py      # MemAvailable RAM forecast
│   ├── decode.py             # RKNN encoder, chunking, decode loop
│   ├── onnx_decoder.py       # CPU ONNX decoder (hybrid)
│   ├── audio_features.py     # mel-спектрограмма (numpy + knf)
│   ├── whisper_languages.py  # language token ids без openai-whisper
│   └── assets/
│       └── mel_filters.npz   # Whisper mel filters (80/128)
├── third_party/
│   ├── librknnrt.so, rknn wheel
├── tests/
├── docs/
│   ├── api.md
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
│   └── download_models.sh
├── docker-compose.dev.yml
└── requirements.txt
```

---

## Тестирование

Dev-образ без NPU (как в CI):

```bash
docker compose -f docker-compose.dev.yml build dev
docker compose -f docker-compose.dev.yml run --rm -T dev ruff check .
docker compose -f docker-compose.dev.yml run --rm dev pytest tests/ -v --tb=short
```

Тесты проверяют чистые функции декодера и HTTP-контракт API с моками RKNN.

---

## CI/CD

| Workflow | Триггер | Назначение |
|----------|---------|------------|
| **Deploy** | Push `main` / `dev` | ruff + pytest |
| **Deploy → prerelease** | `[prerelease]` в коммите на `dev` | `:prerelease` в GHCR |
| **Publish** | Успешный Deploy на `main` | `:main` в GHCR |

Образ: `ghcr.io/shiwarai/whisper-rknn` (**только** `linux/arm64`). Теги: `:main`, `:prerelease`, `:<git-sha>`.

Telegram: secrets `TELEGRAM_TOKEN`, `TELEGRAM_TO` (опционально).

Детали: [docs/cicd.md](docs/cicd.md).

### Интеграция с ботом

В **robotics-openproject-ai-bot** подключите тот же образ/сеть и задайте `OPENAI_BASE_URL=http://whisper-rknn-api:9003/v1` (порт из `.env` whisper-rknn). Язык: `WHISPER_LANGUAGE=ru` в `.env` или `language=ru` в запросе.

---

## Лицензия

MIT — см. [LICENSE](LICENSE).

Код приложения — MIT. Бинарники Rockchip в `third_party/` — по условиям Rockchip RKNN SDK. Системный `ffmpeg` в образе — по лицензии Debian (GPL/LGPL компоненты libav).

_Проект создан с использованием нейросетей._
