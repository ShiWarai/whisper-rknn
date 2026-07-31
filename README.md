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

На **RK3588** (`linux/arm64`): `third_party/`, устройства NPU в `docker-compose.yml`, каталог моделей на хосте. Подробности — [docs/models.md](docs/models.md).

```bash
docker compose build
docker compose up -d
```

Образ из GHCR (`:main`, `:prerelease`), prerelease-теги, локальные тесты как в CI: [docs/cicd.md](docs/cicd.md).

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

Для бота: `OPENAI_BASE_URL=http://whisper-rknn-api:9003/v1` (порт из `.env`).

---

## Структура проекта

```
whisper-rknn/
├── app/
│   ├── api_server.py         # OpenAI-compatible FastAPI
│   ├── openai_response.py    # response_format / SSE helpers
│   ├── system_memory.py      # MemAvailable RAM forecast
│   ├── decode.py             # RKNN encoder, chunking, decode loop
│   ├── encode_pool.py        # parallel NPU encoder workers
│   ├── speech_cut.py         # Silero VAD spans in RAM
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
├── samples/                  # локальные записи (gitignore)
│   ├── audio/
│   └── vad_out/
├── docker-compose.dev.yml
└── requirements.txt
```

---

## Тестирование

```bash
docker compose -f docker-compose.dev.yml build dev
docker compose -f docker-compose.dev.yml run --rm -T dev ruff check .
docker compose -f docker-compose.dev.yml run --rm dev pytest tests/ -v --tb=short
```

Подробнее (как в CI, кэш GHA): [docs/cicd.md](docs/cicd.md#локальные-тесты-как-в-ci).

---

## CI/CD

Workflows, GHCR, prerelease `[prerelease]`, Telegram: **[docs/cicd.md](docs/cicd.md)**.

---

## Лицензия

MIT — см. [LICENSE](LICENSE).

Код приложения — MIT. Бинарники Rockchip в `third_party/` — по условиям Rockchip RKNN SDK. Системный `ffmpeg` в образе — по лицензии Debian (GPL/LGPL компоненты libav).

_Проект создан с использованием нейросетей._
