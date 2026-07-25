# whisper-rknn

[![Deploy](https://github.com/ShiWarai/whisper-rknn/actions/workflows/deploy.yml/badge.svg)](https://github.com/ShiWarai/whisper-rknn/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux%2Farm64-orange.svg)
![Docker](https://img.shields.io/badge/docker-GHCR-blue.svg)

Распознавание **Whisper** в формате **RKNN** (`encoder.rknn` + `decoder.rknn`) на **Rockchip RK3588** через контейнер с **REST API**. Единственный поддерживаемый способ запуска — Docker; локальный CLI не используется.

Репозиторий: [github.com/ShiWarai/whisper-rknn](https://github.com/ShiWarai/whisper-rknn)

## Стек технологий

| Категория | Технологии |
|-----------|------------|
| Inference | RKNN Toolkit Lite 2, `librknnrt`, PyTorch (fbank) |
| API | FastAPI, Uvicorn |
| Аудио | ffmpeg, soundfile, kaldi_native_fbank |
| Инфраструктура | Docker, Docker Compose, GHCR |
| CI | GitHub Actions, ruff, pytest |
| Платформа | **linux/arm64** (RK3588 NPU) |

## Оглавление

| Раздел | Содержание |
|--------|------------|
| [Быстрый старт](#быстрый-старт) | Запуск за 3 шага |
| [Установка и запуск](#установка-и-запуск) | Локальная сборка, prod/prerelease из GHCR |
| [Модели и third_party](#модели-и-third_party) | Файлы `.rknn`, SDK в репозитории |
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

Требуется `third_party/` с `librknnrt.so` и wheel (уже в репозитории). См. [third_party/README.md](third_party/README.md).

```bash
docker compose build
docker compose up -d
```

Сервис: `privileged: true`, `platform: linux/arm64`, порты на хост **не** публикуются.

### Автозагрузка turbo

Как в [video-descriptor-rkllm](https://github.com/ShiWarai/video-descriptor-rkllm) — при старте контейнера:

```bash
# в .env
WHISPER_DOWNLOAD_MODELS=turbo
WHISPER_MODEL_PROFILE=turbo
WHISPER_LANGUAGE=ru
WHISPER_MODELS_DIR=/mnt/nvme0/models/whisper-rknn-turbo
```

Источник: [HF ShiWarai/sherpa-rknn-whisper-turbo](https://huggingface.co/ShiWarai/sherpa-rknn-whisper-turbo) (~1.8 GB, скачивается один раз в volume).

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
| `third_party/rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl` | Python-пакет rknnlite (версия зашита в `Dockerfile`) |
| Каталог моделей на хосте | `encoder.rknn`, `decoder.rknn`, `tokens.txt` |

Версия **`librknnrt.so`** должна совпадать с toolchain, которым собраны `.rknn`. Подробнее: [docs/models.md](docs/models.md).

---

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | `{ "status": "ok" }` или `"loading"` |
| POST | `/transcribe` | `multipart/form-data`, поле `file` → `{ "text", "elapsed_s" }` |

Пример:

```bash
docker run --rm --network whisper_rknn_default curlimages/curl:latest \
  -s -F "file=@/path/to/voice.ogg" \
  http://whisper-rknn-api:9003/transcribe
```

(порт задаётся через `PORT` в `.env`; по умолчанию в образе — `8080`)

Полное описание: [docs/api.md](docs/api.md).

### Переменные окружения

| Переменная | По умолчанию | Смысл |
|------------|--------------|-------|
| `WHISPER_ENCODER` | `/models/encoder.rknn` | Encoder внутри контейнера |
| `WHISPER_DECODER` | `/models/decoder.rknn` | Decoder |
| `WHISPER_TOKENS` | `/models/tokens.txt` | Токены |
| `WHISPER_DOWNLOAD_MODELS` | `0` | `turbo` или `1` — скачать turbo с HF при старте |
| `WHISPER_MODEL_URLS` | — | Свои URL: `file.rknn=https://...` |
| `WHISPER_MODEL_PROFILE` | `turbo` | Профиль декодера (для generic-имён `encoder.rknn`) |
| `WHISPER_LANGUAGE` | `ru` | Язык распознавания: код Whisper (`ru`, `en`, `uk`, …). **Обязательно** задать под ваше аудио — иначе turbo может «галлюцинировать» на английском |
| `WHISPER_NPU_CORE_MASK` | `0_1_2` | Ядра NPU: `0`, `0_1`, `0_1_2` (все), `all`, `auto` |
| `WHISPER_MODELS_DIR` | — | Путь на хосте (для логов; volume в compose) |
| `LIBRKNNRT_SO` | — | Опциональный override пути к `.so` |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | Прослушивание внутри контейнера (переопределяется в `.env`) |
| `MAX_UPLOAD_MB` | `25` | Лимит тела `POST /transcribe` |

---

## Структура проекта

```
whisper-rknn/
├── app/
│   ├── api_server.py      # FastAPI + uvicorn
│   └── decode.py          # fbank → RKNN → текст
├── third_party/           # RKNN SDK (wheel + librknnrt.so)
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

В **robotics-openproject-ai-bot** подключите тот же образ/сеть и задайте `WHISPER_RKNN_URL=http://whisper-rknn-api:9003` (порт из `.env` whisper-rknn). Язык распознавания настраивается в `.env` whisper-rknn: **`WHISPER_LANGUAGE=ru`** (или `en`, `uk`, …).

---

## Лицензия

MIT — см. [LICENSE](LICENSE).

Код приложения — MIT. Бинарники Rockchip в `third_party/` — по условиям Rockchip RKNN SDK.

_Проект создан с использованием нейросетей._
