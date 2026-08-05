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
| [API](#api) | Эндпоинты, пример `stepan_whisper.ogg` |
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

### Пример

Короткий русский диалог (~22 с). Послушать:

<audio controls src="docs/samples/stepan_whisper.ogg"></audio>

Файл в репозитории: [`docs/samples/stepan_whisper.ogg`](docs/samples/stepan_whisper.ogg).

Ниже — **реальные ответы API** на этом файле (`model=whisper-1`, `language=ru`, RK3588, monolith). Как вызвать: **[docs/api.md](docs/api.md)**.

#### `response_format=json`

```json
{"text":"Ну что ты орешь опять в нашей квартире? Лёся, дорогой ты мой человечек. Что? Что? Смотрю я на тебя и думаю, пойду-ка я сосну. А, алкоголик. Прости, Господи"}
```

#### `response_format=text`

```
Ну что ты орешь опять в нашей квартире? Лёся, дорогой ты мой человечек. Что? Что? Смотрю я на тебя и думаю, пойду-ка я сосну. А, алкоголик. Прости, Господи
```

#### `response_format=verbose_json`

```json
{
  "task": "transcribe",
  "language": "ru",
  "duration": 21.502,
  "text": "Ну что ты орешь опять в нашей квартире? Лёся, дорогой ты мой человечек. Что? Что? Смотрю я на тебя и думаю. Пойду-ка я сосну. А, а, а. Сосну. Алкоголик. Фу. Прости, Господи",
  "segments": [
    {"id": 0, "start": 0.0, "end": 1.68, "text": "Ну что ты орешь опять в нашей квартире"},
    {"id": 1, "start": 1.68, "end": 5.56, "text": "Лёся, дорогой ты мой человечек"},
    {"id": 2, "start": 5.96, "end": 6.2, "text": "Что"},
    {"id": 3, "start": 7.4, "end": 8.1, "text": "Что"},
    {"id": 4, "start": 9.42, "end": 10.88, "text": "Смотрю я на тебя и думаю"},
    {"id": 5, "start": 12.36, "end": 13.66, "text": "Пойду-ка я сосну"},
    {"id": 6, "start": 14.94, "end": 15.94, "text": "А, а, а"},
    {"id": 7, "start": 16.28, "end": 17.0, "text": "Сосну"},
    {"id": 8, "start": 17.64, "end": 18.32, "text": "Алкоголик"},
    {"id": 9, "start": 19.12, "end": 19.34, "text": "Фу"},
    {"id": 10, "start": 20.24, "end": 21.06, "text": "Прости, Господи"}
  ],
  "timings": {
    "encoder_ms": 12157.11,
    "decoder_ms": 8614.96,
    "wall_ms": 20997.34,
    "rtf": 0.9765,
    "chunks": 1
  }
}
```

> В полном ответе у каждого сегмента также есть `seek`, `tokens`, `temperature`, `avg_logprob` и др. — см. [docs/api.md](docs/api.md).

#### `response_format=srt`

```
1
00:00:00,000 --> 00:00:01,680
Ну что ты орешь опять в нашей квартире

2
00:00:01,680 --> 00:00:05,560
Лёся, дорогой ты мой человечек

3
00:00:05,960 --> 00:00:06,200
Что

4
00:00:07,400 --> 00:00:08,100
Что

5
00:00:09,420 --> 00:00:10,880
Смотрю я на тебя и думаю

6
00:00:12,360 --> 00:00:13,660
Пойду-ка я сосну

7
00:00:14,940 --> 00:00:15,940
А, а, а

8
00:00:16,280 --> 00:00:17,000
Сосну

9
00:00:17,640 --> 00:00:18,320
Алкоголик

10
00:00:19,120 --> 00:00:19,340
Фу

11
00:00:20,240 --> 00:00:21,060
Прости, Господи
```

#### `response_format=vtt`

```
WEBVTT

00:00:00.000 --> 00:00:01.680
Ну что ты орешь опять в нашей квартире

00:00:01.680 --> 00:00:05.560
Лёся, дорогой ты мой человечек

00:00:05.960 --> 00:00:06.200
Что

00:00:07.400 --> 00:00:08.100
Что

00:00:09.420 --> 00:00:10.880
Смотрю я на тебя и думаю

00:00:12.360 --> 00:00:13.660
Пойду-ка я сосну

00:00:14.940 --> 00:00:15.900
А, а, а

00:00:16.280 --> 00:00:17.000
Сосну

00:00:17.640 --> 00:00:18.320
Алкоголик

00:00:19.120 --> 00:00:19.340
Фу

00:00:20.240 --> 00:00:21.060
Прости, Господи
```

#### `stream=true` (SSE, `response_format` игнорируется)

```
data: {"type": "transcript.text.delta", "delta": "Ну что ты орешь опять в нашей квартире? Лёся, дорогой ты мой человечек. Что? Что? Смотрю я на тебя и думаю, пойду-ка я сосну. А, алкоголик. Прости, Господи"}

data: {"type": "transcript.text.done", "text": "Ну что ты орешь опять в нашей квартире? Лёся, дорогой ты мой человечек. Что? Что? Смотрю я на тебя и думаю, пойду-ка я сосну. А, алкоголик. Прости, Господи"}

data: [DONE]
```

#### `/v1/audio/translations` (`response_format=json`)

```json
{"text":"Ну что ты орешь опять в нашей квартире? Лёся, дорогой ты мой человечек. Что? Что? Смотрю я на тебя и думаю, пойду-ка я сосну. А, а, а, а. Алкоголик. О, прости, Господи"}
```

> `json` / `text` / SSE — без таймкодов Whisper (лучше текст). `verbose_json` / `srt` / `vtt` — с сегментами и таймингами (другой decode-проход).

Параметры (`file`, `language`, `response_format`, `stream`, auth, ошибки, chunking): **[docs/api.md](docs/api.md)**.

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
│   ├── cicd.md
│   └── samples/              # демо-аудио для README (stepan_whisper.ogg)
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
