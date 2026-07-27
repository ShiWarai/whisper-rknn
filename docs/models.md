# Модели Whisper RKNN

Проект рассчитан на **Whisper turbo**. **Гибрид**: encoder на NPU (`encoder.rknn`), decoder на CPU (`decoder.onnx`). Файлы в каталоге, смонтированном в `/models`:

| Файл | Переменная | Описание |
|------|------------|----------|
| `encoder.rknn` | `WHISPER_ENCODER` | Encoder RKNN (NPU) |
| `decoder.onnx` | `WHISPER_DECODER` | Decoder ONNX (CPU) |
| `tokens.txt` | `WHISPER_TOKENS` | Словарь токенов Whisper |

Пути по умолчанию: `docker-compose.yml` (`WHISPER_DECODER=/models/decoder.onnx`, `WHISPER_DECODER_BACKEND=onnx`).

Репозиторий весов: [ShiWarai/sherpa-rknn-whisper-turbo](https://huggingface.co/ShiWarai/sherpa-rknn-whisper-turbo).

## Требования к железу и RAM

Профиль **`turbo`** (~1.7 GiB весов + пик запроса) рассчитан на платы с **≥8 GiB** RAM и достаточным `MemAvailable`. На 2–4 GiB используйте `base`/`small`.

Сервис отказывается загружать модель при старте и возвращает **507** на `/v1/audio/transcriptions`, если прогноз не влезает в `MemAvailable` (см. `WHISPER_MAX_AUDIO_SECONDS` в [api.md](api.md#переменные-окружения)).

Контейнер: `privileged: true`, `platform: linux/arm64`, порты на хост **не** публикуются. Размер образа ~**350 MB** (slim Python + PyAV wheel + apt ffmpeg, без PyTorch).

На хосте нужны устройства NPU (как в [video-descriptor-rkllm](https://github.com/ShiWarai/video-descriptor-rkllm)): `/dev/mpp_service`, `/dev/rga`, `/dev/dri`, `/dev/dma_heap` — проброшены в `docker-compose.yml`.

## third_party и mel-фильтры

| Путь | Назначение |
|------|------------|
| `third_party/librknnrt.so` | Runtime для NPU; копируется в `/usr/lib` при сборке образа |
| `third_party/rknn_toolkit_lite2-2.3.2-…-aarch64.whl` | Python-пакет rknnlite (версия зашита в `Dockerfile`) |
| `app/assets/mel_filters.npz` | Mel-фильтры Whisper (turbo 128 / base 80 mel), без `openai-whisper` |

Подробнее о runtime: [third_party/README.md](../third_party/README.md).

## Каталог на хосте

В `.env`:

```bash
WHISPER_MODELS_DIR=/mnt/nvme0/models/whisper-rknn-turbo
WHISPER_MODEL_PROFILE=turbo
WHISPER_LANGUAGE=ru
```

Compose монтирует каталог в `/models` (read-write, чтобы при автозагрузке файлы сохранялись на хосте).

## Язык распознавания (`WHISPER_LANGUAGE`)

Turbo — **мультиязычная** модель: перед декодированием в prompt подставляется language token (`ru` → `50263`, `en` → `50259` и т.д.). Маппинг встроен в `app/whisper_languages.py` (99 кодов Whisper, **без** зависимости от `openai-whisper`).

```bash
WHISPER_LANGUAGE=ru   # по умолчанию в compose
# WHISPER_LANGUAGE=en
```

Если язык не совпадает с аудио, результат может быть пустым, на другом языке или с повторами (например «What?» вместо русской речи). После смены языка перезапустите контейнер:

```bash
docker compose up -d --force-recreate
```

Коды — как в [Whisper](https://github.com/openai/whisper) (`ru`, `en`, `uk`, `de`, `zh`, …).

## Гибрид NPU + CPU

```
аудио → mel (CPU) → encoder.rknn (NPU) → cross_kv → decoder.onnx (CPU) → текст
```

`decoder.onnx` экспортирован из sherpa-onnx (`export_onnx.py --model turbo`); autoregressive decode на CPU через ONNX Runtime (~2× быстрее, чем decoder на NPU).

```bash
WHISPER_DECODER_BACKEND=onnx
WHISPER_DECODER=/models/decoder.onnx
```

## Длинное аудио

Окно encoder RKNN фиксировано (~30 с). Для подкастов, длинных голосовых и т.п. включена **нарезка с overlap** и склейка текста (см. [docs/api.md](api.md#длинное-аудио-30-с)).

По умолчанию overlap **2 с** — достаточно для стыковки фраз на границе окон. Уменьшить overlap (`0`) — быстрее, но выше риск обрезать слово на стыке.

## Автозагрузка turbo при старте

По аналогии с [`video-descriptor-rkllm`](https://github.com/ShiWarai/video-descriptor-rkllm): opt-in через `WHISPER_DOWNLOAD_MODELS`. Скрипт [`scripts/download_models.sh`](../scripts/download_models.sh) качает только недостающие файлы с Hugging Face.

```bash
WHISPER_DOWNLOAD_MODELS=turbo   # или 1
WHISPER_MODEL_PROFILE=turbo
WHISPER_LANGUAGE=ru
```

Entrypoint при автозагрузке выставляет `WHISPER_ENCODER`, `WHISPER_DECODER=/models/decoder.onnx`, `WHISPER_DECODER_BACKEND=onnx`, `WHISPER_TOKENS`, `WHISPER_MODEL_PROFILE=turbo`.

Свои URL: `WHISPER_MODEL_URLS=имя_файла=https://...` (через запятую или с новой строки).

```bash
MODELS_DIR=./models ./scripts/download_models.sh turbo
```

## Профиль модели (`WHISPER_MODEL_PROFILE`)

Для turbo с generic-именами (`encoder.rknn` без `turbo` в пути) профиль **обязателен**:

```bash
WHISPER_MODEL_PROFILE=turbo
```

По умолчанию в образе уже `turbo`. Профиль влияет на гиперпараметры декодера (число слоёв, mel-каналы и т.д.).

## Совместимость toolchain

1. `encoder.rknn` должен быть собран toolchain, совместимым с `librknnrt.so` из [`third_party/`](../third_party/README.md).
2. Целевая платформа inference: **RK3588** (`privileged: true` в compose для доступа к NPU).
3. Версия `rknn_toolkit_lite2` / `librknnrt.so` в образе: **2.3.2** ([airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2), см. `Dockerfile`).
4. Декод аудио: **PyAV** (libav API in-process, wheel в образе); fallback — CLI `ffmpeg` из `apt`. Устройства NPU в `docker-compose.yml` — для RKNN encoder.

## Раскладка файлов

```
/mnt/nvme0/models/whisper-rknn-turbo/
  encoder.rknn
  decoder.onnx
  tokens.txt
```

## Переменные окружения

Шаблон для `.env`: [`.env.example`](../.env.example). Переменные API, chunking и auth — в [api.md](api.md#переменные-окружения).

| Переменная | По умолчанию | Смысл |
|------------|--------------|-------|
| `WHISPER_ENCODER` | `/models/encoder.rknn` | Encoder RKNN (NPU) |
| `WHISPER_DECODER` | `/models/decoder.onnx` | Decoder (ONNX на CPU по умолчанию) |
| `WHISPER_DECODER_BACKEND` | `onnx` | `onnx` / `auto` (если есть `decoder.onnx`) |
| `WHISPER_TOKENS` | `/models/tokens.txt` | Токены |
| `WHISPER_DOWNLOAD_MODELS` | `0` | `turbo` или `1` — скачать turbo с HF при старте |
| `WHISPER_MODEL_URLS` | — | Свои URL: `file.rknn=https://...` |
| `WHISPER_MODEL_PROFILE` | `turbo` | Профиль декодера (для generic-имён `encoder.rknn`) |
| `WHISPER_LANGUAGE` | `ru` | Код Whisper (`ru`, `en`, `uk`, …). **Обязательно** под ваше аудио |
| `WHISPER_NPU_CORE_MASK` | `0_1_2` | Ядра NPU: `0`, `0_1`, `0_1_2`, `all`, `auto` |
| `WHISPER_MODELS_DIR` | — | Путь на хосте (volume в compose) |
| `LIBRKNNRT_SO` | — | Опциональный override пути к `.so` |
| `FFMPEG_BIN` | из `PATH` | Fallback CLI ffmpeg (основной путь — PyAV) |
