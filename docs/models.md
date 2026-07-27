# Модели Whisper RKNN

Проект рассчитан на **Whisper turbo** (RKNN). Три файла в каталоге, смонтированном в `/models`:

| Файл | Переменная | Описание |
|------|------------|----------|
| `encoder.rknn` | `WHISPER_ENCODER` | Encoder RKNN |
| `decoder.rknn` | `WHISPER_DECODER` | Decoder RKNN |
| `tokens.txt` | `WHISPER_TOKENS` | Словарь токенов Whisper |

Пути по умолчанию заданы в `Dockerfile` и `docker-compose.yml`.

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

## Длинное аудио

Окно encoder RKNN фиксировано (~30 с). Для подкастов, длинных голосовых и т.п. включена **нарезка с overlap** и склейка текста (см. [docs/api.md](api.md#длинное-аудио-30-с)).

По умолчанию overlap **2 с** — достаточно для стыковки фраз на границе окон. Уменьшить overlap (`0`) — быстрее, но выше риск обрезать слово на стыке.

Для гибридного режима (NPU encoder + CPU ONNX decoder) в репозитории `sherpa-rknn-whisper-turbo` на Hugging Face лежит `decoder.onnx`; при `WHISPER_DOWNLOAD_MODELS=turbo` он скачивается вместе с `.rknn`.

## Автозагрузка turbo при старте

По аналогии с [`video-descriptor-rkllm`](https://github.com/ShiWarai/video-descriptor-rkllm): opt-in через `WHISPER_DOWNLOAD_MODELS`. Скрипт [`scripts/download_models.sh`](../scripts/download_models.sh) качает только недостающие файлы с [Hugging Face ShiWarai/sherpa-rknn-whisper-turbo](https://huggingface.co/ShiWarai/sherpa-rknn-whisper-turbo).

```bash
WHISPER_DOWNLOAD_MODELS=turbo   # или 1
WHISPER_MODEL_PROFILE=turbo
WHISPER_LANGUAGE=ru
```

Entrypoint выставляет `WHISPER_ENCODER` / `WHISPER_DECODER` / `WHISPER_TOKENS` и `WHISPER_MODEL_PROFILE=turbo`.

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

1. Модели `.rknn` должны быть собраны toolchain, совместимым с `librknnrt.so` из [`third_party/`](../third_party/README.md).
2. Целевая платформа inference: **RK3588** (`privileged: true` в compose для доступа к NPU).
3. Версия `rknn_toolkit_lite2` / `librknnrt.so` в образе: **2.3.2** ([airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2), см. `Dockerfile`).
4. Декод аудио: **PyAV** (libav API in-process, wheel в образе); fallback — CLI `ffmpeg` из `apt`. Устройства NPU в `docker-compose.yml` — для RKNN inference.

## Раскладка файлов

```
/mnt/nvme0/models/whisper-rknn-turbo/
  encoder.rknn
  decoder.rknn
  tokens.txt
```
