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
```

Compose монтирует каталог в `/models` (read-write, чтобы при автозагрузке файлы сохранялись на хосте).

## Автозагрузка turbo при старте

По аналогии с [`video-descriptor-rkllm`](https://github.com/ShiWarai/video-descriptor-rkllm): opt-in через `WHISPER_DOWNLOAD_MODELS`. Скрипт [`scripts/download_models.sh`](../scripts/download_models.sh) качает только недостающие файлы с [Hugging Face ShiWarai/sherpa-rknn-whisper-turbo](https://huggingface.co/ShiWarai/sherpa-rknn-whisper-turbo).

```bash
WHISPER_DOWNLOAD_MODELS=turbo   # или 1
WHISPER_MODEL_PROFILE=turbo
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
3. Версия `rknn_toolkit_lite2` в образе: **2.1.0** (см. `Dockerfile`).

## Раскладка файлов

```
/mnt/nvme0/models/whisper-rknn-turbo/
  encoder.rknn
  decoder.rknn
  tokens.txt
```
