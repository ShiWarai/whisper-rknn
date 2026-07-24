# Модели Whisper RKNN

Сервис ожидает три файла в каталоге, смонтированном в `/models`:

| Файл (пример) | Переменная | Описание |
|---------------|------------|----------|
| `encoder.rknn` | `WHISPER_ENCODER` | Encoder RKNN |
| `decoder.rknn` | `WHISPER_DECODER` | Decoder RKNN |
| `tokens.txt` | `WHISPER_TOKENS` | Словарь токенов Whisper |

Пути по умолчанию заданы в `Dockerfile` и `docker-compose.yml`. При других именах файлов переопределите переменные в `environment` compose.

## Каталог на хосте

В `.env`:

```bash
WHISPER_MODELS_DIR=/mnt/nvme0/models/whisper-rknn-turbo
```

Compose монтирует его как read-only volume: `${WHISPER_MODELS_DIR}:/models:ro`.

## Профиль модели (`WHISPER_MODEL_PROFILE`)

Если в пути к encoder **нет** подстроки `tiny`, `base`, `small`, `medium` или `turbo`, задайте профиль явно:

```bash
WHISPER_MODEL_PROFILE=turbo
```

Допустимые значения: `tiny`, `base`, `small`, `medium`, `turbo`.

Профиль влияет на гиперпараметры декодера (число слоёв, размер состояния, mel-каналы). Неверный профиль приведёт к некорректному распознаванию или ошибкам runtime.

Альтернатива: переменная `WHISPER_VARIANT` (то же значение, что `WHISPER_MODEL_PROFILE`).

## Совместимость toolchain

1. Модели `.rknn` должны быть собраны toolchain, совместимым с `librknnrt.so` из [`third_party/`](../third_party/README.md).
2. Целевая платформа inference: **RK3588** (`privileged: true` в compose для доступа к NPU).
3. Версия `rknn_toolkit_lite2` в образе: **2.1.0** (см. `Dockerfile`).

## Примеры раскладки файлов

**Turbo (имена без размера в пути):**

```
/mnt/nvme0/models/whisper-rknn-turbo/
  encoder.rknn
  decoder.rknn
  tokens.txt
```

`.env`: `WHISPER_MODEL_PROFILE=turbo`

**Base (размер в имени файла):**

```
/models/
  base-encoder.rknn
  base-decoder.rknn
  base-tokens.txt
```

Переопределите `WHISPER_ENCODER`, `WHISPER_DECODER`, `WHISPER_TOKENS` в compose.
