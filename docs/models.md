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

На хосте для `librknnrt` нужны `/dev/dri` и `/dev/dma_heap` — проброшены в `docker-compose.yml` (MPP/RGA для ASR не требуются).

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
WHISPER_MODELS_DIR=/mnt/nvme0/models/sherpa-rknn-whisper-turbo
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

`decoder.onnx` — dynamic INT8 (MatMul, QInt8) из sherpa-onnx export; autoregressive decode на CPU через ONNX Runtime (~2× быстрее decoder на NPU, ~10% быстрее FP32 decoder на CPU).

```bash
WHISPER_DECODER_BACKEND=onnx
WHISPER_DECODER=/models/decoder.onnx
```

## Параллельный encode + VAD-нарезка (RAM)

Для аудио длиннее окна encoder (~30 с): **Silero VAD** режет по паузам без речи, **encoder pool** (до N× `encoder.rknn` на dedicated NPU cores) кодирует параллельно, **decoder.onnx** на CPU — последовательно. VAD ONNX грузится при старте API (вместе с encoder pool). Всё в ОЗУ.

**Silero VAD (`silero_vad.onnx`):** при `WHISPER_DOWNLOAD_MODELS=turbo` кладётся в `/models` вместе с whisper-весами (`download_models.sh`). Иначе API ищет файл в порядке: `WHISPER_VAD_MODEL` → `/models/silero_vad.onnx` → скачивает в `.cache/silero_vad.onnx` при первом старте (нужен исходящий HTTP).

```bash
WHISPER_PARALLEL_ENCODE=1
WHISPER_ENCODER_WORKERS=0    # 0 = auto N→…→1 по MemAvailable (+ NPU probe)
# WHISPER_ENCODER_MAX_WORKERS=3  # потолок (по умолчанию = число NPU_CORE_N в rknnlite)
WHISPER_MAX_CHUNK_SECONDS=30
WHISPER_VAD_THRESHOLD=0.5
WHISPER_VAD_SEARCH_BACK_SEC=3
WHISPER_VAD_MIN_GAP_MS=250
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
/mnt/nvme0/models/sherpa-rknn-whisper-turbo/
  encoder.rknn
  decoder.onnx          # INT8
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
| `WHISPER_NPU_CORE_MASK` | `0_1_2` | Подмножество NPU cores для encoder pool / single-encoder (`0`, `0_1`, `0_1_2`, …) |
| `WHISPER_PARALLEL_ENCODE` | `1` | VAD + parallel encoder pool |
| `WHISPER_ENCODER_WORKERS` | `0` | `0` = auto (MemAvailable + NPU probe); `1..N` = force ceiling |
| `WHISPER_ENCODER_MAX_WORKERS` | число разрешённых NPU cores | Потолок воркеров encoder pool |
| `WHISPER_CPU_AFFINITY` | — | CPU воркера: `4,5,6` или `4-7` (encoder/decoder/monolith; не gateway) |
| `WHISPER_ONNX_INTRA_OP_THREADS` | `0` | Потоки ONNX decoder: `0`/`auto` = видимые CPU; число = потолок |
| `WHISPER_MAX_CHUNK_SECONDS` | `30` | Макс. длина VAD-чанка |
| `WHISPER_VAD_THRESHOLD` | `0.5` | Порог Silero VAD |
| `WHISPER_VAD_SEARCH_BACK_SEC` | `3` | Окно поиска паузы перед лимитом decode |
| `WHISPER_VAD_MIN_GAP_MS` | `250` | Мин. длина non-speech для среза decode-окна (~30 с) |
| `WHISPER_VAD_SEGMENT_MAX_SECONDS` | `5` | Макс. длина сегмента в `verbose_json`/`srt`/`vtt` |
| `WHISPER_VAD_SEGMENT_MIN_GAP_MS` | `100` | Мин. тишина для мелких timing-сегментов |
| `WHISPER_VAD_SEGMENT_SEARCH_BACK_FRAMES` | `16` | Поиск паузы у лимита мелкого сегмента (кадры Silero ×32 мс) |
| `WHISPER_VAD_MODEL` | — | Путь к `silero_vad.onnx` (иначе `/models/silero_vad.onnx`, иначе auto-download в `.cache/`) |
| `WHISPER_VAD_MODEL_URL` | GitHub silero-vad | URL для auto-download / `download_models.sh` |
| `WHISPER_MODELS_DIR` | — | Путь на хосте (volume в compose) |
| `LIBRKNNRT_SO` | — | Опциональный override пути к `.so` |
| `FFMPEG_BIN` | из `PATH` | Fallback CLI ffmpeg (основной путь — PyAV) |
