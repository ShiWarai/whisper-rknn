# API

HTTP API сервиса `whisper-rknn-api` (FastAPI). Контракт совместим с [hwdsl2/docker-whisper](https://github.com/hwdsl2/docker-whisper) (OpenAI Audio API): один и тот же клиент может переключаться между NPU (`whisper-rknn`) и CPU (`hwdsl2/whisper-server`) сменой контейнера.

По умолчанию слушает `0.0.0.0:8080` **внутри** Docker-сети. Порт переопределяется через `PORT` в `.env` (для связки с ботом — `9003`).

Базовый URL в compose-сети: `http://whisper-rknn-api:${PORT}` (пример ниже — с `PORT=9003`).

OpenAI SDK / клиенты:

```bash
export OPENAI_BASE_URL=http://whisper-rknn-api:9003/v1
export OPENAI_API_KEY=your-key-or-any-non-empty
```

## Авторизация (OpenAI-совместимая)

Если задан **`WHISPER_API_KEY`** (или **`OPENAI_API_KEY`** как alias), защищённые эндпоинты требуют заголовок:

```http
Authorization: Bearer <ваш_ключ>
```

Несколько ключей: через запятую в `WHISPER_API_KEY` (`key1,key2`).

`GET /health` остаётся без авторизации (healthcheck / мониторинг).

Если ключ **не задан**, API открыт — удобно для изолированной Docker-сети.

**401** (тело в стиле OpenAI):

```json
{
  "detail": {
    "error": {
      "message": "Incorrect API key provided: your_key",
      "type": "invalid_request_error",
      "param": null,
      "code": "invalid_api_key"
    }
  }
}
```

## Предобработка аудио

Загруженный файл декодируется **in-process** через **PyAV** (libav API) в **16 kHz mono float32** в RAM.

Fallback: `soundfile` (WAV/FLAC), CLI `ffmpeg` → `f32le` pipe. Переопределение: `FFMPEG_BIN` в `.env`.

## `GET /health`

Проверка готовности модели.

**Ответ 200**

```json
{ "status": "ok", "model": "turbo" }
```

Пока модель загружается:

```json
{ "status": "loading" }
```

## `GET /v1/models`

Список активной модели (OpenAI-совместимый формат). Требует Bearer, если задан `WHISPER_API_KEY`.

```json
{
  "object": "list",
  "data": [
    {
      "id": "turbo",
      "object": "model",
      "created": 0,
      "owned_by": "whisper-rknn"
    }
  ]
}
```

## `POST /v1/audio/transcriptions`

Распознавание речи из загруженного аудиофайла.

**Тело:** `multipart/form-data`

| Поле | Тип | По умолчанию | Смысл |
|------|-----|--------------|-------|
| `file` | file | — | Аудиофайл |
| `model` | string | `whisper-1` | Принимается, фактически используется профиль из `WHISPER_MODEL_PROFILE` |
| `language` | string | из `WHISPER_LANGUAGE` | BCP-47 / код Whisper (`ru`, `en`, …); `auto` — как в env |
| `prompt` | string | — | Принимается для совместимости, RKNN-декодер пока игнорирует |
| `response_format` | string | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | `0` | Принимается (RKNN greedy) |
| `stream` | bool | `false` | SSE-поток по протоколу OpenAI |
| `timestamp_granularities[]` | array | `segment` | Только `segment`; `word` → **400** |

Поддерживаемые форматы: ogg, wav, mp3, m4a, flac, opus, webm и др.

**Лимит размера:** `MAX_UPLOAD_MB` (по умолчанию 25 MB).

### `response_format`

| Значение | Ответ |
|----------|-------|
| `json` | `{"text": "..."}` |
| `text` | plain text |
| `verbose_json` | JSON с `task`, `language`, `duration`, `text`, `segments[]` |
| `srt` | SubRip |
| `vtt` | WebVTT |

Для `verbose_json`, `srt`, `vtt` декодер включает сегментные метки времени Whisper.

### Streaming (`stream=true`)

Ответ: `text/event-stream` (SSE), протокол OpenAI:

```
data: {"type":"transcript.text.delta","delta":"..."}

data: {"type":"transcript.text.done","text":"полный текст"}

data: [DONE]
```

На RKNN дельты приходят после каждого chunk-окна (~30 с).

## `POST /v1/audio/translations`

Перевод аудио на английский. Те же параметры, что у transcriptions (кроме `timestamp_granularities` на translations — всегда segment).

Не поддерживается для English-only моделей (`.en`) — **400**.

## Защита от OOM (RAM)

Перед декодированием сервис сравнивает оценку пика RAM с **`MemAvailable`**. При нехватке — **507**.

Оценка PCM ограничена `WHISPER_MAX_AUDIO_SECONDS` (по умолчанию `600` с).

### Длинное аудио (>30 с)

Статическое окно RKNN — **3000 mel-кадров (~30 с)**. Более длинные файлы режутся на скользящие окна и склеиваются.

| Переменная | По умолчанию | Смысл |
|------------|--------------|-------|
| `WHISPER_CHUNK_SECONDS` | ~30 | Длина окна (не больше 30) |
| `WHISPER_CHUNK_OVERLAP_SECONDS` | `5` | Перекрытие окон |
| `WHISPER_MAX_NGRAM_REPEAT` | `6` | Стоп при зацикливании токенов |

## Ошибки

| Код | Причина |
|-----|---------|
| 413 | Файл слишком большой |
| 507 | Недостаточно свободной RAM (NPU-расширение) |
| 401 | Нет или неверный Bearer |
| 503 | Модель ещё не загружена |
| 400 | Некорректное аудио, неподдерживаемый параметр |

## Примеры

```bash
# health
docker run --rm --network whisper_rknn_default curlimages/curl:latest \
  -s http://whisper-rknn-api:9003/health

# транскрипция (json)
docker run --rm --network whisper_rknn_default \
  -v /path/to/audio:/data:ro \
  curlimages/curl:latest \
  -s -H "Authorization: Bearer $WHISPER_API_KEY" \
  -F "file=@/data/voice.ogg" \
  -F "model=whisper-1" \
  -F "language=ru" \
  http://whisper-rknn-api:9003/v1/audio/transcriptions

# verbose_json с сегментами
curl -s -H "Authorization: Bearer $WHISPER_API_KEY" \
  -F "file=@voice.ogg" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  http://whisper-rknn-api:9003/v1/audio/transcriptions

# streaming
curl -s -F "file=@voice.ogg" -F "model=whisper-1" -F "stream=true" \
  http://whisper-rknn-api:9003/v1/audio/transcriptions
```

## CPU ↔ NPU

| Контейнер | Образ | Железо |
|-----------|-------|--------|
| NPU | `ghcr.io/shiwarai/whisper-rknn` | RK3588 |
| CPU | `hwdsl2/whisper-server` | faster-whisper |

Клиентский URL и multipart-поля одинаковые; меняется только hostname/compose-сервис.

## OpenAPI

При локальной отладке с пробросом порта: `http://localhost:${PORT}/docs`.
