# API

HTTP API сервиса `whisper-rknn-api` (FastAPI). По умолчанию слушает `0.0.0.0:8080` **внутри** Docker-сети (порты на хост не публикуются). Порт переопределяется через `PORT` в `.env` (для связки с ботом — `9003`).

Язык распознавания задаётся **не** в запросе, а переменной окружения **`WHISPER_LANGUAGE`** в `.env` сервиса (по умолчанию `ru`). См. [docs/models.md](models.md).

Базовый URL в compose-сети: `http://whisper-rknn-api:${PORT}` (пример ниже — с `PORT=9003`).

## Авторизация (OpenAI-совместимая)

Если задан **`WHISPER_API_KEY`** (или **`OPENAI_API_KEY`** как alias), эндпоинт `POST /transcribe` требует заголовок:

```http
Authorization: Bearer <ваш_ключ>
```

Несколько ключей: через запятую в `WHISPER_API_KEY` (`key1,key2`).

`GET /health` остаётся без авторизации (healthcheck / мониторинг).

Если ключ **не задан**, API открыт (как раньше) — удобно для изолированной Docker-сети.

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

Загруженный файл декодируется **in-process** через **PyAV**, собранный против vendored **ffmpeg-rockchip** (`libav` из `third_party/`), в **16 kHz mono float32** в RAM (без промежуточного WAV на диске).

Fallback: `soundfile` (WAV/FLAC), CLI `ffmpeg` → `f32le` pipe. Переопределение fallback-бинарника: `FFMPEG_BIN` в `.env`.

## `GET /health`

Проверка готовности модели.

**Ответ 200**

```json
{ "status": "ok" }
```

Пока модель загружается при старте:

```json
{ "status": "loading" }
```

## `POST /transcribe`

Распознавание речи из загруженного аудиофайла.

**Тело:** `multipart/form-data`, поле **`file`**.

Поддерживаемые форматы (через PyAV / ffmpeg-rockchip): ogg, wav, mp3, m4a, flac, opus, webm и др.

**Лимит размера:** `MAX_UPLOAD_MB` (по умолчанию 25 MB).

### Длинное аудио (>30 с)

Статическое окно RKNN — **3000 mel-кадров (~30 с)**. Более длинные файлы:

1. Режутся на скользящие окна того же размера (вход encoder ≤ 3000 кадров).
2. Каждое окно декодируется отдельно.
3. Тексты **склеиваются** с удалением дубликата на стыке (suffix/prefix по словам).

| Переменная | По умолчанию | Смысл |
|------------|--------------|-------|
| `WHISPER_CHUNK_SECONDS` | ~30 (окно модели) | Длина окна в секундах (не больше 30) |
| `WHISPER_CHUNK_OVERLAP_SECONDS` | `5` | Перекрытие соседних окон; `0` — без overlap |
| `WHISPER_MAX_NGRAM_REPEAT` | `6` | Стоп при зацикливании повторяющихся токенов |

Переменные задаются в `.env` (пробрасываются через `docker-compose.yml`).

**Ответ 200**

```json
{
  "text": "распознанный текст",
  "elapsed_s": 1.234
}
```

**Ошибки**

| Код | Причина |
|-----|---------|
| 413 | Файл слишком большой |
| 401 | Нет или неверный `Authorization: Bearer` (если задан `WHISPER_API_KEY`) |
| 503 | Модель ещё не загружена |
| 400 | Некорректное аудио или ошибка декодирования |

## Примеры

Из контейнера в той же сети `whisper_rknn_default`:

```bash
docker run --rm --network whisper_rknn_default curlimages/curl:latest \
  -s http://whisper-rknn-api:9003/health

docker run --rm --network whisper_rknn_default \
  -v /path/to/audio:/data:ro \
  curlimages/curl:latest \
  -s -H "Authorization: Bearer $WHISPER_API_KEY" \
  -F "file=@/data/voice.ogg" \
  http://whisper-rknn-api:9003/transcribe
```

## OpenAPI

При локальной отладке с пробросом порта: `http://localhost:${PORT}/docs`.
