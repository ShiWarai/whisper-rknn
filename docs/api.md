# API

HTTP API сервиса `whisper-rknn-api` (FastAPI). По умолчанию слушает `0.0.0.0:8080` **внутри** Docker-сети (порты на хост не публикуются). Порт переопределяется через `PORT` в `.env` (для связки с ботом — `9003`).

Язык распознавания задаётся **не** в запросе, а переменной окружения **`WHISPER_LANGUAGE`** в `.env` сервиса (по умолчанию `ru`). См. [docs/models.md](models.md).

Базовый URL в compose-сети: `http://whisper-rknn-api:${PORT}` (пример ниже — с `PORT=9003`).

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

Поддерживаемые форматы (через ffmpeg): ogg, wav, mp3, m4a, flac, opus, webm и др.

**Лимит размера:** `MAX_UPLOAD_MB` (по умолчанию 25 MB).

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
  -s -F "file=@/data/voice.ogg" \
  http://whisper-rknn-api:9003/transcribe
```

## OpenAPI

При локальной отладке с пробросом порта: `http://localhost:${PORT}/docs`.
