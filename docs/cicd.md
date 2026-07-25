# CI/CD

Пайплайны в [`.github/workflows/`](../.github/workflows/). Сборка prod-образа **только** для `linux/arm64` (RK3588).

## Workflows

| Workflow | Триггер | Назначение |
|----------|---------|------------|
| **Deploy** (`deploy.yml`) | Push в `main` / `dev`, ручной запуск | Dev-образ, ruff + pytest |
| **Deploy → prerelease** | Push в `dev` с `[prerelease]` в коммите, или ручной флаг `publish_prerelease` | Образ `:prerelease` в GHCR |
| **Publish** (`publish.yml`) | Успешный Deploy на `main` | Образ `:main` в GHCR |

## Образ GHCR

```
ghcr.io/shiwarai/whisper-rknn
```

Теги: `:main`, `:prerelease`, `:<git-sha>`.

## Prerelease

**Автоматически** — коммит в `dev` с меткой в сообщении:

```bash
git commit -m "feat: обновление API [prerelease]"
git push origin dev
```

**Вручную** — Actions → Deploy → Run workflow → включить `publish_prerelease`.

Сборка prerelease использует **GHA cache** BuildKit (`cache-from` / `cache-to`, scope `whisper-rknn-prerelease`) — повторные сборки быстрее, пока не меняются слои `Dockerfile` / `requirements.txt`.

Job **test** (dev-образ `Dockerfile.dev`) — тот же механизм, scope `whisper-rknn-dev` (pip-слои кэшируются между push в `main` / `dev`). Для cache-to нужен Buildx driver `docker-container` (скачивает `moby/buildkit` с Docker Hub); при таймауте Hub setup ретраится, затем fallback на `driver: docker` без записи кэша — тесты всё равно идут.

Workflow **Publish** (`:main`) — scope `whisper-rknn-main`.

На тестовом стенде:

```bash
docker pull ghcr.io/shiwarai/whisper-rknn:prerelease
docker compose -f docker-compose.yml -f docker-compose.prerelease.yml up -d
```

## Stable (main)

После merge в `main` и зелёного Deploy автоматически публикуется `:main`:

```bash
docker pull ghcr.io/shiwarai/whisper-rknn:main
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Локальные тесты (как в CI)

```bash
docker compose -f docker-compose.dev.yml build dev
docker compose -f docker-compose.dev.yml run --rm -T dev ruff check .
docker compose -f docker-compose.dev.yml run --rm dev pytest tests/ -v --tb=short
```

## Telegram-уведомления

Secrets репозитория (Settings → Secrets and variables → Actions):

| Secret | Назначение |
|--------|------------|
| `TELEGRAM_TOKEN` | Токен бота |
| `TELEGRAM_TO` | Chat ID |

Без secrets шаги уведомлений не падают (`continue-on-error: true`). Успешные события — тихие (`disable_notification`).

## Требования к репозиторию

- Включены **GitHub Actions** и **Packages** (GHCR).
- Для публичного образа: visibility пакета `whisper-rknn` → Public (при необходимости).
- `third_party/` с `.whl`, `librknnrt.so` и `ffmpeg-rockchip/` должен быть в git (для prod-сборки в CI).
- Prod-образ ~400 MB (`linux/arm64`): slim Python, без PyTorch/openai-whisper.

## Self-hosted runner

Не требуется: prod-сборка идёт на `ubuntu-latest` через QEMU + Buildx (`platforms: linux/arm64`).
