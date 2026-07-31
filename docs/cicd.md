# CI/CD

Пайплайны в [`.github/workflows/`](../.github/workflows/). Сборка prod-образа **только** для `linux/arm64` (RK3588).

## Workflows

| Workflow | Триггер | Назначение |
|----------|---------|------------|
| **Deploy** (`deploy.yml`) | Push в `main` / `dev`, ручной запуск | ruff + pytest в dev-образе |
| **Deploy → prerelease** | Push в `dev` с `[prerelease]` в коммите, или ручной флаг `publish_prerelease` | Публикация `:prerelease` в GHCR |
| **Publish** (`publish.yml`) | Успешный Deploy на `main` | Публикация `:main` в GHCR |

## Образ GHCR

```
ghcr.io/shiwarai/whisper-rknn
```

Теги: `:main`, `:prerelease`, `:<git-sha>`.

## Prod vs prerelease vs локальная сборка

Три способа запустить **один и тот же** monolith (`docker-compose.yml`), отличается только источник образа:

| Способ | Compose | Образ | Откуда берётся |
|--------|---------|-------|----------------|
| **Локальная сборка** | `docker compose up -d --build` | `whisper-rknn-api:latest` | `Dockerfile` на плате |
| **Prerelease** | `-f docker-compose.yml -f docker-compose.prerelease.yml` | `:prerelease` | CI из ветки `dev` |
| **Prod** | `-f docker-compose.yml -f docker-compose.prod.yml` | `:main` | CI после merge в `main` |

Overlay-файлы `docker-compose.prod.yml` и `docker-compose.prerelease.yml` **идентичны по структуре**: подменяют `image` и отключают `build`. Никаких отдельных «prerelease-настроек» внутри приложения нет — только разный тег GHCR.

### `:prerelease` — тестовый кандидат

- Собирается при push в **`dev`** с `[prerelease]` в сообщении коммита, или вручную (Actions → Deploy → `publish_prerelease`).
- Нужен, чтобы **проверить на стенде** свежий код до merge в `main`.
- Тег **перезаписывается** при каждой новой prerelease-сборке.

```bash
git commit -m "feat: обновление API [prerelease]"
git push origin dev
```

На стенде:

```bash
docker pull ghcr.io/shiwarai/whisper-rknn:prerelease
docker compose -f docker-compose.yml -f docker-compose.prerelease.yml up -d
```

### `:main` — продакшен

- Публикуется **автоматически** workflow `publish.yml` после зелёного Deploy на ветке **`main`**.
- Это стабильная линия для продакшена.

```bash
docker pull ghcr.io/shiwarai/whisper-rknn:main
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Когда что использовать на одной RK3588

| Задача | Рекомендация |
|--------|--------------|
| Разработка на плате | `docker compose build && docker compose up -d` |
| Проверить CI-сборку до релиза | `:prerelease` |
| Боевой сервис | `:main` |

Distributed (`docker-compose.distributed.yml`) — отдельный сценарий для кластера; на одной машине monolith быстрее. См. [architecture.md](architecture.md).

## Кэш сборки

- **prerelease** — GHA cache scope `whisper-rknn-prerelease`
- **main** — scope `whisper-rknn-main`
- **test** (dev-образ) — scope `whisper-rknn-dev`

## Локальные тесты (как в CI)

```bash
docker compose -f docker-compose.dev.yml build dev
docker compose -f docker-compose.dev.yml run --rm -T dev ruff check .
docker compose -f docker-compose.dev.yml run --rm dev pytest tests/ -v --tb=short
```

Или: `./scripts/verify-stack.sh test`

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
- `third_party/` с `.whl` и `librknnrt.so` должен быть в git (для prod-сборки в CI).
- Prod-образ ~400 MB (`linux/arm64`): slim Python, без PyTorch/openai-whisper.

## Self-hosted runner

Не требуется: prod-сборка идёт на `ubuntu-latest` через QEMU + Buildx (`platforms: linux/arm64`).
