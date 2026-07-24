# CI/CD

Пайплайны в [`.github/workflows/`](../.github/workflows/). Сборка prod-образа **только** для `linux/arm64` (RK3588).

## Workflows

| Workflow | Триггер | Назначение |
|----------|---------|------------|
| **Deploy** (`deploy.yml`) | Push в `main` / `dev`, ручной запуск | Dev-образ, ruff + pytest |
| **Deploy → prerelease** | Push в `dev` с `[prerelease]` в коммите, или ручной флаг `publish_prerelease` | Образ `:prerelease` в GHCR |
| **Publish** (`publish.yml`) | Успешный Deploy на `main` | Образ `:main` в GHCR |
| **Release** (`release.yml`) | Push тега `v*.*.*` | Образ `:vX.Y.Z`, GitHub Release + deploy zip |

## Образ GHCR

```
ghcr.io/shiwarai/whisper-rknn
```

Теги: `:main`, `:prerelease`, `:vX.Y.Z`, `:<git-sha>`.

## Prerelease

**Автоматически** — коммит в `dev` с меткой в сообщении:

```bash
git commit -m "feat: обновление API [prerelease]"
git push origin dev
```

**Вручную** — Actions → Deploy → Run workflow → включить `publish_prerelease`.

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

## Semver-релиз

```bash
git tag v0.1.0
git push origin v0.1.0
```

Workflow **Release** соберёт образ `ghcr.io/shiwarai/whisper-rknn:v0.1.0` и создаст GitHub Release с zip (`docker-compose`, `.env.example`, `DEPLOY.md`).

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
- `third_party/` с `.whl` и `librknnrt.so` должен быть в git (для prod-сборки в CI).

## Self-hosted runner

Не требуется: prod-сборка идёт на `ubuntu-latest` через QEMU + Buildx (`platforms: linux/arm64`).
