# third_party — Rockchip RKNN runtime для Docker

Каталог содержит бинарные артефакты, необходимые **только при сборке** prod-образа (`Dockerfile`). В git закоммичены для воспроизводимой CI/CD-сборки на `linux/arm64`.

Аудио декодируется **in-process** через **PyAV** (`pip av`, bundled libav в wheel). CLI `apt ffmpeg` — только аварийный fallback.

## RKNN (NPU)

| Файл | Назначение |
|------|------------|
| `rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | Python-пакет `rknn_toolkit_lite2` (версия **2.3.2**, CPython 3.10, aarch64) |
| `librknnrt.so` | Runtime-библиотека RKNN 2.3.2; копируется в образ как `/usr/lib/librknnrt.so` |

## Совместимость

- Версия `librknnrt.so` / toolkit-lite2 должна совпадать с toolchain, которым собраны ваши `.rknn` (у turbo: toolkit **2.3.2**).
- Образ рассчитан на **Rockchip RK3588** (NPU).

## Обновление

**RKNN:** свежие пакеты — [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) → `rknpu2/runtime/Linux/.../librknnrt.so` и `rknn-toolkit-lite2/packages/*.whl`. Замените файлы здесь и путь к `.whl` в `Dockerfile`.

## Лицензия и распространение

Артефакты Rockchip RKNN SDK — по лицензии Rockchip. Код приложения в этом репозитории — MIT (см. корневой `LICENSE`).
