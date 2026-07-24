# third_party — Rockchip RKNN runtime для сборки Docker-образа

Каталог содержит бинарные артефакты Rockchip, необходимые **только при сборке** prod-образа (`Dockerfile`). В git они закоммичены для воспроизводимой CI/CD-сборки на `linux/arm64`.

| Файл | Назначение |
|------|------------|
| `rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl` | Python-пакет `rknn_toolkit_lite2` (версия **2.1.0**, CPython 3.10, aarch64) |
| `librknnrt.so` | Runtime-библиотека RKNN; копируется в образ как `/usr/lib/librknnrt.so` |

## Совместимость

- **Мажорная версия** `librknnrt.so` и toolkit должна совпадать с toolchain, которым собраны ваши `.rknn` модели (encoder/decoder).
- Образ рассчитан на **Rockchip RK3588** (NPU). Другие SoC — только при совместимом runtime и пересборке моделей.

## Лицензия и распространение

Артефакты Rockchip RKNN SDK распространяются по [лицензии Rockchip](https://github.com/rockchip-linux/rknn-toolkit2). Использование runtime на устройствах Rockchip — в рамках их условий. Код приложения в этом репозитории — MIT (см. корневой `LICENSE`).

При обновлении SDK замените оба файла и поправьте путь к `.whl` в `Dockerfile`, если изменилось имя wheel.
