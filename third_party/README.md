# third_party — Rockchip RKNN runtime и ffmpeg-rockchip для Docker

Каталог содержит бинарные артефакты, необходимые **только при сборке** prod-образа (`Dockerfile`). В git закоммичены для воспроизводимой CI/CD-сборки на `linux/arm64`.

## RKNN (NPU)

| Файл | Назначение |
|------|------------|
| `rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | Python-пакет `rknn_toolkit_lite2` (версия **2.3.2**, CPython 3.10, aarch64) |
| `librknnrt.so` | Runtime-библиотека RKNN 2.3.2; копируется в образ как `/usr/lib/librknnrt.so` |

## ffmpeg-rockchip (MPP/RGA)

Vendored из [video-descriptor-rkllm](https://github.com/ShiWarai/video-descriptor-rkllm) (`third_party/ffmpeg-rockchip/`): аппаратный декод на RK3588 ([nyanmisaka/ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip)).

| Путь | Назначение |
|------|------------|
| `ffmpeg-rockchip/include/` | libav headers для сборки PyAV |
| `ffmpeg-rockchip/lib/pkgconfig/` | `.pc` для `PKG_CONFIG_PATH` |
| `ffmpeg-rockchip/lib/*.so` | FFmpeg + `librockchip_mpp`, `librga` |
| `ffmpeg-rockchip/bin/ffmpeg`, `ffprobe` | Fallback CLI (основной путь — PyAV in-process) |

В образе: `LD_LIBRARY_PATH` указывает на `lib/`; PyAV собирается против этих `.so`. В compose проброшены `/dev/mpp_service`, `/dev/rga`, `/dev/dri`, `/dev/dma_heap`.

Переопределить fallback-бинарник: `FFMPEG_BIN=/path/to/ffmpeg`.

## Совместимость

- Версия `librknnrt.so` / toolkit-lite2 должна совпадать с toolchain, которым собраны ваши `.rknn` (у turbo: toolkit **2.3.2**).
- Образ рассчитан на **Rockchip RK3588** (NPU).

## Обновление

**RKNN:** свежие пакеты — [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) → `rknpu2/runtime/Linux/.../librknnrt.so` и `rknn-toolkit-lite2/packages/*.whl`. Замените файлы здесь и путь к `.whl` в `Dockerfile`.

**ffmpeg-rockchip:** скопируйте каталог `third_party/ffmpeg-rockchip/` из ветки `dev` репозитория [video-descriptor-rkllm](https://github.com/ShiWarai/video-descriptor-rkllm) (или пересоберите по [ffmpeg-rockchip/README.md](ffmpeg-rockchip/README.md)). После замены пересоберите образ.

## Лицензия и распространение

Артефакты Rockchip RKNN SDK — по лицензии Rockchip. ffmpeg-rockchip — GPL v3 (см. upstream [nyanmisaka/ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip)). Код приложения в этом репозитории — MIT (см. корневой `LICENSE`).
