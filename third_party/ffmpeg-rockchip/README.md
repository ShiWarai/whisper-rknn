# ffmpeg-rockchip (vendored)

Prebuilt binaries and headers for RK3588 hardware decode (rkmpp/rkrga).

| Field | Value |
|-------|-------|
| Upstream | [nyanmisaka/ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip) |
| Build tree | См. [video-descriptor-rkllm](https://github.com/ShiWarai/video-descriptor-rkllm) `third_party/ffmpeg-rockchip/` |
| Platform | linux/arm64 (RK3588) |

## Configure

```text
--prefix=/usr/local
--enable-gpl --enable-version3
--enable-libdrm --enable-rkmpp --enable-rkrga
--enable-libopus --enable-libmp3lame --enable-libvorbis --enable-openssl
--enable-shared --disable-static --disable-doc
```

## Contents

- `include/` — vendored libav* headers (для сборки PyAV против pinned `.so`)
- `lib/pkgconfig/` — `.pc` для `PKG_CONFIG_PATH` при `pip install av --no-binary av`
- `bin/ffmpeg`, `bin/ffprobe` — fallback CLI (если PyAV недоступен)
- `lib/libav*.so*`, `lib/libsw*.so*` — FFmpeg shared libraries
- `lib/librga.so*`, `lib/librockchip_mpp.so*` — Rockchip MPP/RGA runtime

В образе whisper-rknn аудио декодируется **in-process** через PyAV, линкуемый к этим `.so`. CLI `ffmpeg` — только аварийный fallback.

Host/container apt deps: `libdrm2`, `zlib1g`, `libopus0`, `libmp3lame0`, `libvorbis0a`, `libvorbisenc2`, `libssl3t64` (Ubuntu 24.04).

Builder image additionally needs `-dev` packages for linking PyAV: `build-essential`, `pkg-config`, `libdrm-dev`, `libopus-dev`, `libmp3lame-dev`, `libvorbis-dev`, `libssl-dev`.

## Rebuild (host)

```bash
cd /root/dev/ffmpeg
sudo apt-get install -y libopus-dev libmp3lame-dev libvorbis-dev libssl-dev pkg-config
./configure --prefix=/usr/local \
  --enable-gpl --enable-version3 \
  --enable-libdrm --enable-rkmpp --enable-rkrga \
  --enable-libopus --enable-libmp3lame --enable-libvorbis --enable-openssl \
  --enable-shared --disable-static --disable-doc
make -j"$(nproc)" && sudo make install
cp -a /usr/local/bin/ffmpeg /usr/local/bin/ffprobe bin/
cp -a /usr/local/lib/libav*.so* /usr/local/lib/libsw*.so* lib/
cp -a /usr/local/include/libav{codec,format,filter,util,device} /usr/local/include/libsw{scale,resample} include/
```
