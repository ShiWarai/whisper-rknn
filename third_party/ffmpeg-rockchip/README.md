# ffmpeg-rockchip (vendored)

Prebuilt binaries for RK3588 hardware decode (rkmpp/rkrga).

| Field | Value |
|-------|-------|
| Upstream | [nyanmisaka/ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip) |
| Build tree | `/root/dev/ffmpeg` (host) |
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

- `bin/ffmpeg`, `bin/ffprobe` — from `/usr/local/bin`
- `lib/libav*.so*`, `lib/libsw*.so*` — FFmpeg shared libraries
- `lib/librga.so*`, `lib/librockchip_mpp.so*` — Rockchip MPP/RGA runtime

Host/container apt deps: `libdrm2`, `zlib1g`, `libopus0`, `libmp3lame0`, `libvorbis0a`, `libvorbisenc2`, `libssl3t64` (Ubuntu 24.04).

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
```
