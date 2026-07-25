#!/usr/bin/env python3
"""Benchmark decode_utterance under different WHISPER_NPU_CORE_MASK values."""
from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("WHISPER_MODEL_PROFILE", "turbo")
os.environ.setdefault("WHISPER_LANGUAGE", "ru")

from app.decode import (  # noqa: E402
    RKNNModel,
    decode_utterance,
    load_tokens,
    model_config_from_encoder_path,
    prepare_audio_16k_mono,
    resolve_npu_core_mask,
)

AUDIO = os.environ.get("BENCH_AUDIO", "/tmp/stepan_whisper.ogg")
ENCODER = os.environ.get("WHISPER_ENCODER", "/models/encoder.rknn")
DECODER = os.environ.get("WHISPER_DECODER", "/models/decoder.rknn")
TOKENS = os.environ.get("WHISPER_TOKENS", "/models/tokens.txt")
MASKS = [m.strip() for m in os.environ.get("BENCH_MASKS", "0,0_1,0_1_2,auto").split(",") if m.strip()]
RUNS = int(os.environ.get("BENCH_RUNS", "3"))


def main() -> None:
    wav, tmp = prepare_audio_16k_mono(AUDIO, Path("/tmp"))
    id2token = load_tokens(TOKENS)
    cfg = model_config_from_encoder_path(ENCODER)
    sot, eot, n_layer, n_ctx, n_state, n_mels, mel_frames = cfg

    print(f"audio={AUDIO}")
    print(f"runs_per_mask={RUNS} (first is warmup, excluded from mean)")
    print()

    results = []
    try:
        for mask in MASKS:
            os.environ["WHISPER_NPU_CORE_MASK"] = mask
            mask_val = resolve_npu_core_mask(mask)
            print(f"=== mask={mask} (value={mask_val}) load ===", flush=True)
            t0 = time.perf_counter()
            model = RKNNModel(
                encoder=ENCODER,
                decoder=DECODER,
                sot_sequence=sot,
                eot=eot,
                n_text_layer=n_layer,
                n_text_ctx=n_ctx,
                n_text_state=n_state,
                n_mels=n_mels,
                mel_time_frames=mel_frames,
                verbose=False,
            )
            load_s = time.perf_counter() - t0
            times = []
            texts = []
            for i in range(RUNS):
                t1 = time.perf_counter()
                text = decode_utterance(model, id2token, wav, verbose=False)
                dt = time.perf_counter() - t1
                times.append(dt)
                texts.append(text)
                print(f"  run {i + 1}/{RUNS}: {dt:.3f}s  text={text[:80]!r}...", flush=True)
            model.release()
            measured = times[1:] if len(times) > 1 else times
            mean = statistics.mean(measured)
            results.append((mask, load_s, times, mean, texts[-1]))
            print(f"  load={load_s:.2f}s  mean(excl warmup)={mean:.3f}s\n", flush=True)
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)

    print("SUMMARY (decode only, after model load)")
    print(f"{'mask':<10} {'load_s':>8} {'runs_s':>28} {'mean_s':>8}")
    for mask, load_s, times, mean, _ in results:
        runs = " ".join(f"{t:.2f}" for t in times)
        print(f"{mask:<10} {load_s:8.2f} {runs:>28} {mean:8.3f}")


if __name__ == "__main__":
    main()
