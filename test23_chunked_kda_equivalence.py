
import time
import json
import statistics
from pathlib import Path

import torch

from model.kda_attention_v3 import (
    kda_recurrence_script,
    kda_recurrence_chunked,
)


# ============================================================
# TEST23 — KDA triangular solve benchmark
# ============================================================

SEED = 20261009
BATCH = 2
HEADS = 8
SEQ_LEN = 128
HEAD_DIM = 96
CHUNK_SIZE = 64

WARMUP = 3
REPEATS = 10

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "test23_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Use FP32 first to make the reference comparison meaningful.
DTYPE = torch.float32


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def make_inputs():
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)

    shape = (BATCH, HEADS, SEQ_LEN, HEAD_DIM)

    q = torch.randn(
        shape, generator=generator,
        device=DEVICE, dtype=DTYPE
    ) * 0.1

    k = torch.nn.functional.normalize(
        torch.randn(
            shape, generator=generator,
            device=DEVICE, dtype=DTYPE
        ),
        dim=-1,
    )

    v = torch.randn(
        shape, generator=generator,
        device=DEVICE, dtype=DTYPE
    ) * 0.1

    # Keep decay away from extreme underflow for this test.
    alpha = torch.full(
        (BATCH, HEADS, SEQ_LEN),
        0.98, device=DEVICE, dtype=DTYPE
    )

    beta = torch.full(
        (BATCH, HEADS, SEQ_LEN),
        0.05, device=DEVICE, dtype=DTYPE
    )

    state = torch.zeros(
        BATCH, HEADS, HEAD_DIM, HEAD_DIM,
        device=DEVICE, dtype=DTYPE
    )

    return q, k, v, alpha, beta, state


def measure(fn):
    for _ in range(WARMUP):
        fn()
    synchronize()

    samples = []

    for _ in range(REPEATS):
        synchronize()
        start = time.perf_counter()
        fn()
        synchronize()
        samples.append(
            (time.perf_counter() - start) * 1000
        )

    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main():
    torch.manual_seed(SEED)

    print("=" * 78)
    print("TEST23 — KDA NUMERICAL AND PERFORMANCE BENCHMARK")
    print("=" * 78)
    print("Device:", DEVICE)
    print("PyTorch:", torch.__version__)
    print("Sequence length:", SEQ_LEN)
    print("Chunk size:", CHUNK_SIZE)
    print("Shape:", (BATCH, HEADS, SEQ_LEN, HEAD_DIM))

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(DEVICE))
        torch.cuda.reset_peak_memory_stats(DEVICE)

    q, k, v, alpha, beta, state = make_inputs()

    print("\n[1] Run recurrent reference...")
    ref_out, ref_state = kda_recurrence_script(
        q, k, v, alpha, beta, state.clone()
    )
    synchronize()

    print("[2] Run chunked recurrence...")
    chunk_out, chunk_state = kda_recurrence_chunked(
        q, k, v, alpha, beta, state.clone(),
        chunk_size=CHUNK_SIZE,
    )
    synchronize()

    out_abs = (ref_out - chunk_out).abs()
    state_abs = (ref_state - chunk_state).abs()

    output_max_abs = out_abs.max().item()
    state_max_abs = state_abs.max().item()

    output_mean_abs = out_abs.mean().item()
    state_mean_abs = state_abs.mean().item()

    print("\nNumerical comparison:")
    print(f"Output max abs error: {output_max_abs:.8e}")
    print(f"Output mean abs error: {output_mean_abs:.8e}")
    print(f"State max abs error:  {state_max_abs:.8e}")
    print(f"State mean abs error: {state_mean_abs:.8e}")

    print("\n[3] Benchmark recurrent reference...")
    ref_timing = measure(
        lambda: kda_recurrence_script(
            q, k, v, alpha, beta, state.clone()
        )
    )

    print("[4] Benchmark chunked recurrence...")
    chunk_timing = measure(
        lambda: kda_recurrence_chunked(
            q, k, v, alpha, beta, state.clone(),
            chunk_size=CHUNK_SIZE,
        )
    )

    speedup = (
        ref_timing["median_ms"]
        / chunk_timing["median_ms"]
        if chunk_timing["median_ms"] > 0
        else None
    )

    print("\nPerformance:")
    print(
        f"Reference median: "
        f"{ref_timing['median_ms']:.3f} ms"
    )
    print(
        f"Chunked median:   "
        f"{chunk_timing['median_ms']:.3f} ms"
    )
    print(f"Reference/chunked ratio: {speedup:.3f}x")

    peak_memory_mb = None
    if DEVICE.type == "cuda":
        synchronize()
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(DEVICE)
            / (1024 ** 2)
        )
        print(f"Peak allocated VRAM: {peak_memory_mb:.2f} MB")

    result = {
        "test": "TEST23",
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "shape": {
            "batch": BATCH,
            "heads": HEADS,
            "seq_len": SEQ_LEN,
            "head_dim": HEAD_DIM,
            "chunk_size": CHUNK_SIZE,
        },
        "numerical": {
            "output_max_abs_error": output_max_abs,
            "output_mean_abs_error": output_mean_abs,
            "state_max_abs_error": state_max_abs,
            "state_mean_abs_error": state_mean_abs,
        },
        "reference_timing": ref_timing,
        "chunked_timing": chunk_timing,
        "reference_over_chunked_ratio": speedup,
        "peak_memory_mb": peak_memory_mb,
    }

    result_path = OUT_DIR / "test23_results.json"
    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nJSON:", result_path)
    print("=" * 78)


if __name__ == "__main__":
    main()