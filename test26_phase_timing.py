import gc
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindForCausalLM
from test22_real_text_comparison import (
    build_config,
    ROOT,
    TOKENIZER_DIR,
)

# ============================================================
# TEST26 — Phase Timing Benchmark (CUDA Event version)
# ============================================================
#
# Changes compared to the perf_counter version:
#
# 1. Uses torch.cuda.Event for timing instead of time.perf_counter().
#    time.perf_counter() around async CUDA kernels measures kernel
#    launch latency, not execution time, which produces wild swings.
#
# 2. Records each phase (forward / backward / optimizer) as a
#    separate event pair, and only synchronizes once at the end of
#    the step. This gives accurate per-phase GPU time.
#
# 3. Reports median, min and robust_mean (drop slowest 20%) instead
#    of relying on the median alone. Throttling and OS background
#    work regularly double or triple individual steps.
#
# 4. Increases warmup to 15 and measured steps to 50 for stability.
#
# 5. Resets peak memory stats before each variant so the reported
#    peak actually corresponds to that variant.
#
# ============================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2
SEQ_LEN = 128
WARMUP_STEPS = 15
MEASURE_STEPS = 50
LEARNING_RATE = 3e-4

OUT_DIR = ROOT / "test26_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def autocast_context():
    enabled = (
        DEVICE.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    return torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.bfloat16,
        enabled=enabled,
    )


# ============================================================
# CUDA event helpers
# ============================================================

def _event():
    """Create a CUDA event, or a no-op placeholder on CPU."""
    if DEVICE.type == "cuda":
        return torch.cuda.Event(enable_timing=True)
    return None


def _elapsed_ms(start, end):
    """Elapsed milliseconds between two CUDA events."""
    if DEVICE.type == "cuda":
        return start.elapsed_time(end)
    return 0.0


# ============================================================
# Timed step (CUDA Event)
# ============================================================

def timed_step(model, optimizer, input_ids):
    """
    Run one train step and return per-phase GPU timings in ms.

    Timeline:
        e0 ---- forward ---- e1 ---- backward ---- e2 ---- step ---- e3
        e_start -------------------------------------------------- e_end

    We synchronize only once, at the very end of the step. All
    intermediate timings come from CUDA events, so they measure
    actual GPU work, not kernel launch latency.
    """

    model.train()

    # -----------------------------------------------------
    # Make sure the previous iteration is fully done before
    # we start the timeline.
    # -----------------------------------------------------

    sync()

    e0 = _event()
    e1 = _event()
    e2 = _event()
    e3 = _event()

    if DEVICE.type == "cuda":
        e0.record()

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------

    optimizer.zero_grad(set_to_none=True)

    with autocast_context():
        result = model(
            input_ids=input_ids,
            labels=input_ids,
            use_cache=False,
        )
        loss = result.loss

    if DEVICE.type == "cuda":
        e1.record()

    # -----------------------------------------------------
    # Backward
    # -----------------------------------------------------

    loss.backward()

    if DEVICE.type == "cuda":
        e2.record()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer.step()

    if DEVICE.type == "cuda":
        e3.record()

    # -----------------------------------------------------
    # Single synchronize at the end of the step
    # -----------------------------------------------------

    sync()

    forward_ms = _elapsed_ms(e0, e1) if DEVICE.type == "cuda" else 0.0
    backward_ms = _elapsed_ms(e1, e2) if DEVICE.type == "cuda" else 0.0
    optimizer_ms = _elapsed_ms(e2, e3) if DEVICE.type == "cuda" else 0.0
    step_ms = _elapsed_ms(e0, e3) if DEVICE.type == "cuda" else 0.0

    return {
        "loss": float(loss.detach().float().item()),
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "step_ms": step_ms,
    }


# ============================================================
# Robust statistics
# ============================================================

def summarize(values):
    """
    Return median / min / max / mean / robust_mean.

    robust_mean drops the slowest 20% of samples before averaging.
    Those slow samples usually come from thermal throttling or
    OS background work, not from the model itself.
    """
    values = sorted(values)
    n = len(values)

    keep_n = max(1, int(n * 0.8))
    kept = values[:keep_n]

    return {
        "mean": sum(values) / n,
        "median": values[n // 2],
        "min": values[0],
        "max": values[-1],
        "robust_mean": sum(kept) / len(kept),
        "n_samples": n,
        "n_kept": keep_n,
    }


# ============================================================
# Benchmark one variant
# ============================================================

def benchmark_variant(name, use_kda, use_attn_res, tokenizer):
    print("\n" + "=" * 72)
    print("TEST26:", name)
    print("=" * 72)

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)

    torch.manual_seed(20261009)

    config = build_config(
        vocab_size=len(tokenizer),
        use_kda=use_kda,
        use_attn_res=use_attn_res,
        tokenizer=tokenizer,
    )

    model = MiniMindForCausalLM(config).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    input_ids = torch.randint(
        low=0,
        high=len(tokenizer),
        size=(BATCH_SIZE, SEQ_LEN),
        device=DEVICE,
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # Warmup
    # -----------------------------------------------------

    print("Warmup steps:", WARMUP_STEPS)

    for i in range(WARMUP_STEPS):
        stats = timed_step(model, optimizer, input_ids)
        print(
            f"Warmup {i + 1}/{WARMUP_STEPS}: "
            f"loss={stats['loss']:.4f}, "
            f"forward={stats['forward_ms']:.2f} ms, "
            f"backward={stats['backward_ms']:.2f} ms, "
            f"optimizer={stats['optimizer_ms']:.2f} ms, "
            f"step={stats['step_ms']:.2f} ms"
        )

    # -----------------------------------------------------
    # Measured loop
    # -----------------------------------------------------

    measurements = []

    print("\nMeasured steps:")

    for i in range(MEASURE_STEPS):
        stats = timed_step(model, optimizer, input_ids)
        measurements.append(stats)

        print(
            f"{i + 1:02d}/{MEASURE_STEPS}: "
            f"forward={stats['forward_ms']:7.2f} ms | "
            f"backward={stats['backward_ms']:7.2f} ms | "
            f"optimizer={stats['optimizer_ms']:6.2f} ms | "
            f"step={stats['step_ms']:7.2f} ms"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    keys = [
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "step_ms",
        "loss",
    ]

    summary = {}
    for key in keys:
        values = [item[key] for item in measurements]
        summary[key] = summarize(values)

    # -----------------------------------------------------
    # Memory
    # -----------------------------------------------------

    if DEVICE.type == "cuda":
        allocated_mb = torch.cuda.memory_allocated(DEVICE) / 1024**2
        peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    else:
        allocated_mb = None
        peak_mb = None

    result = {
        "name": name,
        "use_kda": use_kda,
        "use_attn_res": use_attn_res,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQ_LEN,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURE_STEPS,
        "summary": summary,
        "allocated_memory_mb": allocated_mb,
        "peak_memory_mb": peak_mb,
        "raw_measurements": measurements,
    }

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\nSummary (median / robust_mean / min):")

    for key in keys:
        s = summary[key]
        print(
            f"  {key:14s} "
            f"median={s['median']:9.3f}  "
            f"robust_mean={s['robust_mean']:9.3f}  "
            f"min={s['min']:9.3f}  "
            f"max={s['max']:9.3f}"
        )

    print(
        f"\nMemory: allocated={allocated_mb:.1f} MB, "
        f"peak={peak_mb:.1f} MB"
        if allocated_mb is not None
        else "\nMemory: N/A (CPU)"
    )

    del model, optimizer, input_ids, config
    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("TEST26 — PHASE TIMING BENCHMARK (CUDA EVENT)")
    print("=" * 72)
    print("Device:", DEVICE)
    print("PyTorch:", torch.__version__)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(DEVICE))
        print("BF16 supported:", torch.cuda.is_bf16_supported())

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        trust_remote_code=True,
    )

    experiments = [
        ("baseline_attention", False, False),
        ("kda_v3", True, False),
        ("kda_v3_attnres", True, True),
    ]

    results = []

    for name, use_kda, use_attn_res in experiments:
        results.append(
            benchmark_variant(
                name,
                use_kda,
                use_attn_res,
                tokenizer,
            )
        )

    output_path = OUT_DIR / "test26_results.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Final comparison table
    # -----------------------------------------------------

    print("\n" + "=" * 72)
    print("FINAL COMPARISON — ROBUST MEAN STEP TIME")
    print("=" * 72)

    print(
        f"{'Experiment':<22}"
        f"{'forward':>12}"
        f"{'backward':>12}"
        f"{'optimizer':>12}"
        f"{'step':>12}"
        f"{'ratio':>10}"
    )
    print("-" * 80)

    baseline_step = None

    for result in results:
        s = result["summary"]

        fwd = s["forward_ms"]["robust_mean"]
        bwd = s["backward_ms"]["robust_mean"]
        opt = s["optimizer_ms"]["robust_mean"]
        stp = s["step_ms"]["robust_mean"]

        if baseline_step is None:
            baseline_step = stp
            ratio_str = "1.00x"
        else:
            ratio_str = f"{stp / baseline_step:.2f}x"

        print(
            f"{result['name']:<22}"
            f"{fwd:>10.2f}ms"
            f"{bwd:>10.2f}ms"
            f"{opt:>10.2f}ms"
            f"{stp:>10.2f}ms"
            f"{ratio_str:>10}"
        )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()