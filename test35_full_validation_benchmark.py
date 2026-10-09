
# -*- coding: utf-8 -*-
"""
TEST35 — FULL VALIDATION + INFERENCE BENCHMARK

Models:
    baseline_attention
    kda_v3
    kda_v3_attnres

Features:
    1. Reconstruct the exact TEST34 record split.
    2. Load existing checkpoints; do not train.
    3. Evaluate every validation block.
    4. Calculate token-weighted loss, PPL and next-token accuracy.
    5. Benchmark inference latency and throughput after warm-up.
    6. Save a JSON report.

Run with the minimind-kda Python environment.
"""

import sys
import json
import time
import math
import traceback
from pathlib import Path
from statistics import median

import numpy as np
import torch


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "test35_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = ROOT / "test34_outputs" / "checkpoints"

SEED = 3401

# Must match TEST34.
MAX_RECORDS = 30000
TRAIN_RATIO = 0.90
SEQ_LEN = 128
BATCH_SIZE = 2
USE_BF16 = True

# Inference benchmark settings.
BENCH_WARMUP_STEPS = 15
BENCH_MEASURE_STEPS = 60
BENCH_ROUNDS = 3

VARIANTS = [
    ("baseline_attention", False, False),
    ("kda_v3", True, False),
    ("kda_v3_attnres", True, True),
]


# ============================================================
# 2. ENVIRONMENT
# ============================================================

def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_header(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_autocast_dtype(device):
    if device.type == "cuda" and USE_BF16:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return None


# ============================================================
# 3. REUSE TEST34 DATA AND MODEL IMPLEMENTATION
# ============================================================

def load_dependencies():
    """
    Reuse TEST34 helpers to preserve the same dataset selection,
    record-level split, tokenization, model configuration and loss.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        import test34_train_real_corpus as t34
    except Exception as exc:
        raise RuntimeError(
            "Cannot import test34_train_real_corpus.py. "
            "Keep TEST35 in the project root and retain TEST34."
        ) from exc

    build_config, model_class = t34.load_project_components()
    tokenizer = t34.load_tokenizer()

    return t34, build_config, model_class, tokenizer


def prepare_validation_blocks(t34, tokenizer):
    """
    Rebuild the TEST34 split using the same seed and preprocessing.
    The selected validation records should match TEST34 as long as
    the TEST34 script and dataset have not changed.
    """
    train_blocks, val_blocks = t34.prepare_dataset(tokenizer)

    print(f"\nTraining blocks reconstructed:   {len(train_blocks):,}")
    print(f"Validation blocks reconstructed: {len(val_blocks):,}")

    return val_blocks


# ============================================================
# 4. MODEL AND CHECKPOINT
# ============================================================

def load_model(
    t34,
    build_config,
    model_class,
    tokenizer,
    variant_name,
    use_kda,
    use_attn_res,
    device,
):
    checkpoint_path = CHECKPOINT_DIR / f"{variant_name}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = t34.build_model(
        build_config=build_config,
        model_class=model_class,
        tokenizer=tokenizer,
        use_kda=use_kda,
        use_attn_res=use_attn_res,
        device=device,
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility fallback for older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unexpected checkpoint format: {type(checkpoint)}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"'model_state_dict' missing from {checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()
    model.to(device)

    parameters = sum(p.numel() for p in model.parameters())

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parameters: {parameters:,}")
    print(f"Loaded strictly: True")

    return model, parameters, checkpoint_path


# ============================================================
# 5. FULL VALIDATION
# ============================================================

@torch.inference_mode()
def evaluate_full_validation(
    t34,
    model,
    val_blocks,
    device,
    autocast_dtype,
):
    """
    Evaluate every validation block.

    Each stored block has SEQ_LEN + 1 tokens:
        input  = block[:, :-1]
        target = block[:, 1:]

    Aggregate loss by token count instead of averaging batch means.
    """
    model.eval()

    total_loss_sum = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    start_time = time.perf_counter()

    for start in range(0, len(val_blocks), BATCH_SIZE):
        batch_blocks = val_blocks[start:start + BATCH_SIZE]

        block = batch_blocks.to(
            device,
            non_blocking=True,
        )

        input_ids = block[:, :-1]
        targets = block[:, 1:]

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=(autocast_dtype is not None),
        ):
            logits = t34.forward_logits(model, input_ids)

        loss = t34.causal_lm_loss(logits, block)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss in batch {num_batches}"
            )

        token_count = targets.numel()

        total_loss_sum += float(loss.item()) * token_count

        predictions = logits.argmax(dim=-1)
        total_correct += int(
            (predictions == targets).sum().item()
        )

        total_tokens += token_count
        num_batches += 1

        if num_batches % 300 == 0:
            print(
                f"  Evaluated {num_batches:,} batches; "
                f"{total_tokens:,} target tokens"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    if total_tokens == 0:
        raise RuntimeError("Validation set contains no target tokens.")

    mean_loss = total_loss_sum / total_tokens

    return {
        "validation_blocks": len(val_blocks),
        "validation_batches": num_batches,
        "target_tokens": total_tokens,
        "token_weighted_loss": mean_loss,
        "ppl": math.exp(min(mean_loss, 20.0)),
        "next_token_accuracy": total_correct / total_tokens,
        "evaluation_seconds": elapsed,
        "evaluation_target_tokens_per_second": (
            total_tokens / elapsed if elapsed > 0 else None
        ),
    }


# ============================================================
# 6. INFERENCE LATENCY / THROUGHPUT
# ============================================================

@torch.inference_mode()
def benchmark_inference(
    t34,
    model,
    val_blocks,
    device,
    autocast_dtype,
):
    """
    Benchmark repeated full-sequence forward passes.

    This is prefill/full-sequence forward throughput, not cached
    one-token autoregressive generation throughput.
    """
    model.eval()

    fixed_block = val_blocks[0:BATCH_SIZE].to(
        device,
        non_blocking=True,
    )

    input_ids = fixed_block[:, :-1]

    def run_one_forward():
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=(autocast_dtype is not None),
        ):
            logits = t34.forward_logits(model, input_ids)

        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits during benchmark.")

        return logits

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up: excluded from timing.
    for _ in range(BENCH_WARMUP_STEPS):
        run_one_forward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    round_results = []

    for round_index in range(BENCH_ROUNDS):
        times = []

        for _ in range(BENCH_MEASURE_STEPS):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            run_one_forward()

            if device.type == "cuda":
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

        round_median = median(times)
        round_mean = sum(times) / len(times)

        round_results.append({
            "round": round_index + 1,
            "median_latency_ms": round_median * 1000,
            "mean_latency_ms": round_mean * 1000,
            "min_latency_ms": min(times) * 1000,
            "max_latency_ms": max(times) * 1000,
            "throughput_tokens_per_second": (
                BATCH_SIZE * SEQ_LEN / round_mean
                if round_mean > 0
                else None
            ),
        })

        print(
            f"  Benchmark round {round_index + 1}/{BENCH_ROUNDS}: "
            f"median={round_median * 1000:.2f} ms, "
            f"mean={round_mean * 1000:.2f} ms, "
            f"throughput="
            f"{BATCH_SIZE * SEQ_LEN / round_mean:.1f} tokens/s"
        )

    medians = [
        item["median_latency_ms"] for item in round_results
    ]
    means = [
        item["mean_latency_ms"] for item in round_results
    ]
    throughputs = [
        item["throughput_tokens_per_second"]
        for item in round_results
    ]

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

    return {
        "benchmark_type": "full_sequence_forward_no_cache",
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQ_LEN,
        "warmup_steps": BENCH_WARMUP_STEPS,
        "measured_steps_per_round": BENCH_MEASURE_STEPS,
        "rounds": BENCH_ROUNDS,
        "median_of_round_medians_ms": median(medians),
        "mean_of_round_means_ms": sum(means) / len(means),
        "median_round_throughput_tokens_per_second": median(throughputs),
        "peak_allocated_memory_mb": peak_memory_mb,
        "round_results": round_results,
    }


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print_header("TEST35 — FULL VALIDATION + INFERENCE BENCHMARK")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {CHECKPOINT_DIR}"
        )

    set_seed(SEED)

    device = get_device()
    autocast_dtype = get_autocast_dtype(device)

    print(f"Device: {device}")
    print(f"Autocast dtype: {autocast_dtype}")

    t34, build_config, model_class, tokenizer = load_dependencies()

    val_blocks = prepare_validation_blocks(t34, tokenizer)

    all_results = {
        "test": "TEST35_FULL_VALIDATION_BENCHMARK",
        "seed": SEED,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "pytorch": torch.__version__,
        "autocast_dtype": str(autocast_dtype),
        "checkpoint_directory": str(CHECKPOINT_DIR),
        "validation_blocks": len(val_blocks),
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "results": [],
    }

    for variant_name, use_kda, use_attn_res in VARIANTS:
        print_header(f"TEST35 — {variant_name}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        model, parameters, checkpoint_path = load_model(
            t34=t34,
            build_config=build_config,
            model_class=model_class,
            tokenizer=tokenizer,
            variant_name=variant_name,
            use_kda=use_kda,
            use_attn_res=use_attn_res,
            device=device,
        )

        print("\n1. Full validation...")
        validation = evaluate_full_validation(
            t34=t34,
            model=model,
            val_blocks=val_blocks,
            device=device,
            autocast_dtype=autocast_dtype,
        )

        print(
            f"  Loss: {validation['token_weighted_loss']:.6f}"
        )
        print(f"  PPL: {validation['ppl']:.3f}")
        print(
            f"  Next-token accuracy: "
            f"{validation['next_token_accuracy'] * 100:.3f}%"
        )
        print(
            f"  Evaluation time: "
            f"{validation['evaluation_seconds']:.2f} s"
        )

        print("\n2. Inference benchmark...")
        benchmark = benchmark_inference(
            t34=t34,
            model=model,
            val_blocks=val_blocks,
            device=device,
            autocast_dtype=autocast_dtype,
        )

        result = {
            "variant": variant_name,
            "use_kda": use_kda,
            "use_attn_res": use_attn_res,
            "parameters": parameters,
            "checkpoint": str(checkpoint_path),
            "validation": validation,
            "benchmark": benchmark,
        }

        all_results["results"].append(result)

        # Save partial results after each model, so completed
        # evaluations are preserved if a later model fails.
        partial_path = OUTPUT_DIR / "test35_results_partial.json"

        with partial_path.open("w", encoding="utf-8") as f:
            json.dump(
                all_results,
                f,
                ensure_ascii=False,
                indent=2,
            )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_path = OUTPUT_DIR / "test35_results.json"

    with final_path.open("w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_header("TEST35 — FINAL SUMMARY")

    print(
        f"{'Variant':<24}"
        f"{'Val Loss':>12}"
        f"{'PPL':>12}"
        f"{'Accuracy %':>12}"
        f"{'Latency ms':>14}"
        f"{'Throughput':>14}"
        f"{'VRAM MB':>12}"
    )

    print("-" * 100)

    for result in all_results["results"]:
        val = result["validation"]
        bench = result["benchmark"]

        print(
            f"{result['variant']:<24}"
            f"{val['token_weighted_loss']:>12.5f}"
            f"{val['ppl']:>12.2f}"
            f"{val['next_token_accuracy'] * 100:>12.3f}"
            f"{bench['median_of_round_medians_ms']:>14.2f}"
            f"{bench['median_round_throughput_tokens_per_second']:>14.1f}"
            f"{bench['peak_allocated_memory_mb']:>12.1f}"
        )

    print("-" * 100)
    print(f"JSON report: {final_path}")
    print("No training was performed.")
    print("TEST35 completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 88)
        print("TEST35 FAILED")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("!" * 88)
        raise