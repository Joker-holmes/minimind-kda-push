
# -*- coding: utf-8 -*-
"""
TEST36 — KDA OPERATOR-LEVEL PROFILER

Profiles the existing TEST34 checkpoints without training.
Uses identical input shapes and BF16 autocast settings.

Outputs:
    test36_outputs/test36_profiler_summary.json
    test36_outputs/<variant>_trace.json

Chrome trace files can be opened in chrome://tracing or Perfetto.
"""

import sys
import json
import time
import traceback
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "test36_outputs"
CHECKPOINT_DIR = ROOT / "test34_outputs" / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 3601
SEQ_LEN = 128
BATCH_SIZE = 2
USE_BF16 = True

WARMUP_STEPS = 10
PROFILE_STEPS = 5
TOP_K = 20

VARIANTS = [
    ("baseline_attention", False, False),
    ("kda_v3", True, False),
    ("kda_v3_attnres", True, True),
]


def print_header(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def load_dependencies():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import test34_train_real_corpus as t34

    build_config, model_class = t34.load_project_components()
    tokenizer = t34.load_tokenizer()

    return t34, build_config, model_class, tokenizer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_autocast_dtype(device):
    if device.type == "cuda" and USE_BF16:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return None


def load_checkpoint_model(
    t34,
    build_config,
    model_class,
    tokenizer,
    name,
    use_kda,
    use_attn_res,
    device,
):
    path = CHECKPOINT_DIR / f"{name}.pt"

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {path}")

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
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()
    model.to(device)

    return model, path


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def run_forward(t34, model, input_ids, device, autocast_dtype):
    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=(autocast_dtype is not None),
    ):
        logits = t34.forward_logits(model, input_ids)

    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits detected during profiling.")

    return logits


def profile_variant(
    t34,
    build_config,
    model_class,
    tokenizer,
    name,
    use_kda,
    use_attn_res,
    input_ids,
    device,
    autocast_dtype,
):
    print_header(f"PROFILING — {name}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model, checkpoint_path = load_checkpoint_model(
        t34=t34,
        build_config=build_config,
        model_class=model_class,
        tokenizer=tokenizer,
        name=name,
        use_kda=use_kda,
        use_attn_res=use_attn_res,
        device=device,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Warm-up steps: {WARMUP_STEPS}")
    print(f"Profiled steps: {PROFILE_STEPS}")

    # Warm-up outside the profiler.
    for _ in range(WARMUP_STEPS):
        run_forward(
            t34, model, input_ids, device, autocast_dtype
        )

    synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_path = OUTPUT_DIR / f"{name}_trace.json"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(PROFILE_STEPS):
            with torch.profiler.record_function("TEST36_MODEL_FORWARD"):
                run_forward(
                    t34,
                    model,
                    input_ids,
                    device,
                    autocast_dtype,
                )
            synchronize(device)

    prof.export_chrome_trace(str(trace_path))

    events = prof.key_averages()

    # Sort separately by CPU self time and CUDA self time.
    cpu_sorted = sorted(
        events,
        key=lambda e: float(getattr(e, "self_cpu_time_total", 0.0)),
        reverse=True,
    )

    cuda_sorted = sorted(
        events,
        key=lambda e: float(
            getattr(e, "self_device_time_total",
                    getattr(e, "self_cuda_time_total", 0.0))
        ),
        reverse=True,
    )

    def get_cuda_time(event):
        # Newer PyTorch versions expose device time; older versions
        # may expose CUDA time.
        value = getattr(event, "self_device_time_total", None)
        if value is None:
            value = getattr(event, "self_cuda_time_total", 0.0)
        return float(value or 0.0)

    def event_row(event):
        return {
            "operator": event.key,
            "calls": int(event.count),
            "self_cpu_time_us": float(
                getattr(event, "self_cpu_time_total", 0.0)
            ),
            "cpu_time_total_us": float(
                getattr(event, "cpu_time_total", 0.0)
            ),
            "self_cuda_time_us": get_cuda_time(event),
            "cuda_time_total_us": float(
                getattr(event, "device_time_total",
                        getattr(event, "cuda_time_total", 0.0))
                or 0.0
            ),
            "cpu_memory_usage_bytes": int(
                getattr(event, "self_cpu_memory_usage", 0)
            ),
            "device_memory_usage_bytes": int(
                getattr(event, "self_device_memory_usage",
                        getattr(event, "self_cuda_memory_usage", 0))
                or 0
            ),
        }

    cpu_top = [event_row(e) for e in cpu_sorted[:TOP_K]]
    cuda_top = [event_row(e) for e in cuda_sorted[:TOP_K]]

    print("\nTop operators by CPU self time:")
    print(
        f"{'Operator':<42} {'Calls':>8} "
        f"{'CPU self ms':>13} {'CUDA self ms':>14}"
    )
    print("-" * 82)

    for row in cpu_top:
        print(
            f"{row['operator'][:41]:<42}"
            f"{row['calls']:>8}"
            f"{row['self_cpu_time_us'] / 1000:>13.3f}"
            f"{row['self_cuda_time_us'] / 1000:>14.3f}"
        )

    print("\nTop operators by CUDA/device self time:")
    print(
        f"{'Operator':<42} {'Calls':>8} "
        f"{'CPU self ms':>13} {'CUDA self ms':>14}"
    )
    print("-" * 82)

    for row in cuda_top:
        print(
            f"{row['operator'][:41]:<42}"
            f"{row['calls']:>8}"
            f"{row['self_cpu_time_us'] / 1000:>13.3f}"
            f"{row['self_cuda_time_us'] / 1000:>14.3f}"
        )

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

    result = {
        "variant": name,
        "checkpoint": str(checkpoint_path),
        "input_shape": list(input_ids.shape),
        "dtype": str(autocast_dtype),
        "warmup_steps": WARMUP_STEPS,
        "profile_steps": PROFILE_STEPS,
        "trace_path": str(trace_path),
        "peak_allocated_memory_mb": peak_memory_mb,
        "top_cpu_operators": cpu_top,
        "top_cuda_operators": cuda_top,
    }

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\nChrome trace saved: {trace_path}")
    return result


def main():
    print_header("TEST36 — KDA OPERATOR-LEVEL PROFILER")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {CHECKPOINT_DIR}"
        )

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = get_device()
    autocast_dtype = get_autocast_dtype(device)

    t34, build_config, model_class, tokenizer = load_dependencies()

    # Reuse the TEST34 split and preprocessing, so input tokens come
    # from the same validation set.
    _, val_blocks = t34.prepare_dataset(tokenizer)

    if len(val_blocks) < BATCH_SIZE:
        raise RuntimeError("Insufficient validation blocks.")

    input_ids = val_blocks[:BATCH_SIZE, :-1].to(device)

    report = {
        "test": "TEST36_OPERATOR_PROFILER",
        "pytorch": torch.__version__,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "autocast_dtype": str(autocast_dtype),
        "input_shape": list(input_ids.shape),
        "results": [],
    }

    for name, use_kda, use_attn_res in VARIANTS:
        result = profile_variant(
            t34=t34,
            build_config=build_config,
            model_class=model_class,
            tokenizer=tokenizer,
            name=name,
            use_kda=use_kda,
            use_attn_res=use_attn_res,
            input_ids=input_ids,
            device=device,
            autocast_dtype=autocast_dtype,
        )

        report["results"].append(result)

        partial_path = OUTPUT_DIR / "test36_summary_partial.json"
        with partial_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    summary_path = OUTPUT_DIR / "test36_profiler_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_header("TEST36 — SUMMARY")

    for result in report["results"]:
        print(f"\n{result['variant']}")
        print(
            f"  Peak allocated memory: "
            f"{result['peak_allocated_memory_mb']}"
        )
        print("  Top CUDA operators:")

        for row in result["top_cuda_operators"][:8]:
            print(
                f"    {row['operator']}: "
                f"self CUDA/device time="
                f"{row['self_cuda_time_us'] / 1000:.3f} ms, "
                f"calls={row['calls']}"
            )

    print(f"\nJSON summary: {summary_path}")
    print("No model weights were modified.")
    print("TEST36 completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 88)
        print("TEST36 FAILED")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("!" * 88)
        raise