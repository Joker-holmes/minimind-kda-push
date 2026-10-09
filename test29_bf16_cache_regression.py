import gc
import json
import traceback
from datetime import datetime
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
# TEST29 — BF16 CACHE REGRESSION (with FP32 CONTROL)
# ============================================================
#
# Two-layer design:
#
#   Layer 1 (BF16): verify that the cached path is semantically
#   equivalent to the full-sequence path under BF16 autocast.
#   Judged by argmax match rate + cosine similarity, because
#   BF16 cannot be expected to match bit-for-bit.
#
#   Layer 2 (FP32): disable autocast and verify that the two
#   paths match to a tight fp32 tolerance (~1e-4). This is the
#   real proof that the caching logic is correct. If FP32
#   matches but BF16 does not, the difference is purely
#   precision loss, not a bug.
#
# ============================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 20261009
SEQ_LEN = 96
PREFILL_LENGTHS = [1, 8, 32, 64]

# BF16 element-wise tolerance. BF16 has ~7-bit mantissa, so
# accumulated error of 1e-2..5e-2 over a 96-token sequence is
# expected, not a defect.
ATOL_BF16 = 5e-2
RTOL_BF16 = 5e-2

# BF16 secondary judgement.
ARGMIN_MATCH_MIN = 0.95
COSINE_MIN = 0.999

# FP32 element-wise tolerance. The two paths are numerically
# equivalent in fp32 up to floating point reassociation.
ATOL_FP32 = 1e-4
RTOL_FP32 = 1e-4

OUT_DIR = ROOT / "test29_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def get_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError(
        f"Cannot extract logits from output type: {type(output)}"
    )


def get_cache(output):
    if hasattr(output, "past_key_values"):
        cache = output.past_key_values
    elif isinstance(output, dict):
        cache = output.get("past_key_values", output.get("cache"))
    else:
        cache = None

    if cache is None:
        raise RuntimeError(
            "No cache returned. Check model.forward and use_cache=True."
        )
    return cache


def check_finite_recursive(obj, name="tensor"):
    checked = 0

    if torch.is_tensor(obj):
        if not torch.isfinite(obj).all().item():
            raise AssertionError(f"{name} contains NaN or Inf")
        return 1

    if isinstance(obj, dict):
        for k, v in obj.items():
            checked += check_finite_recursive(v, f"{name}.{k}")
        return checked

    if isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            checked += check_finite_recursive(v, f"{name}[{i}]")
        return checked

    if hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):
        checked += check_finite_recursive(obj.key_cache, f"{name}.key_cache")
        checked += check_finite_recursive(obj.value_cache, f"{name}.value_cache")
        return checked

    return checked


# ============================================================
# Autocast context helpers
# ============================================================

def bf16_ctx():
    return torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.bfloat16,
        enabled=(
            DEVICE.type == "cuda"
            and torch.cuda.is_bf16_supported()
        ),
    )


def fp32_ctx():
    """Disable autocast, force fp32 arithmetic."""
    return torch.autocast(device_type=DEVICE.type, enabled=False)


# ============================================================
# Core measurement routines
# ============================================================

def compute_reference_logits(model, input_ids, ctx):
    """Full-sequence forward, no cache."""
    with torch.inference_mode():
        with ctx():
            output = model(
                input_ids=input_ids,
                use_cache=False,
            )
        logits = get_logits(output).float()
    return logits


def compute_incremental_logits(model, input_ids, prefill_len, ctx):
    """
    Prefill + incremental decode with cache.

    Returns:
        combined_logits: [B, T, V] fp32
        last_cache
        cache_tensor_count
    """
    total_len = input_ids.shape[1]
    pieces = []
    cache_tensor_count = 0

    with torch.inference_mode():
        # ---------------------------------------------------
        # Prefill
        # ---------------------------------------------------

        with ctx():
            output = model(
                input_ids=input_ids[:, :prefill_len],
                use_cache=True,
            )

        logits = get_logits(output).float()
        past = get_cache(output)

        check_finite_recursive(logits, "prefill_logits")
        cache_tensor_count = check_finite_recursive(
            past, "prefill_cache"
        )
        pieces.append(logits)

        # ---------------------------------------------------
        # Decode one token at a time
        # ---------------------------------------------------

        for pos in range(prefill_len, total_len):

            with ctx():
                output = model(
                    input_ids=input_ids[:, pos:pos + 1],
                    past_key_values=past,
                    use_cache=True,
                )

            logits = get_logits(output).float()
            past = get_cache(output)

            if logits.shape[1] != 1:
                raise AssertionError(
                    "Incremental decode must return one position; "
                    f"got {tuple(logits.shape)}"
                )

            check_finite_recursive(logits, f"decode_logits_{pos}")
            cache_tensor_count = max(
                cache_tensor_count,
                check_finite_recursive(past, f"decode_cache_{pos}"),
            )
            pieces.append(logits)

    combined = torch.cat(pieces, dim=1)
    return combined, past, cache_tensor_count


def compare_logits(reference, candidate, atol, rtol):
    """Compute error metrics between two logits tensors."""
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"Shape mismatch: candidate={tuple(candidate.shape)}, "
            f"reference={tuple(reference.shape)}"
        )

    diff = (candidate - reference).abs()

    max_error = diff.max().item()
    mean_error = diff.mean().item()

    close = torch.allclose(
        candidate, reference, atol=atol, rtol=rtol,
    )

    ref_tokens = reference.argmax(dim=-1)
    cand_tokens = candidate.argmax(dim=-1)
    match_rate = (ref_tokens == cand_tokens).float().mean().item()

    cosine = torch.nn.functional.cosine_similarity(
        candidate.reshape(1, -1),
        reference.reshape(1, -1),
        dim=1,
        eps=1e-8,
    ).item()

    return {
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "allclose": close,
        "argmax_match_rate": match_rate,
        "logits_cosine_similarity": cosine,
    }


# ============================================================
# One variant
# ============================================================

def run_variant(name, use_kda, use_attn_res, tokenizer):
    print("\n" + "=" * 78)
    print(f"TEST29 — {name}")
    print("=" * 78)

    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    config = build_config(
        vocab_size=len(tokenizer),
        use_kda=use_kda,
        use_attn_res=use_attn_res,
        tokenizer=tokenizer,
    )

    model = None
    result = {
        "name": name,
        "status": "ERROR",
        "sequence_length": SEQ_LEN,
        "prefill_lengths": PREFILL_LENGTHS,
        "bf16_atol": ATOL_BF16,
        "bf16_rtol": RTOL_BF16,
        "fp32_atol": ATOL_FP32,
        "fp32_rtol": RTOL_FP32,
        "prefill_results": [],
        "error": None,
    }

    try:
        model = MiniMindForCausalLM(config).to(DEVICE).eval()

        input_ids = torch.randint(
            low=0,
            high=len(tokenizer),
            size=(1, SEQ_LEN),
            dtype=torch.long,
            device=DEVICE,
        )

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)

        # ==========================================================
        # Layer 1: BF16 reference
        # ==========================================================

        print("\n--- LAYER 1: BF16 (autocast enabled) ---")

        reference_bf16 = compute_reference_logits(
            model,
            input_ids,
            bf16_ctx,
        )
        check_finite_recursive(reference_bf16, "reference_bf16")

        # ==========================================================
        # Layer 2: FP32 reference (cast model to fp32)
        # ==========================================================

        model_fp32 = model.float()

        reference_fp32 = compute_reference_logits(
            model_fp32,
            input_ids,
            fp32_ctx,
        )
        check_finite_recursive(reference_fp32, "reference_fp32")

        # ==========================================================
        # Per-prefill-length comparisons
        # ==========================================================

        for prefill_len in PREFILL_LENGTHS:

            print(f"\nPrefill length: {prefill_len}")

            # ----------------------------------------------
            # BF16
            # ----------------------------------------------

            combined_bf16, _, cache_count = compute_incremental_logits(
                model,
                input_ids,
                prefill_len,
                bf16_ctx,
            )

            metrics_bf16 = compare_logits(
                reference_bf16,
                combined_bf16,
                ATOL_BF16,
                RTOL_BF16,
            )

            argmax_ok = (
                metrics_bf16["argmax_match_rate"] >= ARGMIN_MATCH_MIN
            )
            cosine_ok = (
                metrics_bf16["logits_cosine_similarity"] >= COSINE_MIN
            )

            bf16_pass = (
                metrics_bf16["allclose"]
                or (argmax_ok and cosine_ok)
            )

            print(
                f"  [BF16] max_err={metrics_bf16['max_abs_error']:.4e}  "
                f"mean_err={metrics_bf16['mean_abs_error']:.4e}  "
                f"argmax={metrics_bf16['argmax_match_rate']:.2%}  "
                f"cosine={metrics_bf16['logits_cosine_similarity']:.6f}  "
                f"allclose={metrics_bf16['allclose']}"
            )
            print(
                f"  [BF16] finite cache tensors checked: {cache_count}"
            )

            # ----------------------------------------------
            # FP32 control
            # ----------------------------------------------

            combined_fp32, _, _ = compute_incremental_logits(
                model_fp32,
                input_ids,
                prefill_len,
                fp32_ctx,
            )

            metrics_fp32 = compare_logits(
                reference_fp32,
                combined_fp32,
                ATOL_FP32,
                RTOL_FP32,
            )

            fp32_pass = metrics_fp32["allclose"]

            print(
                f"  [FP32] max_err={metrics_fp32['max_abs_error']:.4e}  "
                f"mean_err={metrics_fp32['mean_abs_error']:.4e}  "
                f"argmax={metrics_fp32['argmax_match_rate']:.2%}  "
                f"cosine={metrics_fp32['logits_cosine_similarity']:.6f}  "
                f"allclose={metrics_fp32['allclose']}"
            )

            # ----------------------------------------------
            # Combined status
            #
            # FP32 must pass: this is the real proof that the
            # cache logic is correct.
            #
            # BF16 may either pass strictly (allclose) or
            # semantically (argmax + cosine).
            # ----------------------------------------------

            status = "PASS" if (fp32_pass and bf16_pass) else "FAILED"

            row = {
                "prefill_length": prefill_len,

                # BF16
                "bf16_max_abs_error":
                    metrics_bf16["max_abs_error"],
                "bf16_mean_abs_error":
                    metrics_bf16["mean_abs_error"],
                "bf16_allclose":
                    metrics_bf16["allclose"],
                "bf16_argmax_match_rate":
                    metrics_bf16["argmax_match_rate"],
                "bf16_cosine_similarity":
                    metrics_bf16["logits_cosine_similarity"],
                "bf16_finite_cache_tensor_count":
                    cache_count,
                "bf16_pass":
                    bf16_pass,

                # FP32 control
                "fp32_max_abs_error":
                    metrics_fp32["max_abs_error"],
                "fp32_mean_abs_error":
                    metrics_fp32["mean_abs_error"],
                "fp32_allclose":
                    metrics_fp32["allclose"],
                "fp32_argmax_match_rate":
                    metrics_fp32["argmax_match_rate"],
                "fp32_cosine_similarity":
                    metrics_fp32["logits_cosine_similarity"],
                "fp32_pass":
                    fp32_pass,

                "status": status,
            }
            result["prefill_results"].append(row)

            print(f"  STATUS: {status}")

        if DEVICE.type == "cuda":
            sync()
            result["peak_memory_mb"] = round(
                torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2),
                2,
            )

        result["status"] = (
            "PASS"
            if all(
                r["status"] == "PASS"
                for r in result["prefill_results"]
            )
            else "FAILED"
        )

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()

    finally:
        if model is not None:
            del model
        del config
        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\nVariant final status:", result["status"])
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("TEST29 — BF16 CACHE REGRESSION (with FP32 CONTROL)")
    print("=" * 78)
    print("Timestamp:", datetime.now().isoformat(timespec="seconds"))
    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Prefill lengths: {PREFILL_LENGTHS}")
    print(f"BF16 tolerance: atol={ATOL_BF16}, rtol={RTOL_BF16}")
    print(
        f"BF16 secondary: argmax>={ARGMIN_MATCH_MIN}, "
        f"cosine>={COSINE_MIN}"
    )
    print(f"FP32 tolerance: atol={ATOL_FP32}, rtol={RTOL_FP32}")

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "This test is intended for CUDA BF16 autocast. "
            "Run it with your CUDA-enabled PyTorch environment."
        )

    print("GPU:", torch.cuda.get_device_name(DEVICE))

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "The current GPU/device does not report BF16 support."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        trust_remote_code=True,
    )
    print("Tokenizer vocabulary size:", len(tokenizer))

    experiments = [
        ("baseline_attention", False, False),
        ("kda_v3", True, False),
        ("kda_v3_attnres", True, True),
    ]

    results = []

    for name, use_kda, use_attn_res in experiments:
        results.append(
            run_variant(
                name=name,
                use_kda=use_kda,
                use_attn_res=use_attn_res,
                tokenizer=tokenizer,
            )
        )

    output_path = OUT_DIR / "test29_results.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("TEST29 FINAL SUMMARY")
    print("=" * 78)

    for result in results:
        print(f"{result['name']:22s} {result['status']}")

    print("Saved:", output_path)

    if all(r["status"] == "PASS" for r in results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — inspect errors and JSON metrics.")


if __name__ == "__main__":
    main()