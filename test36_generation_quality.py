# -*- coding: utf-8 -*-
"""
TEST36 — GENERATION QUALITY REGRESSION

Loads the checkpoints produced by TEST34 and compares:

1. Greedy decoding (deterministic)
2. Sampled decoding (temperature=0.85, top_p=0.85, top_k=50)

Metrics per variant:
- generation throughput (tokens/s)
- 2/3/4-gram repetition rate
- distinct token ratio
- EOS occurrence rate
- mean log-probability of generated tokens
- mean entropy of next-token distribution

All generated texts are written to test36_outputs/<variant>/<mode>.txt
for manual inspection.

Run:
    python test36_generation_quality.py
"""

import gc
import json
import math
import random
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.model_minimind import MiniMindForCausalLM
from test22_real_text_comparison import (
    build_config,
    ROOT,
    TOKENIZER_DIR,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 20261009

CHECKPOINT_DIR = ROOT / "test34_outputs" / "checkpoints"

OUT_DIR = ROOT / "test36_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Prompts
#
# Keep them short, in Chinese, matching the training corpus.
# The exact content matters less than having a consistent
# starting point across variants.
# ------------------------------------------------------------

PROMPTS = [
    "人工智能是",
    "在这个世界上，",
    "他走进房间，",
    "中国的历史可以追溯到",
    "科学技术的进步",
    "她轻轻地",
    "这本书讲述了",
    "在遥远的未来，",
]

MAX_NEW_TOKENS = 96
BATCH_SIZE = 1

# Sampling parameters (shared across variants)
TEMPERATURE = 0.85
TOP_P = 0.85
TOP_K = 50

# Variants to test (must match TEST34 checkpoints)
VARIANTS = [
    ("baseline_attention", False, False),
    ("kda_v3", True, False),
    ("kda_v3_attnres", True, True),
]


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def ngram_repetition_rate(token_ids, n):
    """
    Fraction of n-grams that appear more than once.

    Higher means more repetitive.
    """
    if len(token_ids) < n:
        return 0.0

    ngrams = [
        tuple(token_ids[i:i + n])
        for i in range(len(token_ids) - n + 1)
    ]

    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)

    return repeated / len(counts) if counts else 0.0


def distinct_token_ratio(token_ids):
    """Unique tokens / total tokens."""
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / len(token_ids)


def mean_logprob_of_generated(model, prompt_ids, generated_ids):
    """
    Teacher-forced log-prob of the generated continuation given the prompt.

    Returns mean log-prob per generated token and mean entropy of the
    next-token distribution.
    """
    full = torch.tensor(
        [prompt_ids + generated_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.bfloat16,
            enabled=(DEVICE.type == "cuda"),
        ):
            out = model(input_ids=full, use_cache=False)

        logits = out.logits.float()

    # logits[:, i, :] predicts token i+1
    prompt_len = len(prompt_ids)
    gen_len = len(generated_ids)

    if gen_len == 0:
        return 0.0, 0.0

    # slice: positions [prompt_len-1 .. prompt_len+gen_len-2]
    # predict tokens [prompt_len .. prompt_len+gen_len-1]
    start = prompt_len - 1
    end = start + gen_len

    gen_logits = logits[0, start:end, :]          # [gen_len, V]
    gen_targets = torch.tensor(
        generated_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    log_probs = F.log_softmax(gen_logits, dim=-1)
    token_logp = log_probs.gather(
        1, gen_targets.unsqueeze(1)
    ).squeeze(1)

    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)

    return float(token_logp.mean().item()), float(entropy.mean().item())


# ============================================================
# Generation
# ============================================================

def generate_greedy(model, prompt_ids, max_new_tokens):
    """Greedy decoding."""
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    generated = []

    sync()
    start = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.bfloat16,
            enabled=(DEVICE.type == "cuda"),
        ):
            for _ in range(max_new_tokens):
                out = model(input_ids=input_ids, use_cache=False)
                next_logits = out.logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)

                if next_token.item() == model.config.eos_token_id:
                    break

                generated.append(int(next_token.item()))
                input_ids = torch.cat(
                    [input_ids, next_token], dim=-1,
                )

    sync()
    elapsed = time.perf_counter() - start

    return generated, elapsed


def generate_sampled(model, prompt_ids, max_new_tokens,
                     temperature, top_k, top_p, seed):
    """Sampled decoding with top-k + top-p."""
    # Use a local generator for reproducibility
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    generated = []

    sync()
    start = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.bfloat16,
            enabled=(DEVICE.type == "cuda"),
        ):
            for _ in range(max_new_tokens):
                out = model(input_ids=input_ids, use_cache=False)
                logits = out.logits[:, -1, :].float()

                logits = logits / max(temperature, 1e-5)

                # top-k
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.shape[-1])
                    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
                    logits = torch.where(
                        logits < kth,
                        torch.full_like(logits, float("-inf")),
                        logits,
                    )

                # top-p
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True, dim=-1
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)

                    mask = cumprobs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False

                    remove = torch.zeros_like(mask)
                    remove.scatter_(1, sorted_indices, mask)

                    logits = logits.masked_fill(remove, float("-inf"))

                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                next_token = torch.multinomial(
                    probs, num_samples=1, generator=generator,
                )

                if next_token.item() == model.config.eos_token_id:
                    break

                generated.append(int(next_token.item()))
                input_ids = torch.cat(
                    [input_ids, next_token], dim=-1,
                )

    sync()
    elapsed = time.perf_counter() - start

    return generated, elapsed


# ============================================================
# One variant
# ============================================================

def run_variant(name, use_kda, use_attn_res, tokenizer):
    print("\n" + "=" * 78)
    print(f"TEST36 — {name}")
    print("=" * 78)

    checkpoint_path = CHECKPOINT_DIR / f"{name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run TEST34 first."
        )

    set_seed(SEED)

    config = build_config(
        vocab_size=len(tokenizer),
        use_kda=use_kda,
        use_attn_res=use_attn_res,
        tokenizer=tokenizer,
    )

    model = MiniMindForCausalLM(config).to(DEVICE).eval()

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    missing, unexpected = model.load_state_dict(
        state["model_state_dict"],
        strict=False,
    )

    print(f"Loaded: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}, unexpected: {len(unexpected)}")

    run_dir = OUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "name": name,
        "use_kda": use_kda,
        "use_attn_res": use_attn_res,
        "greedy": [],
        "sampled": [],
    }

    # ========================================================
    # Greedy
    # ========================================================

    print("\n--- GREEDY ---")

    greedy_lines = []
    greedy_metrics = []

    for i, prompt in enumerate(PROMPTS):
        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False,
        )

        generated, elapsed = generate_greedy(
            model, prompt_ids, MAX_NEW_TOKENS,
        )

        text = tokenizer.decode(generated, skip_special_tokens=True)

        greedy_lines.append(
            f"[{i}] PROMPT: {prompt}\n"
            f"    OUTPUT: {text}\n"
        )

        metrics = {
            "prompt": prompt,
            "num_generated": len(generated),
            "elapsed_sec": elapsed,
            "tokens_per_sec": (
                len(generated) / elapsed if elapsed > 0 else 0.0
            ),
            "hit_eos": len(generated) < MAX_NEW_TOKENS,
            "rep_2gram": ngram_repetition_rate(generated, 2),
            "rep_3gram": ngram_repetition_rate(generated, 3),
            "rep_4gram": ngram_repetition_rate(generated, 4),
            "distinct_ratio": distinct_token_ratio(generated),
        }

        mlp, ent = mean_logprob_of_generated(
            model, prompt_ids, generated,
        )
        metrics["mean_logprob"] = mlp
        metrics["mean_entropy"] = ent

        greedy_metrics.append(metrics)

        print(
            f"  prompt={prompt!r} "
            f"len={len(generated):3d} "
            f"eos={metrics['hit_eos']} "
            f"rep2={metrics['rep_2gram']:.2f} "
            f"rep3={metrics['rep_3gram']:.2f} "
            f"dist={metrics['distinct_ratio']:.2f} "
            f"logp={mlp:.2f} "
            f"H={ent:.2f}"
        )

    (run_dir / "greedy.txt").write_text(
        "\n".join(greedy_lines),
        encoding="utf-8",
    )

    # Summary
    def agg(metrics, key):
        return statistics.mean(m[key] for m in metrics)

    results["greedy"] = {
        "per_prompt": greedy_metrics,
        "summary": {
            "mean_length": agg(greedy_metrics, "num_generated"),
            "mean_tokens_per_sec":
                agg(greedy_metrics, "tokens_per_sec"),
            "eos_rate":
                sum(m["hit_eos"] for m in greedy_metrics)
                / len(greedy_metrics),
            "mean_rep_2gram":
                agg(greedy_metrics, "rep_2gram"),
            "mean_rep_3gram":
                agg(greedy_metrics, "rep_3gram"),
            "mean_rep_4gram":
                agg(greedy_metrics, "rep_4gram"),
            "mean_distinct_ratio":
                agg(greedy_metrics, "distinct_ratio"),
            "mean_logprob":
                agg(greedy_metrics, "mean_logprob"),
            "mean_entropy":
                agg(greedy_metrics, "mean_entropy"),
        },
    }

    # ========================================================
    # Sampled
    # ========================================================

    print("\n--- SAMPLED (T=0.85, top_p=0.85, top_k=50) ---")

    sampled_lines = []
    sampled_metrics = []

    for i, prompt in enumerate(PROMPTS):
        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False,
        )

        generated, elapsed = generate_sampled(
            model,
            prompt_ids,
            MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            seed=SEED + i,
        )

        text = tokenizer.decode(generated, skip_special_tokens=True)

        sampled_lines.append(
            f"[{i}] PROMPT: {prompt}\n"
            f"    OUTPUT: {text}\n"
        )

        metrics = {
            "prompt": prompt,
            "num_generated": len(generated),
            "elapsed_sec": elapsed,
            "tokens_per_sec": (
                len(generated) / elapsed if elapsed > 0 else 0.0
            ),
            "hit_eos": len(generated) < MAX_NEW_TOKENS,
            "rep_2gram": ngram_repetition_rate(generated, 2),
            "rep_3gram": ngram_repetition_rate(generated, 3),
            "rep_4gram": ngram_repetition_rate(generated, 4),
            "distinct_ratio": distinct_token_ratio(generated),
        }

        mlp, ent = mean_logprob_of_generated(
            model, prompt_ids, generated,
        )
        metrics["mean_logprob"] = mlp
        metrics["mean_entropy"] = ent

        sampled_metrics.append(metrics)

        print(
            f"  prompt={prompt!r} "
            f"len={len(generated):3d} "
            f"eos={metrics['hit_eos']} "
            f"rep2={metrics['rep_2gram']:.2f} "
            f"rep3={metrics['rep_3gram']:.2f} "
            f"dist={metrics['distinct_ratio']:.2f} "
            f"logp={mlp:.2f} "
            f"H={ent:.2f}"
        )

    (run_dir / "sampled.txt").write_text(
        "\n".join(sampled_lines),
        encoding="utf-8",
    )

    results["sampled"] = {
        "per_prompt": sampled_metrics,
        "summary": {
            "mean_length": agg(sampled_metrics, "num_generated"),
            "mean_tokens_per_sec":
                agg(sampled_metrics, "tokens_per_sec"),
            "eos_rate":
                sum(m["hit_eos"] for m in sampled_metrics)
                / len(sampled_metrics),
            "mean_rep_2gram":
                agg(sampled_metrics, "rep_2gram"),
            "mean_rep_3gram":
                agg(sampled_metrics, "rep_3gram"),
            "mean_rep_4gram":
                agg(sampled_metrics, "rep_4gram"),
            "mean_distinct_ratio":
                agg(sampled_metrics, "distinct_ratio"),
            "mean_logprob":
                agg(sampled_metrics, "mean_logprob"),
            "mean_entropy":
                agg(sampled_metrics, "mean_entropy"),
        },
    }

    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("TEST36 — GENERATION QUALITY REGRESSION")
    print("=" * 78)
    print("Timestamp:", datetime.now().isoformat(timespec="seconds"))
    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(DEVICE))
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Sampling: T={TEMPERATURE} top_p={TOP_P} top_k={TOP_K}")
    print(f"Prompts: {len(PROMPTS)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        trust_remote_code=True,
    )
    print(f"Tokenizer vocab: {len(tokenizer)}")

    all_results = []

    for name, use_kda, use_attn_res in VARIANTS:
        try:
            result = run_variant(
                name, use_kda, use_attn_res, tokenizer,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"\n!!! {name} FAILED: {exc}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "name": name,
                "error": str(exc),
            })

    output_path = OUT_DIR / "test36_results.json"
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ========================================================
    # Final comparison table
    # ========================================================

    print("\n" + "=" * 78)
    print("TEST36 FINAL SUMMARY — GREEDY")
    print("=" * 78)

    print(
        f"{'Variant':<22}"
        f"{'Len':>6}"
        f"{'tok/s':>9}"
        f"{'EOS%':>7}"
        f"{'rep2':>7}"
        f"{'rep3':>7}"
        f"{'rep4':>7}"
        f"{'dist%':>7}"
        f"{'logp':>8}"
        f"{'H':>7}"
    )
    print("-" * 96)

    for r in all_results:
        if "greedy" not in r:
            continue
        s = r["greedy"]["summary"]
        print(
            f"{r['name']:<22}"
            f"{s['mean_length']:>6.1f}"
            f"{s['mean_tokens_per_sec']:>9.1f}"
            f"{s['eos_rate'] * 100:>6.1f}%"
            f"{s['mean_rep_2gram']:>7.3f}"
            f"{s['mean_rep_3gram']:>7.3f}"
            f"{s['mean_rep_4gram']:>7.3f}"
            f"{s['mean_distinct_ratio'] * 100:>6.1f}%"
            f"{s['mean_logprob']:>8.2f}"
            f"{s['mean_entropy']:>7.3f}"
        )

    print("\n" + "=" * 78)
    print("TEST36 FINAL SUMMARY — SAMPLED")
    print("=" * 78)

    print(
        f"{'Variant':<22}"
        f"{'Len':>6}"
        f"{'tok/s':>9}"
        f"{'EOS%':>7}"
        f"{'rep2':>7}"
        f"{'rep3':>7}"
        f"{'rep4':>7}"
        f"{'dist%':>7}"
        f"{'logp':>8}"
        f"{'H':>7}"
    )
    print("-" * 96)

    for r in all_results:
        if "sampled" not in r:
            continue
        s = r["sampled"]["summary"]
        print(
            f"{r['name']:<22}"
            f"{s['mean_length']:>6.1f}"
            f"{s['mean_tokens_per_sec']:>9.1f}"
            f"{s['eos_rate'] * 100:>6.1f}%"
            f"{s['mean_rep_2gram']:>7.3f}"
            f"{s['mean_rep_3gram']:>7.3f}"
            f"{s['mean_rep_4gram']:>7.3f}"
            f"{s['mean_distinct_ratio'] * 100:>6.1f}%"
            f"{s['mean_logprob']:>8.2f}"
            f"{s['mean_entropy']:>7.3f}"
        )

    print()
    print("Per-prompt texts:")
    for name, _, _ in VARIANTS:
        print(f"  greedy  : {OUT_DIR / name / 'greedy.txt'}")
        print(f"  sampled : {OUT_DIR / name / 'sampled.txt'}")

    print()
    print("JSON:", output_path)
    print("TEST36 completed.")


if __name__ == "__main__":
    main()