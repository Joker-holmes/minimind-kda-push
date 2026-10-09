# -*- coding: utf-8 -*-
"""
TEST37 — SFT training on MiniMind-3 sft_t2t format

Trains MiniMind on the multi-turn conversation schema used by
sft_t2t_mini.jsonl / sft_t2t.jsonl.

Data format (one record per line):

    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }

Optional fields:
    - assistant turn may carry "tool_calls": "[...]"
    - system turn may carry "tools": "[...]"
    - role="tool" turns carry the tool return value in "content"

Key properties:
- Manual ChatML-style template (MiniMind tokenizer has no chat_template).
- Loss masking: only assistant turns contribute to the loss.
- Tool calls are serialized into the assistant content.
- PAD tokens are ignored in the loss.
- Gradient accumulation + periodic validation.
- Non-finite gradient steps are skipped.
- A long run of consecutive non-finite steps aborts early, because
  once the parameters themselves enter a permanently bad region
  the skip loop cannot recover.
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import json
import math
import random
import statistics
import time
import traceback
from collections import Counter
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
# 1. CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 20261009

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATA_DIR = ROOT / "dataset"

SFT_CANDIDATES = [
    DATA_DIR / "sft_t2t_mini.jsonl",
    DATA_DIR / "sft_t2t.jsonl",
]

OUT_DIR = ROOT / "test37_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

MAX_RECORDS = 60000
MAX_SEQ_LEN = 1024
VAL_FRACTION = 0.02

# ------------------------------------------------------------
# Training
#
# LEARNING_RATE was previously 5e-5 which drove the KDA alpha
# gates towards 1.0 within ~800 steps and pushed the recurrent
# state into a non-recoverable region. 2e-5 keeps the gate
# dynamics slower without materially changing the convergence.
#
# GRAD_CLIP was previously 1.0. KDA's recurrent state amplifies
# gradient signals more than standard attention, so a tighter
# clip of 0.5 further reduces the chance of a bad step.
# ------------------------------------------------------------

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4          # effective batch = 16
TRAIN_STEPS = 2000
EVAL_EVERY = 200
EVAL_BATCHES = 30

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5
WARMUP_RATIO = 0.03

# If this many consecutive NaN-grad steps happen in a row,
# abort the variant. Continuing would only waste time because
# the parameters are already in a bad region.
MAX_CONSECUTIVE_NAN = 50

USE_BF16 = (
    DEVICE.type == "cuda"
    and torch.cuda.is_bf16_supported()
)

# ------------------------------------------------------------
# Variants
#
# The baseline already finished successfully on the previous run
# with final val loss 3.94408. Only the KDA variant needs to be
# re-run with the numerical fixes.
# ------------------------------------------------------------

VARIANTS = [
    ("kda_v3", True, False),
    # ("baseline_attention", False, False),
    # ("kda_v3_attnres", True, True),
]

# ------------------------------------------------------------
# Template
# ------------------------------------------------------------

ROLE_ORDER = ("system", "user", "assistant", "tool")

ROLE_MARKERS = {
    "system":    "<|system|>\n",
    "user":      "<|user|>\n",
    "assistant": "<|assistant|>\n",
    "tool":      "<|tool|>\n",
}


# ============================================================
# 2. DATA LOADING AND TEMPLATING
# ============================================================

def find_sft_file():
    for path in SFT_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No SFT file found. Tried:\n"
        + "\n".join(f"  {p}" for p in SFT_CANDIDATES)
    )


def load_sft_records(path, max_records):
    records = []
    malformed = 0
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            conversations = obj.get("conversations")
            if not isinstance(conversations, list) or not conversations:
                skipped += 1
                continue

            cleaned = []
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                role = turn.get("role")
                if role not in ROLE_ORDER:
                    continue

                content = turn.get("content", "") or ""
                has_calls = bool(turn.get("tool_calls"))
                has_tools = bool(turn.get("tools"))

                if not content and not has_calls and not has_tools:
                    continue

                cleaned.append(turn)

            if not cleaned:
                skipped += 1
                continue

            records.append({"conversations": cleaned})

            if len(records) >= max_records:
                break

    return records, malformed, skipped


def serialize_tool_calls(tool_calls):
    if tool_calls is None:
        return None
    if isinstance(tool_calls, str):
        return tool_calls
    if isinstance(tool_calls, (list, dict)):
        return json.dumps(tool_calls, ensure_ascii=False)
    return str(tool_calls)


def normalize_turn_content(turn):
    role = turn.get("role")
    content = turn.get("content", "") or ""

    if role == "assistant":
        tc = serialize_tool_calls(turn.get("tool_calls"))
        if tc:
            content = f"{content}\n{tc}".strip() if content else tc

    elif role == "system":
        tools = turn.get("tools")
        if tools:
            if not isinstance(tools, str):
                tools = json.dumps(tools, ensure_ascii=False)
            content = f"{content}\n{tools}".strip() if content else tools

    return content


def build_training_sequence(tokenizer, conversations, max_len):
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id

    input_ids = []
    labels = []

    if bos_id is not None:
        input_ids.append(int(bos_id))
        labels.append(-100)

    for turn in conversations:
        role = turn.get("role")
        marker = ROLE_MARKERS.get(role)
        if marker is None:
            continue

        content = normalize_turn_content(turn)

        marker_ids = tokenizer.encode(
            marker,
            add_special_tokens=False,
        )
        input_ids.extend(marker_ids)
        labels.extend([-100] * len(marker_ids))

        content_ids = tokenizer.encode(
            content,
            add_special_tokens=False,
        )
        input_ids.extend(content_ids)

        if role == "assistant":
            labels.extend(content_ids)
        else:
            labels.extend([-100] * len(content_ids))

        newline_ids = tokenizer.encode(
            "\n",
            add_special_tokens=False,
        )
        input_ids.extend(newline_ids)
        labels.extend([-100] * len(newline_ids))

        if role == "assistant" and eos_id is not None:
            input_ids.append(int(eos_id))
            labels.append(int(eos_id))

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return input_ids, labels


def tokenize_records(tokenizer, records, max_len, label=""):
    sequences = []

    total_tokens = 0
    total_supervised = 0

    for i, record in enumerate(records):
        input_ids, labels = build_training_sequence(
            tokenizer,
            record["conversations"],
            max_len,
        )

        if len(input_ids) < 3:
            continue

        supervised = sum(1 for t in labels if t != -100)
        if supervised == 0:
            continue

        sequences.append((input_ids, labels))

        total_tokens += len(input_ids)
        total_supervised += supervised

        if (i + 1) % 5000 == 0:
            print(f"  {label} tokenized {i + 1}/{len(records)} records")

    stats = {
        "num_sequences": len(sequences),
        "total_tokens": total_tokens,
        "total_supervised": total_supervised,
        "supervised_fraction": (
            total_supervised / total_tokens
            if total_tokens
            else 0.0
        ),
    }

    return sequences, stats


# ============================================================
# 3. BATCHING
# ============================================================

def pad_batch(batch_sequences, pad_id):
    max_len = max(len(ids) for ids, _ in batch_sequences)

    inputs = []
    labels = []
    attn = []

    for ids, labs in batch_sequences:
        pad_len = max_len - len(ids)

        inputs.append(ids + [pad_id] * pad_len)
        labels.append(labs + [-100] * pad_len)
        attn.append([1] * len(ids) + [0] * pad_len)

    return (
        torch.tensor(inputs, dtype=torch.long, device=DEVICE),
        torch.tensor(labels, dtype=torch.long, device=DEVICE),
        torch.tensor(attn, dtype=torch.long, device=DEVICE),
    )


def sample_batch(sequences, batch_size, rng):
    return [rng.choice(sequences) for _ in range(batch_size)]


# ============================================================
# 4. LOSS AND EVALUATION
# ============================================================

def compute_loss(model, input_ids, labels, attention_mask):
    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.bfloat16,
        enabled=USE_BF16,
    ):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

    loss = getattr(output, "loss", None)
    if loss is None:
        raise RuntimeError(
            "Model did not return a loss. "
            "Check MiniMindForCausalLM.forward(labels=...)."
        )

    aux_loss = getattr(output, "aux_loss", None)
    if aux_loss is not None:
        loss = loss + aux_loss

    return loss


@torch.no_grad()
def evaluate(model, sequences, batch_size, n_batches, pad_id, seed):
    model.eval()

    rng = random.Random(seed)
    losses = []

    for _ in range(n_batches):
        batch = sample_batch(sequences, batch_size, rng)
        input_ids, labels, attn = pad_batch(batch, pad_id)

        loss = compute_loss(model, input_ids, labels, attn)
        losses.append(float(loss.detach().float().item()))

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(min(mean_loss, 80.0))

    return {
        "val_loss": mean_loss,
        "val_ppl": ppl,
        "eval_batches": len(losses),
    }


# ============================================================
# 5. TRAINING ONE VARIANT
# ============================================================

def lr_at_step(step, total, base_lr, warmup_ratio):
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_variant(
    name,
    use_kda,
    use_attn_res,
    tokenizer,
    train_seqs,
    val_seqs,
    vocab_size,
    pad_id,
):
    print("\n" + "=" * 78)
    print(f"TEST37 — SFT: {name}")
    print("=" * 78)

    run_dir = OUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    config = build_config(
        vocab_size=vocab_size,
        use_kda=use_kda,
        use_attn_res=use_attn_res,
        tokenizer=tokenizer,
    )

    model = MiniMindForCausalLM(config).to(DEVICE)

    param_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"use_kda:      {use_kda}")
    print(f"use_attn_res: {use_attn_res}")
    print(f"Parameters:   {param_count:,}")
    print(f"Train seqs:   {len(train_seqs)}")
    print(f"Val seqs:     {len(val_seqs)}")
    print(f"LR:           {LEARNING_RATE}")
    print(f"Grad clip:    {GRAD_CLIP}")
    print(f"Grad accum:   {GRAD_ACCUM_STEPS} (effective batch "
          f"{BATCH_SIZE * GRAD_ACCUM_STEPS})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = random.Random(SEED)

    history = []
    nan_skips = 0
    consecutive_nan = 0

    # --------------------------------------------------------
    # Initial eval
    # --------------------------------------------------------

    initial_eval = evaluate(
        model, val_seqs, BATCH_SIZE, EVAL_BATCHES, pad_id, SEED,
    )

    print(
        f"\nInitial val loss: {initial_eval['val_loss']:.5f}  "
        f"val PPL: {initial_eval['val_ppl']:.3f}"
    )

    history.append({
        "step": 0,
        "train_loss": None,
        **initial_eval,
    })

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    model.train()

    start_time = time.perf_counter()
    running_losses = []

    for step in range(1, TRAIN_STEPS + 1):

        lr = lr_at_step(step, TRAIN_STEPS, LEARNING_RATE, WARMUP_RATIO)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        accum_losses = []

        for _ in range(GRAD_ACCUM_STEPS):
            batch = sample_batch(train_seqs, BATCH_SIZE, rng)
            input_ids, labels, attn = pad_batch(batch, pad_id)

            loss = compute_loss(model, input_ids, labels, attn)
            (loss / GRAD_ACCUM_STEPS).backward()

            accum_losses.append(float(loss.detach().float().item()))

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP,
        )

        grad_finite = bool(torch.isfinite(grad_norm).item())

        # ------------------------------------------------------
        # NaN / Inf gradient guard
        #
        # Skip the update for this step. If we hit too many in
        # a row, abort: the parameters themselves are in a bad
        # region and skipping will not recover them.
        # ------------------------------------------------------

        if not grad_finite:
            nan_skips += 1
            consecutive_nan += 1

            if consecutive_nan >= MAX_CONSECUTIVE_NAN:
                raise RuntimeError(
                    f"[{name}] Aborting: "
                    f"{consecutive_nan} consecutive non-finite "
                    f"grad steps at step {step}. "
                    f"The model has entered a permanent bad region. "
                    f"Lower max_state_norm or learning rate further."
                )

            print(
                f"[{name}] WARNING step={step}: "
                f"non-finite grad norm, skipping update "
                f"(total skips={nan_skips}, "
                f"consecutive={consecutive_nan})"
            )

            optimizer.zero_grad(set_to_none=True)
            running_losses.clear()

            continue

        # Successful update
        consecutive_nan = 0

        optimizer.step()

        train_loss = sum(accum_losses) / len(accum_losses)
        running_losses.append(train_loss)

        if len(running_losses) > 50:
            running_losses.pop(0)

        smoothed = sum(running_losses) / len(running_losses)

        if step % 10 == 0 or step == 1:
            print(
                f"[{name}] step={step:5d}/{TRAIN_STEPS} "
                f"loss={train_loss:.5f} "
                f"smoothed={smoothed:.5f} "
                f"lr={lr:.2e} "
                f"grad={float(grad_norm):.3f} "
                f"skips={nan_skips}"
            )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if step % EVAL_EVERY == 0 or step == TRAIN_STEPS:
            elapsed = time.perf_counter() - start_time
            val = evaluate(
                model, val_seqs, BATCH_SIZE, EVAL_BATCHES, pad_id, SEED,
            )

            if DEVICE.type == "cuda":
                peak_mb = (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                )
            else:
                peak_mb = None

            row = {
                "step": step,
                "train_loss": train_loss,
                "smoothed_loss": smoothed,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "nan_skips": nan_skips,
                **val,
                "elapsed_seconds": elapsed,
                "peak_vram_mb": peak_mb,
            }
            history.append(row)

            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )

            print(
                f"  VALIDATION step={step}: "
                f"val_loss={val['val_loss']:.5f}  "
                f"val_ppl={val['val_ppl']:.3f}"
            )

            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "seed": SEED,
                },
                run_dir / f"checkpoint_step_{step:05d}.pt",
            )

    # --------------------------------------------------------
    # Save final
    # --------------------------------------------------------

    torch.save(
        {
            "step": TRAIN_STEPS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "seed": SEED,
        },
        run_dir / "checkpoint_final.pt",
    )

    final_val = history[-1]

    result = {
        "name": name,
        "status": "success",
        "use_kda": use_kda,
        "use_attn_res": use_attn_res,
        "parameters": param_count,
        "initial_val_loss": history[0]["val_loss"],
        "final_val_loss": final_val["val_loss"],
        "final_val_ppl": final_val["val_ppl"],
        "elapsed_seconds": time.perf_counter() - start_time,
        "nan_skips": nan_skips,
        "history": history,
    }

    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"\nCompleted: {name}")
    print(f"  Initial val loss: {result['initial_val_loss']:.5f}")
    print(f"  Final   val loss: {result['final_val_loss']:.5f}")
    print(f"  Final   val PPL:  {result['final_val_ppl']:.3f}")
    print(f"  NaN skips:        {nan_skips}")

    del model, optimizer
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("=" * 78)
    print("TEST37 — SFT TRAINING (sft_t2t format)")
    print("=" * 78)
    print("PyTorch:", torch.__version__)
    print("Device: ", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:    ", torch.cuda.get_device_name(DEVICE))
    print("BF16:   ", USE_BF16)

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        trust_remote_code=True,
    )

    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id or 0

    print(f"Tokenizer vocab: {vocab_size}")
    print(f"BOS id: {tokenizer.bos_token_id}")
    print(f"EOS id: {tokenizer.eos_token_id}")
    print(f"PAD id: {pad_id}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    sft_file = find_sft_file()
    print(f"\nSFT file: {sft_file}")

    records, malformed, skipped = load_sft_records(
        sft_file, MAX_RECORDS,
    )

    print(f"Records loaded:     {len(records)}")
    print(f"Malformed skipped:  {malformed}")
    print(f"Empty/clean skipped:{skipped}")

    if len(records) < 100:
        raise RuntimeError(
            "Too few usable records. Check the data file."
        )

    rng = random.Random(SEED)
    rng.shuffle(records)

    n_val = max(64, int(len(records) * VAL_FRACTION))
    val_records = records[:n_val]
    train_records = records[n_val:]

    print(f"Train records:      {len(train_records)}")
    print(f"Val records:        {len(val_records)}")

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    print("\nTokenizing training records...")
    train_seqs, train_stats = tokenize_records(
        tokenizer, train_records, MAX_SEQ_LEN, label="train",
    )

    print("Tokenizing validation records...")
    val_seqs, val_stats = tokenize_records(
        tokenizer, val_records, MAX_SEQ_LEN, label="val",
    )

    print("\nDataset summary:")
    print(f"  Train sequences:  {train_stats['num_sequences']}")
    print(f"  Val sequences:    {val_stats['num_sequences']}")
    print(
        f"  Train tokens:     {train_stats['total_tokens']:,} "
        f"(supervised {train_stats['total_supervised']:,}, "
        f"{train_stats['supervised_fraction']:.2%})"
    )
    print(
        f"  Val tokens:       {val_stats['total_tokens']:,} "
        f"(supervised {val_stats['total_supervised']:,})"
    )

    if not train_seqs or not val_seqs:
        raise RuntimeError(
            "Tokenization produced no usable sequences."
        )

    print("\nFirst training sequence (decoded):")
    ids, labs = train_seqs[0]
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"  length={len(ids)} supervised={sum(1 for t in labs if t != -100)}")
    print(f"  text preview: {decoded[:300]!r}")

    # --------------------------------------------------------
    # Run each variant
    # --------------------------------------------------------

    results = []

    for name, use_kda, use_attn_res in VARIANTS:
        try:
            result = train_variant(
                name=name,
                use_kda=use_kda,
                use_attn_res=use_attn_res,
                tokenizer=tokenizer,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                vocab_size=vocab_size,
                pad_id=pad_id,
            )
            results.append(result)

        except Exception as exc:
            print("\n" + "!" * 78)
            print(f"FAILED: {name}")
            print(traceback.format_exc())
            print("!" * 78)

            results.append({
                "name": name,
                "status": "failed",
                "error": str(exc),
            })

            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_path = OUT_DIR / "test37_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("TEST37 FINAL SUMMARY")
    print("=" * 78)

    print(
        f"{'Variant':<22}"
        f"{'Params':>14}"
        f"{'Init Loss':>12}"
        f"{'Final Loss':>12}"
        f"{'Final PPL':>12}"
        f"{'Skips':>8}"
    )
    print("-" * 82)

    for r in results:
        if r.get("status") != "success":
            print(f"{r['name']:<22} FAILED: {r.get('error')}")
            continue

        print(
            f"{r['name']:<22}"
            f"{r['parameters']:>14,}"
            f"{r['initial_val_loss']:>12.5f}"
            f"{r['final_val_loss']:>12.5f}"
            f"{r['final_val_ppl']:>12.3f}"
            f"{r.get('nan_skips', 0):>8d}"
        )

    print()
    print("Results: ", summary_path)
    print("Checkpoints:", OUT_DIR)


if __name__ == "__main__":
    main()