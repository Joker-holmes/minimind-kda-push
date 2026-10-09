# -*- coding: utf-8 -*-
"""
TEST 22 — MiniMind real-text controlled comparison

Experiments:
1. Baseline Attention
2. KDA V3
3. KDA V3 + Block Attention Residual

Fixes and safeguards:
- Validation batches are moved to the model device.
- Training and validation use the same tokenized corpus.
- Tokenizer special-token IDs are used instead of hard-coded IDs.
- Each experiment has an independent output directory.
- Old checkpoints outside test22_outputs are not modified.
- Errors are saved to error.json and do not stop later experiments.
- Writes a summary JSON and CSV.
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import csv
import gc
import json
import math
import random
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.model_minimind import (
    MiniMindConfig,
    MiniMindForCausalLM,
)


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "test10_1_data"
TRAIN_FILE = DATA_DIR / "train.txt"
VALID_FILE = DATA_DIR / "valid.txt"
TOKENIZER_DIR = ROOT / "model"

OUTPUT_DIR = ROOT / "test22_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261009

HIDDEN_SIZE = 768
NUM_LAYERS = 4
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS

SEQ_LEN = 128
BATCH_SIZE = 2
TRAIN_STEPS = 200
EVAL_EVERY = 50
EVAL_BATCHES = 40

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

EXPERIMENTS = [
    {
        "name": "baseline_attention",
        "use_kda": False,
        "use_attn_res": False,
    },
    {
        "name": "kda_v3",
        "use_kda": True,
        "use_attn_res": False,
    },
    {
        "name": "kda_v3_attnres",
        "use_kda": True,
        "use_attn_res": True,
    },
]


# ============================================================
# 2. DEVICE AND REPRODUCIBILITY
# ============================================================

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

USE_BF16 = (
    DEVICE.type == "cuda"
    and torch.cuda.is_bf16_supported()
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def print_environment():
    print("=" * 88)
    print("TEST 22 — REAL TEXT CONTROLLED COMPARISON")
    print("=" * 88)
    print(f"Project:       {ROOT}")
    print(f"Train file:    {TRAIN_FILE}")
    print(f"Valid file:    {VALID_FILE}")
    print(f"Tokenizer:     {TOKENIZER_DIR}")
    print(f"Output:        {OUTPUT_DIR}")
    print(f"Device:        {DEVICE}")
    print(f"BF16 autocast: {USE_BF16}")

    if DEVICE.type == "cuda":
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime:  {torch.version.cuda}")

    print(f"PyTorch:       {torch.__version__}")
    print(f"Sequence len:  {SEQ_LEN}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Training steps:{TRAIN_STEPS}")
    print()


# ============================================================
# 3. FILES AND TOKENIZATION
# ============================================================

def read_text_file(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required data file not found: {path}"
        )

    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030")


def encode_text(tokenizer, text):
    """
    Encode each non-empty line independently.
    Append EOS after each non-empty line when EOS is available.
    """
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id

    all_ids = []

    if bos_id is not None:
        all_ids.append(int(bos_id))

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        ids = tokenizer.encode(
            line,
            add_special_tokens=False,
        )

        if ids:
            all_ids.extend(int(token_id) for token_id in ids)

            if eos_id is not None:
                all_ids.append(int(eos_id))

    if len(all_ids) < SEQ_LEN + 2:
        raise ValueError(
            f"Not enough tokens: got {len(all_ids)}, "
            f"need at least {SEQ_LEN + 2}."
        )

    return torch.tensor(all_ids, dtype=torch.long)


def load_data_and_tokenizer():
    print("=" * 88)
    print("1. LOAD TOKENIZER AND REAL TEXT")
    print("=" * 88)

    for path in (TRAIN_FILE, VALID_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"Data file missing: {path}\n"
                "Check the project path."
            )

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        local_files_only=True,
    )

    train_text = read_text_file(TRAIN_FILE)
    valid_text = read_text_file(VALID_FILE)

    train_ids = encode_text(tokenizer, train_text)
    valid_ids = encode_text(tokenizer, valid_text)

    tokenizer_size = len(tokenizer)
    max_token_id = max(
        int(train_ids.max().item()),
        int(valid_ids.max().item()),
    )
    vocab_size = max(tokenizer_size, max_token_id + 1)

    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Tokenizer size:  {tokenizer_size}")
    print(f"Model vocab size:{vocab_size}")
    print(f"Train chars:     {len(train_text):,}")
    print(f"Valid chars:     {len(valid_text):,}")
    print(f"Train tokens:    {len(train_ids):,}")
    print(f"Valid tokens:    {len(valid_ids):,}")
    print(f"BOS token ID:    {tokenizer.bos_token_id}")
    print(f"EOS token ID:    {tokenizer.eos_token_id}")
    print(f"PAD token ID:    {tokenizer.pad_token_id}")
    print()

    return tokenizer, train_ids, valid_ids, vocab_size


# ============================================================
# 4. FIXED BATCH SCHEDULE
# ============================================================

def make_batch_schedule(train_length):
    max_start = train_length - SEQ_LEN - 1

    if max_start <= 0:
        raise ValueError("Training text is too short.")

    rng = random.Random(SEED)
    schedule = []

    for _ in range(TRAIN_STEPS):
        starts = [
            rng.randint(0, max_start)
            for _ in range(BATCH_SIZE)
        ]
        schedule.append(starts)

    return schedule


def make_train_batch(tokens, starts):
    # Keep the entire tokenized corpus on CPU.
    x = torch.stack([
        tokens[s:s + SEQ_LEN]
        for s in starts
    ])

    # Preserve the original MiniMind label convention.
    # Whether labels are shifted internally depends on the model's
    # forward() implementation, so do not change it silently here.
    y = x.clone()

    return (
        x.to(DEVICE, non_blocking=True),
        y.to(DEVICE, non_blocking=True),
    )


def make_eval_batches(tokens):
    """
    Create deterministic validation windows on CPU.
    evaluate() transfers each batch to the model device.
    """
    total = len(tokens)
    starts = list(range(0, total - SEQ_LEN, SEQ_LEN))

    if not starts:
        raise ValueError("Validation text is too short.")

    max_windows = EVAL_BATCHES * BATCH_SIZE

    if len(starts) > max_windows:
        rng = random.Random(SEED + 1)
        starts = sorted(
            rng.sample(starts, max_windows)
        )

    batches = []

    for offset in range(0, len(starts), BATCH_SIZE):
        selected = starts[offset:offset + BATCH_SIZE]

        if len(selected) != BATCH_SIZE:
            continue

        x = torch.stack([
            tokens[s:s + SEQ_LEN]
            for s in selected
        ])

        batches.append(x)

    if not batches:
        raise ValueError(
            "No complete validation batches were created."
        )

    return batches


# ============================================================
# 5. MODEL CONFIGURATION
# ============================================================

def build_config(
    vocab_size,
    use_kda,
    use_attn_res,
    tokenizer,
):
    """
    Use tokenizer IDs when available.
    Keep all other architecture settings identical.
    """
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # Some tokenizers do not define a PAD token.
    # For this unpadded causal-LM experiment, EOS is a safe metadata
    # fallback when PAD is absent.
    if pad_token_id is None:
        pad_token_id = eos_token_id

    kwargs = {
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "use_moe": False,
        "vocab_size": vocab_size,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "use_kda": use_kda,
        "use_attn_res": use_attn_res,
        "attn_res_max_blocks": NUM_LAYERS,
        "attn_res_dropout": 0.0,
        "flash_attn": False,
        "dropout": 0.0,
        "tie_word_embeddings": True,
    }

    if bos_token_id is not None:
        kwargs["bos_token_id"] = int(bos_token_id)

    if eos_token_id is not None:
        kwargs["eos_token_id"] = int(eos_token_id)

    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)

    return MiniMindConfig(**kwargs)


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def extract_loss(output):
    loss = getattr(output, "loss", None)

    if loss is None:
        raise RuntimeError(
            "Model forward did not return a loss. "
            "Check MiniMindForCausalLM.forward."
        )

    aux_loss = getattr(output, "aux_loss", None)

    if aux_loss is not None:
        loss = loss + aux_loss

    return loss


def assert_model_device(model):
    """Check model placement without confusing cuda and cuda:0."""
    try:
        parameter = next(model.parameters())
    except StopIteration:
        raise RuntimeError("Model has no parameters.")

    expected = DEVICE
    actual = parameter.device

    same_device = (
        actual.type == expected.type
        and (
            expected.index is None
            or actual.index == expected.index
        )
    )

    if not same_device:
        raise RuntimeError(
            f"Model device mismatch: expected {expected}, "
            f"got {actual}"
        )


# ============================================================
# 6. VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(model, eval_batches):
    """
    FIX: evaluation batches are on CPU.
    Move each input and label tensor to the model device before forward.
    """
    model.eval()
    losses = []

    assert_model_device(model)

    model_device = next(model.parameters()).device

    for x_cpu in eval_batches:
        x = x_cpu.to(model_device, non_blocking=True)
        labels = x

        if x.device != model_device:
            raise RuntimeError(
                f"Validation input device mismatch: "
                f"input={x.device}, model={model_device}"
            )

        with torch.autocast(
            device_type=model_device.type,
            dtype=torch.bfloat16,
            enabled=USE_BF16,
        ):
            output = model(
                input_ids=x,
                labels=labels,
                use_cache=False,
            )
            loss = extract_loss(output)

        value = float(loss.detach().float().item())

        if not math.isfinite(value):
            raise FloatingPointError(
                f"Non-finite validation loss: {value}"
            )

        losses.append(value)

    if not losses:
        raise RuntimeError("Validation produced no loss values.")

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(min(mean_loss, 80.0))

    return {
        "val_loss": mean_loss,
        "val_ppl": perplexity,
        "eval_batches": len(losses),
    }


# ============================================================
# 7. OUTPUT HELPERS
# ============================================================

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            obj,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(path)


def save_checkpoint(
    model,
    optimizer,
    run_dir,
    step,
    config_dict,
):
    checkpoint_path = (
        run_dir / f"checkpoint_step_{step:04d}.pt"
    )

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_dict,
            "seed": SEED,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved: {checkpoint_path}")


# ============================================================
# 8. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    experiment,
    vocab_size,
    tokenizer,
    train_ids,
    eval_batches,
    schedule,
):
    name = experiment["name"]

    run_dir = OUTPUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"EXPERIMENT: {name}")
    print("=" * 88)

    set_seed(SEED)

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    config = build_config(
        vocab_size=vocab_size,
        use_kda=experiment["use_kda"],
        use_attn_res=experiment["use_attn_res"],
        tokenizer=tokenizer,
    )

    model = MiniMindForCausalLM(config).to(DEVICE)
    assert_model_device(model)

    param_count = count_parameters(model)

    print(f"use_kda:       {experiment['use_kda']}")
    print(f"use_attn_res:  {experiment['use_attn_res']}")
    print(f"Parameters:    {param_count:,}")
    print(f"Learning rate: {LEARNING_RATE}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config_dict = {
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": vocab_size,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "use_kda": experiment["use_kda"],
        "use_attn_res": experiment["use_attn_res"],
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    history = []
    start_time = time.perf_counter()
    tokens_processed = 0

    try:
        initial_eval = evaluate(model, eval_batches)

        print(
            f"Initial validation loss: "
            f"{initial_eval['val_loss']:.5f}"
        )
        print(
            f"Initial validation PPL:  "
            f"{initial_eval['val_ppl']:.5f}"
        )

        history.append({
            "step": 0,
            "train_loss": None,
            **initial_eval,
            "elapsed_seconds": 0.0,
        })

        model.train()

        for step in range(1, TRAIN_STEPS + 1):
            x, y = make_train_batch(
                train_ids,
                schedule[step - 1],
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.bfloat16,
                enabled=USE_BF16,
            ):
                output = model(
                    input_ids=x,
                    labels=y,
                    use_cache=False,
                )
                loss = extract_loss(output)

            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    f"Non-finite training loss at step {step}: "
                    f"{loss.detach().float().item()}"
                )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at step {step}"
                )

            optimizer.step()

            train_loss = float(
                loss.detach().float().item()
            )

            tokens_processed += BATCH_SIZE * SEQ_LEN

            should_evaluate = (
                step % EVAL_EVERY == 0
                or step == TRAIN_STEPS
            )

            if should_evaluate:
                synchronize()
                elapsed = time.perf_counter() - start_time

                val_result = evaluate(model, eval_batches)

                if DEVICE.type == "cuda":
                    peak_vram_mb = (
                        torch.cuda.max_memory_allocated()
                        / (1024 ** 2)
                    )
                else:
                    peak_vram_mb = None

                tokens_per_second = (
                    tokens_processed / elapsed
                    if elapsed > 0
                    else 0.0
                )

                row = {
                    "step": step,
                    "train_loss": train_loss,
                    **val_result,
                    "elapsed_seconds": elapsed,
                    "tokens_processed": tokens_processed,
                    "tokens_per_second": tokens_per_second,
                    "peak_vram_mb": peak_vram_mb,
                    "grad_norm": float(
                        grad_norm.detach().float().item()
                    ),
                }

                history.append(row)

                write_json(
                    run_dir / "history.json",
                    history,
                )

                print(
                    f"[{name}] step={step:4d}/{TRAIN_STEPS} "
                    f"train_loss={train_loss:.5f} "
                    f"val_loss={val_result['val_loss']:.5f} "
                    f"val_ppl={val_result['val_ppl']:.3f} "
                    f"tokens/s={tokens_per_second:.1f} "
                    f"elapsed={elapsed:.1f}s"
                )

                save_checkpoint(
                    model,
                    optimizer,
                    run_dir,
                    step,
                    config_dict,
                )

        synchronize()
        total_elapsed = time.perf_counter() - start_time

        if DEVICE.type == "cuda":
            peak_vram_mb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )
        else:
            peak_vram_mb = None

        final_eval = evaluate(model, eval_batches)

        initial_loss = history[0]["val_loss"]
        final_loss = final_eval["val_loss"]

        result = {
            "status": "success",
            "name": name,
            "config": config_dict,
            "trainable_parameters": param_count,
            "initial_val_loss": initial_loss,
            "final_val_loss": final_loss,
            "initial_val_ppl": history[0]["val_ppl"],
            "final_val_ppl": final_eval["val_ppl"],
            "val_loss_change": final_loss - initial_loss,
            "val_loss_reduction_percent": (
                100.0 * (initial_loss - final_loss) / initial_loss
                if initial_loss != 0
                else None
            ),
            "elapsed_seconds": total_elapsed,
            "tokens_processed": tokens_processed,
            "tokens_per_second": (
                tokens_processed / total_elapsed
                if total_elapsed > 0
                else 0.0
            ),
            "peak_vram_mb": peak_vram_mb,
            "history": history,
        }

        write_json(run_dir / "result.json", result)

        print(f"\nCompleted: {name}")
        print(f"Initial val loss: {initial_loss:.5f}")
        print(f"Final val loss:   {final_loss:.5f}")
        print(f"Final val PPL:    {final_eval['val_ppl']:.5f}")
        print(f"Elapsed:          {total_elapsed:.1f}s")
        print(f"Peak VRAM:        {peak_vram_mb}")

        return result

    finally:
        del model
        del optimizer
        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


# ============================================================
# 9. SUMMARY
# ============================================================

def write_summary(results):
    successful = [
        result
        for result in results
        if result.get("status") == "success"
    ]

    summary_path = OUTPUT_DIR / "test22_summary.json"
    write_json(summary_path, results)

    csv_path = OUTPUT_DIR / "test22_summary.csv"

    columns = [
        "name",
        "status",
        "trainable_parameters",
        "initial_val_loss",
        "final_val_loss",
        "initial_val_ppl",
        "final_val_ppl",
        "val_loss_reduction_percent",
        "elapsed_seconds",
        "tokens_per_second",
        "peak_vram_mb",
        "error",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print("\n" + "=" * 88)
    print("TEST 22 SUMMARY")
    print("=" * 88)

    if not successful:
        print("No experiment completed successfully.")
    else:
        print(
            f"{'Experiment':<25}"
            f"{'Val Loss':>12}"
            f"{'Val PPL':>12}"
            f"{'Tokens/s':>12}"
            f"{'VRAM MB':>12}"
        )
        print("-" * 73)

        for result in successful:
            vram = result.get("peak_vram_mb")
            vram_text = (
                f"{vram:.1f}"
                if vram is not None
                else "N/A"
            )

            print(
                f"{result['name']:<25}"
                f"{result['final_val_loss']:>12.5f}"
                f"{result['final_val_ppl']:>12.3f}"
                f"{result['tokens_per_second']:>12.1f}"
                f"{vram_text:>12}"
            )

    print(f"\nJSON: {summary_path}")
    print(f"CSV:  {csv_path}")

    if len(successful) == len(EXPERIMENTS):
        print("\nAll three experiments completed.")
    else:
        print(
            f"\nCompleted {len(successful)}/{len(EXPERIMENTS)} "
            "experiments. Check each run's error.json."
        )


# ============================================================
# 10. MAIN
# ============================================================

def main():
    print_environment()

    model_source = ROOT / "model" / "model_minimind.py"

    if not model_source.is_file():
        raise FileNotFoundError(
            f"Model source not found: {model_source}"
        )

    tokenizer, train_ids, valid_ids, vocab_size = (
        load_data_and_tokenizer()
    )

    schedule = make_batch_schedule(len(train_ids))
    eval_batches = make_eval_batches(valid_ids)

    print("=" * 88)
    print("2. EXPERIMENT PLAN")
    print("=" * 88)
    print(f"Training windows:   {len(schedule)} steps")
    print(f"Validation batches: {len(eval_batches)}")
    print("Old checkpoints:    untouched")
    print("Shared batch order: enabled")
    print("Validation device:  transferred to model device")
    print()

    all_results = []

    for experiment in EXPERIMENTS:
        try:
            result = run_experiment(
                experiment=experiment,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                train_ids=train_ids,
                eval_batches=eval_batches,
                schedule=schedule,
            )
            all_results.append(result)

        except Exception as exc:
            error_text = traceback.format_exc()

            print("\n" + "!" * 88)
            print(f"EXPERIMENT FAILED: {experiment['name']}")
            print(f"Error: {exc}")
            print(error_text)
            print("!" * 88)

            failed_result = {
                "status": "failed",
                "name": experiment["name"],
                "error": str(exc),
                "traceback": error_text,
            }

            write_json(
                OUTPUT_DIR / experiment["name"] / "error.json",
                failed_result,
            )

            all_results.append(failed_result)

            gc.collect()

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    write_summary(all_results)


if __name__ == "__main__":
    main()