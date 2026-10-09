
# -*- coding: utf-8 -*-
"""
TEST34 — REAL CORPUS TRAINING & VALIDATION

Variants:
  1. baseline_attention
  2. kda_v3
  3. kda_v3_attnres

Protocol:
  - Read real UTF-8 JSONL records from pretrain_t2t_mini.jsonl
  - Deterministic record-level train/validation split
  - Causal next-token prediction with explicit label shifting
  - Identical token budget, batch size, optimizer and training steps
  - Evaluate held-out validation blocks
  - Save JSON results and model checkpoints

Run from PyCharm using the minimind-kda Python environment.
"""

import os
import sys
import json
import time
import math
import random
import inspect
import traceback
from pathlib import Path
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "pretrain_t2t_mini.jsonl"
OUTPUT_DIR = ROOT / "test34_outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_DIR = ROOT / "model"

SEED = 3401

# Pilot experiment: bounded data and compute budget.
MAX_RECORDS = 30000
TRAIN_RATIO = 0.90

SEQ_LEN = 128
BATCH_SIZE = 2
TRAIN_STEPS = 100
EVAL_BATCHES = 40
EVAL_INTERVAL = 25

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

USE_BF16 = True
SAVE_CHECKPOINTS = True

VARIANTS = [
    ("baseline_attention", False, False),
    ("kda_v3", True, False),
    ("kda_v3_attnres", True, True),
]


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep kernels free to select efficient implementations.
    # This is not a bitwise-determinism guarantee.
    torch.backends.cudnn.benchmark = False


def print_header(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_autocast_dtype(device):
    if device.type != "cuda" or not USE_BF16:
        return None

    if hasattr(torch.cuda, "is_bf16_supported"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

    return None


# ============================================================
# 3. TOKENIZER AND MODEL IMPORTS
# ============================================================

def load_project_components():
    """
    Reuse the project's existing build_config function so model
    configuration stays consistent with TEST22.
    """
    sys.path.insert(0, str(ROOT))

    try:
        from test22_real_text_comparison import build_config
    except Exception as exc:
        raise RuntimeError(
            "Cannot import build_config from "
            "test22_real_text_comparison.py. "
            "Keep that script in the project root and check its imports."
        ) from exc

    try:
        from model.model_minimind import MiniMindForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "Cannot import MiniMindForCausalLM from "
            "model/model_minimind.py. "
            "Check the model class name in that file."
        ) from exc

    return build_config, MiniMindForCausalLM


def load_tokenizer():
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(
            f"Tokenizer directory not found: {TOKENIZER_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"Tokenizer directory: {TOKENIZER_DIR}")
    print(f"Tokenizer vocabulary: {len(tokenizer)}")

    return tokenizer


# ============================================================
# 4. READ RECORDS AND SPLIT BY RECORD
# ============================================================

def load_text_records():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    records = []
    malformed = 0
    missing_text = 0

    print(f"Dataset: {DATA_PATH}")
    print(f"Maximum records for this pilot: {MAX_RECORDS}")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if len(records) >= MAX_RECORDS:
                break

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            text = item.get("text") if isinstance(item, dict) else None

            if not isinstance(text, str) or not text.strip():
                missing_text += 1
                continue

            records.append(text.strip())

            if len(records) % 5000 == 0:
                print(f"  Loaded {len(records):,} records")

    if len(records) < 100:
        raise RuntimeError(
            f"Only {len(records)} valid records loaded; "
            "please check the JSONL file."
        )

    print(f"Valid records loaded: {len(records):,}")
    print(f"Malformed JSON lines skipped: {malformed}")
    print(f"Missing/empty text records skipped: {missing_text}")

    # Shuffle records before splitting; records are split before tokenization.
    rng = random.Random(SEED)
    rng.shuffle(records)

    split_at = int(len(records) * TRAIN_RATIO)
    train_records = records[:split_at]
    val_records = records[split_at:]

    if not train_records or not val_records:
        raise RuntimeError("Train/validation split is empty.")

    print(f"Training records:   {len(train_records):,}")
    print(f"Validation records: {len(val_records):,}")

    return train_records, val_records


# ============================================================
# 5. TOKENIZE AND BUILD CAUSAL-LM BLOCKS
# ============================================================

def encode_records(records, tokenizer, description):
    """
    Each record is tokenized independently. EOS is appended between
    records when available, so record boundaries are represented.

    Records are already assigned to train or validation before this
    function is called, preventing the same record from crossing splits.
    """
    eos_id = tokenizer.eos_token_id
    all_ids = []

    total_records = len(records)

    for i, text in enumerate(records, start=1):
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if not ids:
            continue

        all_ids.extend(ids)

        if eos_id is not None:
            all_ids.append(eos_id)

        if i % 5000 == 0 or i == total_records:
            print(
                f"  {description}: tokenized "
                f"{i:,}/{total_records:,} records"
            )

    return all_ids


def make_blocks(token_ids, seq_len):
    """
    Each block contains seq_len + 1 tokens.
    Input:  block[:, :-1]
    Target: block[:, 1:]

    The target at position t is the next token after input position t.
    """
    block_width = seq_len + 1
    usable = (len(token_ids) // block_width) * block_width

    if usable == 0:
        raise RuntimeError(
            f"Not enough tokens to create blocks of length {block_width}."
        )

    token_ids = token_ids[:usable]

    array = np.asarray(token_ids, dtype=np.int64)
    blocks = array.reshape(-1, block_width)

    return torch.from_numpy(blocks.copy())


def prepare_dataset(tokenizer):
    train_records, val_records = load_text_records()

    print("\nTokenizing training records...")
    train_ids = encode_records(
        train_records, tokenizer, "train"
    )

    print("\nTokenizing validation records...")
    val_ids = encode_records(
        val_records, tokenizer, "validation"
    )

    train_blocks = make_blocks(train_ids, SEQ_LEN)
    val_blocks = make_blocks(val_ids, SEQ_LEN)

    print("\nDataset summary:")
    print(f"Training tokens before block trimming:   {len(train_ids):,}")
    print(f"Validation tokens before block trimming: {len(val_ids):,}")
    print(f"Training blocks:   {len(train_blocks):,}")
    print(f"Validation blocks: {len(val_blocks):,}")
    print(f"Tokens per model input: {SEQ_LEN}")
    print(f"Tokens per stored block: {SEQ_LEN + 1}")

    if len(train_blocks) < BATCH_SIZE:
        raise RuntimeError("Too few training blocks for the selected batch size.")

    if len(val_blocks) < BATCH_SIZE:
        raise RuntimeError("Too few validation blocks for the selected batch size.")

    return train_blocks, val_blocks


# ============================================================
# 6. MODEL OUTPUT NORMALIZATION
# ============================================================

def extract_logits(output):
    """
    Support common model return conventions:
      - Tensor logits
      - object with .logits
      - dict with 'logits'
      - tuple/list whose first element is logits
    """
    if torch.is_tensor(output):
        return output

    if hasattr(output, "logits"):
        logits = output.logits
        if torch.is_tensor(logits):
            return logits

    if isinstance(output, dict):
        logits = output.get("logits")
        if torch.is_tensor(logits):
            return logits

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item

    raise TypeError(
        "Could not extract [batch, sequence, vocab] logits from model output. "
        f"Output type: {type(output)}"
    )


def forward_logits(model, input_ids):
    """
    This experiment computes the loss externally to make the causal
    next-token shift explicit and consistent across all variants.
    """
    output = model(input_ids)
    logits = extract_logits(output)

    if logits.ndim != 3:
        raise RuntimeError(
            f"Expected logits with 3 dimensions [B,T,V], got {tuple(logits.shape)}"
        )

    if logits.shape[0] != input_ids.shape[0]:
        raise RuntimeError(
            f"Batch mismatch: input={tuple(input_ids.shape)}, "
            f"logits={tuple(logits.shape)}"
        )

    if logits.shape[1] != input_ids.shape[1]:
        raise RuntimeError(
            f"Sequence mismatch: input={tuple(input_ids.shape)}, "
            f"logits={tuple(logits.shape)}"
        )

    return logits



def causal_lm_loss(logits, block):
    """
    Correct causal next-token alignment.

    block:       [B, T+1]
    input_ids:   block[:, :-1] -> [B, T]
    logits:      [B, T, vocab_size]
    targets:     block[:, 1:]  -> [B, T]

    Logits at position t predict the token at position t+1.
    Do NOT slice logits[:, :-1], because the input is already
    one token shorter than the stored block.
    """
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits [B,T,V], got {tuple(logits.shape)}"
        )

    targets = block[:, 1:].contiguous()
    prediction_logits = logits.contiguous().float()

    if prediction_logits.shape[:2] != targets.shape:
        raise ValueError(
            "Causal LM alignment mismatch: "
            f"logits={tuple(prediction_logits.shape)}, "
            f"targets={tuple(targets.shape)}"
        )

    return F.cross_entropy(
        prediction_logits.reshape(-1, prediction_logits.shape[-1]),
        targets.reshape(-1),
    )

# ============================================================
# 7. BATCH SAMPLING AND EVALUATION
# ============================================================

def sample_batch(blocks, batch_size, device, generator=None):
    indices = torch.randint(
        low=0,
        high=len(blocks),
        size=(batch_size,),
        generator=generator,
    )

    batch = blocks[indices].to(device, non_blocking=True)
    return batch


@torch.no_grad()
def evaluate(model, val_blocks, device, autocast_dtype, seed):
    model.eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    losses = []
    total_tokens = 0

    for _ in range(EVAL_BATCHES):
        block = sample_batch(
            val_blocks, BATCH_SIZE, device, generator
        )

        input_ids = block[:, :-1]

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=(autocast_dtype is not None),
        ):
            logits = forward_logits(model, input_ids)

        loss = causal_lm_loss(logits, block)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss detected.")

        losses.append(float(loss.item()))
        total_tokens += BATCH_SIZE * SEQ_LEN

    model.train()

    mean_loss = float(np.mean(losses))
    ppl = math.exp(min(mean_loss, 20.0))

    return {
        "val_loss": mean_loss,
        "val_ppl": ppl,
        "eval_tokens": total_tokens,
    }


# ============================================================
# 8. MODEL CONSTRUCTION
# ============================================================

def build_model(
    build_config,
    model_class,
    tokenizer,
    use_kda,
    use_attn_res,
    device,
):
    config = build_config(
        vocab_size=len(tokenizer),
        use_kda=use_kda,
        use_attn_res=use_attn_res,
        tokenizer=tokenizer,
    )

    model = model_class(config).to(device)

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# 9. TRAIN ONE VARIANT
# ============================================================

def train_variant(
    name,
    use_kda,
    use_attn_res,
    build_config,
    model_class,
    tokenizer,
    train_blocks,
    val_blocks,
    device,
    autocast_dtype,
):
    print_header(f"TRAINING — {name}")

    set_seed(SEED)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(
        build_config=build_config,
        model_class=model_class,
        tokenizer=tokenizer,
        use_kda=use_kda,
        use_attn_res=use_attn_res,
        device=device,
    )

    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    print(f"Device: {device}")
    print(f"Autocast dtype: {autocast_dtype}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Separate RNG for data sampling to keep the batch sequence
    # reproducible across model variants.
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(SEED + 100)

    train_losses = []
    step_times = []
    eval_history = []

    model.train()

    # Warm-up CUDA kernels and optimizer states before timing.
    warmup_steps = 3

    print(f"Warm-up steps: {warmup_steps}")
    print(f"Measured training steps: {TRAIN_STEPS}")

    for step in range(1, TRAIN_STEPS + 1):
        block = sample_batch(
            train_blocks, BATCH_SIZE, device, batch_generator
        )
        input_ids = block[:, :-1]

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_start = time.perf_counter()

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=(autocast_dtype is not None),
        ):
            logits = forward_logits(model, input_ids)
            loss = causal_lm_loss(logits, block)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{name}: non-finite training loss at step {step}: "
                f"{loss.item()}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"{name}: non-finite gradient norm at step {step}."
            )

        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_elapsed = time.perf_counter() - step_start

        if step > warmup_steps:
            step_times.append(step_elapsed)

        train_losses.append(float(loss.item()))

        if step == 1 or step % 10 == 0 or step == TRAIN_STEPS:
            print(
                f"[{name}] step={step:03d}/{TRAIN_STEPS} "
                f"loss={loss.item():.4f} "
                f"grad_norm={float(grad_norm.item()):.4f} "
                f"step_time={step_elapsed * 1000:.1f} ms"
            )

        if step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS:
            metrics = evaluate(
                model=model,
                val_blocks=val_blocks,
                device=device,
                autocast_dtype=autocast_dtype,
                seed=SEED + step,
            )

            metrics["step"] = step
            eval_history.append(metrics)

            print(
                f"  VALIDATION step={step}: "
                f"loss={metrics['val_loss']:.4f}, "
                f"PPL={metrics['val_ppl']:.2f}"
            )

    # Final validation with a fixed seed for reproducibility.
    final_eval = evaluate(
        model=model,
        val_blocks=val_blocks,
        device=device,
        autocast_dtype=autocast_dtype,
        seed=SEED + 9999,
    )

    elapsed = float(sum(step_times))
    median_step_time = (
        float(median(step_times)) if step_times else None
    )

    measured_steps = max(1, len(step_times))
    measured_tokens = measured_steps * BATCH_SIZE * SEQ_LEN

    throughput = (
        measured_tokens / elapsed if elapsed > 0 else None
    )

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

    checkpoint_path = None

    if SAVE_CHECKPOINTS:
        checkpoint_path = CHECKPOINT_DIR / f"{name}.pt"

        torch.save(
            {
                "variant": name,
                "seed": SEED,
                "model_state_dict": model.state_dict(),
                "config": {
                    "use_kda": use_kda,
                    "use_attn_res": use_attn_res,
                    "seq_len": SEQ_LEN,
                    "batch_size": BATCH_SIZE,
                    "train_steps": TRAIN_STEPS,
                    "learning_rate": LEARNING_RATE,
                },
                "final_val_loss": final_eval["val_loss"],
                "final_val_ppl": final_eval["val_ppl"],
            },
            checkpoint_path,
        )

    result = {
        "variant": name,
        "use_kda": use_kda,
        "use_attn_res": use_attn_res,
        "seed": SEED,
        "parameters": params,
        "train_records": int(len(train_blocks)),
        "validation_records": int(len(val_blocks)),
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "train_loss_first": float(train_losses[0]),
        "train_loss_last": float(train_losses[-1]),
        "train_loss_mean_last_10": float(
            np.mean(train_losses[-10:])
        ),
        "final_val_loss": float(final_eval["val_loss"]),
        "final_val_ppl": float(final_eval["val_ppl"]),
        "measured_step_count": len(step_times),
        "median_step_time_ms": (
            median_step_time * 1000
            if median_step_time is not None
            else None
        ),
        "mean_measured_step_time_ms": (
            elapsed / measured_steps * 1000
            if elapsed > 0
            else None
        ),
        "measured_tokens_per_second": throughput,
        "measured_training_seconds": elapsed,
        "peak_memory_mb": peak_memory_mb,
        "validation_history": eval_history,
        "checkpoint": (
            str(checkpoint_path) if checkpoint_path else None
        ),
    }

    del optimizer
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# 10. MAIN
# ============================================================

def main():
    print_header("TEST34 — REAL CORPUS TRAINING & VALIDATION")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Dataset: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}"
        )

    set_seed(SEED)

    device = get_device()
    autocast_dtype = get_autocast_dtype(device)

    build_config, model_class = load_project_components()
    tokenizer = load_tokenizer()

    train_blocks, val_blocks = prepare_dataset(tokenizer)

    all_results = {
        "test": "TEST34_REAL_CORPUS_TRAINING",
        "seed": SEED,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "pytorch": torch.__version__,
        "dataset": str(DATA_PATH),
        "pilot_record_limit": MAX_RECORDS,
        "train_block_count": len(train_blocks),
        "validation_block_count": len(val_blocks),
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "train_steps": TRAIN_STEPS,
        "eval_batches": EVAL_BATCHES,
        "autocast_dtype": str(autocast_dtype),
        "results": [],
    }

    for name, use_kda, use_attn_res in VARIANTS:
        result = train_variant(
            name=name,
            use_kda=use_kda,
            use_attn_res=use_attn_res,
            build_config=build_config,
            model_class=model_class,
            tokenizer=tokenizer,
            train_blocks=train_blocks,
            val_blocks=val_blocks,
            device=device,
            autocast_dtype=autocast_dtype,
        )

        all_results["results"].append(result)

        result_path = OUTPUT_DIR / "test34_results_partial.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(
                all_results,
                f,
                ensure_ascii=False,
                indent=2,
            )

    final_path = OUTPUT_DIR / "test34_results.json"

    with final_path.open("w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_header("TEST34 — FINAL SUMMARY")

    print(
        f"{'Variant':<24} "
        f"{'Params':>12} "
        f"{'Val Loss':>10} "
        f"{'Val PPL':>10} "
        f"{'Tok/s':>12} "
        f"{'VRAM MB':>10}"
    )

    print("-" * 88)

    for result in all_results["results"]:
        tok_s = result["measured_tokens_per_second"]
        vram = result["peak_memory_mb"]

        print(
            f"{result['variant']:<24} "
            f"{result['parameters']:>12,} "
            f"{result['final_val_loss']:>10.4f} "
            f"{result['final_val_ppl']:>10.2f} "
            f"{tok_s if tok_s is not None else float('nan'):>12.2f} "
            f"{vram if vram is not None else float('nan'):>10.1f}"
        )

    print("-" * 88)
    print(f"Results JSON: {final_path}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print("TEST34 completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 88)
        print("TEST34 FAILED")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("!" * 88)
        raise