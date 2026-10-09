
# -*- coding: utf-8 -*-
"""
TEST 21 — MiniMind + KDA V3 TRAINING PREFLIGHT

Project:
    D:\\appplication\\pycharm\\minimind-kda

Model package:
    model.model_minimind
    model.kda_attention_v3
    model.attention_residual

Checks:
    1. Environment
    2. Model import
    3. Baseline Attention forward/backward/update
    4. KDA V3 forward/backward/update
    5. KDA V3 + AttnRes forward/backward/update
    6. Finite loss/logits/gradients
    7. Optimizer parameter update
    8. CUDA peak memory and approximate throughput

This is a smoke/preflight test, not a model-quality benchmark.
It does not establish that KDA improves perplexity or speed.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Actual model package discovered in the project directory.
MODEL_MODULE_NAME = "model.model_minimind"

# Change this only if your actual model implementation is elsewhere.
KDA_MODULE_NAME = "model.kda_attention_v3"
ATTN_RES_MODULE_NAME = "model.attention_residual"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.float32

SEED = 20261009

BATCH_SIZE = 2
SEQUENCE_LENGTH = 64
VOCAB_SIZE = 256

# This preflight is deliberately small. It does not load a pretrained model.
SMALL_HIDDEN_SIZE = 128
SMALL_NUM_LAYERS = 4
SMALL_NUM_HEADS = 4
SMALL_NUM_KV_HEADS = 4
SMALL_INTERMEDIATE_SIZE = 256
SMALL_MAX_POSITION = 128

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

# Number of training iterations per configuration.
TRAINING_STEPS = 5

OUTPUT_DIR = PROJECT_ROOT / "test21_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUTPUT_DIR / "test21_results.json"
RESULT_LOG = OUTPUT_DIR / "test21_log.txt"

# If True, also test the model configuration with AttnRes enabled.
TEST_ATTN_RES = True


# ============================================================
# 2. GLOBAL RESULTS
# ============================================================

RESULTS: list[dict[str, Any]] = []
FAILURES: list[str] = []


def log(message: str = "") -> None:
    """Print and append a line to the test log."""
    print(message)
    with RESULT_LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def record_result(
    name: str,
    passed: bool,
    details: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    result = {
        "name": name,
        "passed": bool(passed),
        "details": details,
        "extra": extra or {},
    }
    RESULTS.append(result)

    status = "PASS" if passed else "FAIL"
    log(f"[{status}] {name}")

    if details:
        log(f"       {details}")

    if not passed:
        FAILURES.append(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total, trainable


def get_model_logits(output: Any) -> torch.Tensor:
    """
    Extract logits from common MiniMind/Hugging Face output structures.
    """
    if isinstance(output, torch.Tensor):
        return output

    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits

    if isinstance(output, dict):
        logits = output.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits

    if isinstance(output, (tuple, list)) and output:
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 3:
                return item

    raise TypeError(
        "Cannot find a [batch, sequence, vocabulary] logits tensor "
        f"in model output type {type(output).__name__}."
    )


def get_model_loss(output: Any) -> torch.Tensor | None:
    loss = getattr(output, "loss", None)
    if isinstance(loss, torch.Tensor):
        return loss

    if isinstance(output, dict):
        loss = output.get("loss")
        if isinstance(loss, torch.Tensor):
            return loss

    if isinstance(output, (tuple, list)) and output:
        # Some implementations return (loss, logits, ...).
        if isinstance(output[0], torch.Tensor) and output[0].ndim == 0:
            return output[0]

    return None


# ============================================================
# 3. ENVIRONMENT
# ============================================================

def test_environment() -> None:
    log("=" * 78)
    log("TEST 21 — MiniMind + KDA V3 TRAINING PREFLIGHT")
    log("=" * 78)

    log(f"Timestamp         : {datetime.now().isoformat(timespec='seconds')}")
    log(f"Python executable : {sys.executable}")
    log(f"Python version    : {sys.version.split()[0]}")
    log(f"PyTorch           : {torch.__version__}")
    log(f"CUDA available    : {torch.cuda.is_available()}")
    log(f"Device            : {DEVICE}")
    log(f"Model dtype       : {MODEL_DTYPE}")
    log(f"Batch size        : {BATCH_SIZE}")
    log(f"Sequence length   : {SEQUENCE_LENGTH}")
    log(f"Vocabulary size   : {VOCAB_SIZE}")
    log(f"Training steps    : {TRAINING_STEPS}")
    log(f"Project root      : {PROJECT_ROOT}")

    if torch.cuda.is_available():
        log(f"GPU               : {torch.cuda.get_device_name(0)}")
        log(f"CUDA runtime      : {torch.version.cuda}")

        props = torch.cuda.get_device_properties(0)
        total_vram_gib = props.total_memory / (1024**3)
        log(f"Total VRAM        : {total_vram_gib:.2f} GiB")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    if torch.cuda.is_available():
        try:
            x = torch.randn(4, 4, device="cuda")
            y = x @ x.T
            synchronize_cuda()

            if not tensor_is_finite(y):
                raise RuntimeError("CUDA smoke test produced non-finite values.")

        except Exception as exc:
            record_result("Environment", False, repr(exc))
            return

    record_result(
        "Environment",
        True,
        f"device={DEVICE}, dtype={MODEL_DTYPE}",
    )


# ============================================================
# 4. MODEL IMPORT
# ============================================================

def import_model_classes():
    """
    Import the real model package.

    The project structure indicates:
        model/model_minimind.py
        model/kda_attention_v3.py
        model/attention_residual.py
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    log(f"Project root: {PROJECT_ROOT}")
    log(f"Model module: {MODEL_MODULE_NAME}")

    module = importlib.import_module(MODEL_MODULE_NAME)

    config_cls = getattr(module, "MiniMindConfig", None)
    model_cls = getattr(module, "MiniMindForCausalLM", None)

    if config_cls is None:
        raise ImportError(
            f"{MODEL_MODULE_NAME} does not expose MiniMindConfig."
        )

    if model_cls is None:
        raise ImportError(
            f"{MODEL_MODULE_NAME} does not expose MiniMindForCausalLM."
        )

    # Verify that the KDA V3 file can be imported.
    kda_module = importlib.import_module(KDA_MODULE_NAME)

    if not hasattr(kda_module, "KDAAttentionV3"):
        raise ImportError(
            f"{KDA_MODULE_NAME} does not expose KDAAttentionV3."
        )

    # AttnRes is optional for the baseline and KDA-only tests.
    attn_res_module = None
    attn_res_import_error = None

    if TEST_ATTN_RES:
        try:
            attn_res_module = importlib.import_module(ATTN_RES_MODULE_NAME)
        except Exception as exc:
            attn_res_import_error = repr(exc)

    log(f"Config class      : {config_cls.__module__}.{config_cls.__name__}")
    log(f"Model class       : {model_cls.__module__}.{model_cls.__name__}")
    log(f"KDA V3 class      : {kda_module.KDAAttentionV3.__module__}."
        f"{kda_module.KDAAttentionV3.__name__}")

    if attn_res_module is not None:
        log(f"AttnRes module    : {ATTN_RES_MODULE_NAME}")
    elif TEST_ATTN_RES:
        log(f"AttnRes import warning: {attn_res_import_error}")

    return config_cls, model_cls, attn_res_module


# ============================================================
# 5. CONFIGURATION ADAPTER
# ============================================================

def build_config(config_cls, use_kda: bool, use_attn_res: bool):
    """
    Construct a small config using only parameters supported by the
    actual MiniMindConfig constructor.

    This avoids passing unsupported keyword arguments, but deliberately
    does not silently assume a missing KDA switch exists.
    """
    try:
        signature = inspect.signature(config_cls)
        parameters = signature.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in parameters.values()
        )
    except Exception:
        parameters = {}
        accepts_kwargs = True

    candidate_values = {
        "hidden_size": SMALL_HIDDEN_SIZE,
        "num_hidden_layers": SMALL_NUM_LAYERS,
        "num_attention_heads": SMALL_NUM_HEADS,
        "num_key_value_heads": SMALL_NUM_KV_HEADS,
        "intermediate_size": SMALL_INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": SMALL_MAX_POSITION,
        "dropout": 0.0,
        "dropout_p": 0.0,
        "use_kda": use_kda,
        "use_attn_res": use_attn_res,
        "use_moe": False,
        "flash_attn": False,
        "use_flash_attn": False,
        "attn_res_block_size": 2,
        "attn_res_max_blocks": 4,
        "tie_word_embeddings": False,
    }

    accepted = {}

    for key, value in candidate_values.items():
        if accepts_kwargs or key in parameters:
            accepted[key] = value

    # Require the important architectural toggles when requested.
    for toggle_name, toggle_value in (
        ("use_kda", use_kda),
        ("use_attn_res", use_attn_res),
    ):
        if toggle_value and not (
            accepts_kwargs or toggle_name in parameters
        ):
            raise TypeError(
                f"{config_cls.__name__} does not accept {toggle_name}. "
                "The current model config may not be wired to this feature."
            )

    config = config_cls(**accepted)

    # Some MiniMind config classes use mutable attributes.
    # Set the feature flags explicitly if the object permits it.
    for name, value in (
        ("use_kda", use_kda),
        ("use_attn_res", use_attn_res),
    ):
        try:
            setattr(config, name, value)
        except Exception:
            pass

    # Try to set small dimensions if they were not constructor arguments
    # but are mutable config attributes.
    for name, value in (
        ("hidden_size", SMALL_HIDDEN_SIZE),
        ("num_hidden_layers", SMALL_NUM_LAYERS),
        ("num_attention_heads", SMALL_NUM_HEADS),
        ("num_key_value_heads", SMALL_NUM_KV_HEADS),
        ("intermediate_size", SMALL_INTERMEDIATE_SIZE),
        ("vocab_size", VOCAB_SIZE),
        ("max_position_embeddings", SMALL_MAX_POSITION),
    ):
        if hasattr(config, name):
            try:
                setattr(config, name, value)
            except Exception:
                pass

    return config


def validate_config_flags(config, use_kda: bool, use_attn_res: bool) -> None:
    if use_kda and getattr(config, "use_kda", None) is not True:
        raise RuntimeError(
            "KDA test requested, but config.use_kda is not True. "
            "Inspect MiniMindConfig and the model's attention selection."
        )

    if use_attn_res and getattr(config, "use_attn_res", None) is not True:
        raise RuntimeError(
            "AttnRes test requested, but config.use_attn_res is not True."
        )


# ============================================================
# 6. MODEL BUILDING
# ============================================================

def build_model(config_cls, model_cls, use_kda: bool, use_attn_res: bool):
    config = build_config(
        config_cls=config_cls,
        use_kda=use_kda,
        use_attn_res=use_attn_res,
    )

    validate_config_flags(config, use_kda, use_attn_res)

    model = model_cls(config)

    if not isinstance(model, nn.Module):
        raise TypeError(
            "MiniMindForCausalLM(config) did not return a torch.nn.Module."
        )

    model = model.to(device=DEVICE, dtype=MODEL_DTYPE)

    return config, model


# ============================================================
# 7. SYNTHETIC BATCH
# ============================================================

def create_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random token IDs and language-model labels.

    The synthetic batch checks tensor mechanics only. It is not a
    meaningful language-model dataset.
    """
    input_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, SEQUENCE_LENGTH),
        device=DEVICE,
        dtype=torch.long,
    )

    labels = input_ids.clone()

    return input_ids, labels


# ============================================================
# 8. FORWARD ADAPTER
# ============================================================

def forward_model(model: nn.Module, input_ids: torch.Tensor,
                  labels: torch.Tensor):
    """
    Try common MiniMind forward signatures.

    If the model returns logits but does not calculate loss internally,
    compute causal language-model cross entropy here.
    """
    forward_signature = inspect.signature(model.forward)
    parameters = forward_signature.parameters

    kwargs = {}

    if "input_ids" in parameters:
        kwargs["input_ids"] = input_ids

    if "labels" in parameters:
        kwargs["labels"] = labels

    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 0

    if "use_cache" in parameters:
        kwargs["use_cache"] = False

    if not kwargs:
        output = model(input_ids, labels=labels)
    else:
        output = model(**kwargs)

    logits = get_model_logits(output)
    loss = get_model_loss(output)

    if logits.ndim != 3:
        raise RuntimeError(
            f"Expected 3D logits, got shape {tuple(logits.shape)}."
        )

    if logits.shape[0] != input_ids.shape[0]:
        raise RuntimeError(
            f"Logits batch mismatch: {tuple(logits.shape)}."
        )

    if logits.shape[-1] != VOCAB_SIZE:
        raise RuntimeError(
            f"Logits vocabulary dimension is {logits.shape[-1]}, "
            f"expected {VOCAB_SIZE}. "
            "Check whether the config's vocab_size is respected."
        )

    if loss is None:
        # Causal next-token prediction: token t predicts token t+1.
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )

    if loss.ndim != 0:
        raise RuntimeError(
            f"Expected scalar loss, got shape {tuple(loss.shape)}."
        )

    return output, logits, loss


# ============================================================
# 9. ONE CONFIGURATION TEST
# ============================================================

def run_configuration(
    config_cls,
    model_cls,
    name: str,
    use_kda: bool,
    use_attn_res: bool,
) -> dict[str, Any]:
    log("")
    log("=" * 78)
    log(f"CONFIGURATION: {name}")
    log("=" * 78)

    model = None
    optimizer = None

    try:
        set_seed(SEED)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        config, model = build_model(
            config_cls,
            model_cls,
            use_kda=use_kda,
            use_attn_res=use_attn_res,
        )

        total_params, trainable_params = count_parameters(model)

        log(f"Total parameters     : {total_params:,}")
        log(f"Trainable parameters : {trainable_params:,}")
        log(f"Config use_kda       : {getattr(config, 'use_kda', None)}")
        log(
            f"Config use_attn_res  : "
            f"{getattr(config, 'use_attn_res', None)}"
        )

        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        input_ids, labels = create_batch()

        step_losses = []
        step_times = []
        all_grads_finite = True
        any_grad_found = False
        optimizer_update_verified = False
        output_shape = None
        max_abs_grad = 0.0
        grad_norm_last = 0.0

        for step in range(1, TRAINING_STEPS + 1):
            optimizer.zero_grad(set_to_none=True)

            synchronize_cuda()
            start_time = time.perf_counter()

            _, logits, loss = forward_model(model, input_ids, labels)

            if not tensor_is_finite(logits):
                raise FloatingPointError(
                    f"Step {step}: logits contain NaN or Inf."
                )

            if not tensor_is_finite(loss):
                raise FloatingPointError(
                    f"Step {step}: loss is NaN or Inf."
                )

            output_shape = list(logits.shape)
            loss_value = float(loss.detach().item())

            loss.backward()

            grad_sq_sum = 0.0
            found_grad_this_step = False
            max_abs_grad_this_step = 0.0
            bad_gradient_names = []

            for parameter_name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                found_grad_this_step = True

                if not tensor_is_finite(grad):
                    all_grads_finite = False
                    bad_gradient_names.append(parameter_name)
                    continue

                if grad.numel() > 0:
                    grad_abs_max = float(grad.detach().abs().max().item())
                    max_abs_grad_this_step = max(
                        max_abs_grad_this_step,
                        grad_abs_max,
                    )

                    grad_sq_sum += float(
                        grad.detach().float().pow(2).sum().item()
                    )

            any_grad_found = any_grad_found or found_grad_this_step
            max_abs_grad = max(max_abs_grad, max_abs_grad_this_step)
            grad_norm_last = math.sqrt(grad_sq_sum)

            if not found_grad_this_step:
                raise RuntimeError(
                    f"Step {step}: no trainable parameter received gradients."
                )

            if bad_gradient_names:
                raise FloatingPointError(
                    f"Step {step}: non-finite gradients in "
                    f"{bad_gradient_names[:10]}"
                )

            # Verify that at least one parameter with a gradient changes
            # after an optimizer step.
            parameter_to_check = None
            parameter_before_step = None

            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    parameter_to_check = parameter
                    parameter_before_step = parameter.detach().clone()
                    break

            if parameter_to_check is None:
                raise RuntimeError("No parameter available for update check.")

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
                error_if_nonfinite=True,
            )

            optimizer.step()

            if not tensor_is_finite(parameter_to_check):
                raise FloatingPointError(
                    f"Step {step}: parameter became NaN/Inf after optimizer.step()."
                )

            changed = not torch.equal(
                parameter_before_step,
                parameter_to_check.detach(),
            )

            if not changed:
                raise RuntimeError(
                    f"Step {step}: checked parameter did not change after "
                    "optimizer.step()."
                )

            optimizer_update_verified = True

            synchronize_cuda()
            elapsed = time.perf_counter() - start_time

            step_losses.append(loss_value)
            step_times.append(elapsed)

            log(
                f"Step {step:02d}/{TRAINING_STEPS} | "
                f"loss={loss_value:.6f} | "
                f"grad_norm={grad_norm_last:.6f} | "
                f"max_abs_grad={max_abs_grad_this_step:.6f} | "
                f"time={elapsed:.3f}s | "
                f"parameter_updated={changed}"
            )

            del logits, loss

        mean_step_time = sum(step_times) / max(len(step_times), 1)
        tokens_per_step = BATCH_SIZE * max(SEQUENCE_LENGTH - 1, 1)
        tokens_per_second = tokens_per_step / max(mean_step_time, 1e-12)

        peak_memory_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = (
                torch.cuda.max_memory_allocated() / (1024**2)
            )

        summary = {
            "configuration": name,
            "passed": True,
            "use_kda": use_kda,
            "use_attn_res": use_attn_res,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "output_shape": output_shape,
            "steps": TRAINING_STEPS,
            "losses": step_losses,
            "initial_loss": step_losses[0] if step_losses else None,
            "final_loss": step_losses[-1] if step_losses else None,
            "all_gradients_finite": all_grads_finite,
            "any_gradient_found": any_grad_found,
            "max_abs_gradient": max_abs_grad,
            "last_gradient_norm": grad_norm_last,
            "optimizer_update_verified": optimizer_update_verified,
            "mean_step_seconds": mean_step_time,
            "approx_tokens_per_second": tokens_per_second,
            "peak_cuda_memory_mb": peak_memory_mb,
        }

        log("")
        log(f"Output shape             : {output_shape}")
        log(f"Initial loss             : {summary['initial_loss']:.6f}")
        log(f"Final loss               : {summary['final_loss']:.6f}")
        log(f"All gradients finite     : {all_grads_finite}")
        log(f"Any gradient found       : {any_grad_found}")
        log(f"Max absolute gradient    : {max_abs_grad:.6f}")
        log(f"Last gradient norm       : {grad_norm_last:.6f}")
        log(f"Optimizer update verified: {optimizer_update_verified}")
        log(f"Mean step time           : {mean_step_time:.4f}s")
        log(f"Approx tokens/s          : {tokens_per_second:.2f}")

        if peak_memory_mb is not None:
            log(f"Peak allocated VRAM      : {peak_memory_mb:.2f} MiB")

        record_result(
            name,
            True,
            "forward/backward/finite gradients/optimizer update passed",
            summary,
        )

        return summary

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"

        log(f"[ERROR] {name}: {error_text}")
        log(traceback.format_exc())

        summary = {
            "configuration": name,
            "passed": False,
            "error": error_text,
            "traceback": traceback.format_exc(),
        }

        record_result(name, False, error_text, summary)

        return summary

    finally:
        if optimizer is not None:
            del optimizer

        if model is not None:
            del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 10. RESULT OUTPUT
# ============================================================

def save_results() -> None:
    payload = {
        "test_name": "TEST 21 — MiniMind + KDA V3 TRAINING PREFLIGHT",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(DEVICE),
        "project_root": str(PROJECT_ROOT),
        "model_module": MODEL_MODULE_NAME,
        "settings": {
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "vocab_size": VOCAB_SIZE,
            "training_steps": TRAINING_STEPS,
            "hidden_size": SMALL_HIDDEN_SIZE,
            "num_layers": SMALL_NUM_LAYERS,
            "num_heads": SMALL_NUM_HEADS,
        },
        "results": RESULTS,
        "failed_checks": FAILURES,
        "passed_count": sum(1 for x in RESULTS if x["passed"]),
        "failed_count": sum(1 for x in RESULTS if not x["passed"]),
    }

    with RESULT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log("")
    log(f"JSON results saved: {RESULT_JSON}")
    log(f"Text log saved   : {RESULT_LOG}")


# ============================================================
# 11. MAIN
# ============================================================

def main() -> int:
    # Clear the previous log for an unambiguous test report.
    RESULT_LOG.write_text("", encoding="utf-8")

    test_environment()

    if FAILURES:
        log("Environment check failed; stopping.")
        save_results()
        return 1

    try:
        config_cls, model_cls, attn_res_module = import_model_classes()

        record_result(
            "Model Import",
            True,
            f"loaded {MODEL_MODULE_NAME}; KDA V3 module imported",
        )

    except Exception as exc:
        record_result(
            "Model Import",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        log(traceback.format_exc())
        save_results()
        print_summary()
        return 1

    # Configuration 1: standard attention baseline.
    run_configuration(
        config_cls=config_cls,
        model_cls=model_cls,
        name="Baseline Attention",
        use_kda=False,
        use_attn_res=False,
    )

    # Configuration 2: KDA V3 only.
    run_configuration(
        config_cls=config_cls,
        model_cls=model_cls,
        name="KDA V3",
        use_kda=True,
        use_attn_res=False,
    )

    # Configuration 3: KDA V3 + AttnRes.
    if TEST_ATTN_RES:
        if attn_res_module is None:
            record_result(
                "KDA V3 + AttnRes",
                False,
                "AttnRes module could not be imported; see import warning.",
            )
        else:
            run_configuration(
                config_cls=config_cls,
                model_cls=model_cls,
                name="KDA V3 + AttnRes",
                use_kda=True,
                use_attn_res=True,
            )

    save_results()
    print_summary()

    return 1 if FAILURES else 0


def print_summary() -> None:
    log("")
    log("=" * 78)
    log("TEST 21 SUMMARY")
    log("=" * 78)

    for result in RESULTS:
        status = "PASS" if result["passed"] else "FAIL"
        log(f"[{status}] {result['name']}: {result['details']}")

    passed_count = sum(1 for x in RESULTS if x["passed"])
    failed_count = sum(1 for x in RESULTS if not x["passed"])

    log("-" * 78)
    log(f"Passed: {passed_count}")
    log(f"Failed: {failed_count}")

    if failed_count == 0:
        log("RESULT: PREFLIGHT PASSED.")
        log(
            "Next: run a controlled training experiment and evaluate "
            "validation perplexity on a real held-out dataset."
        )
    else:
        log("RESULT: PREFLIGHT FAILED.")
        log(
            "Fix the first failing check before interpreting performance "
            "or starting a longer training run."
        )

    log("=" * 78)


if __name__ == "__main__":
    exit_code = main()
    raise SystemExit(exit_code)
