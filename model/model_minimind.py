import math

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers import (
    PreTrainedModel,
    GenerationMixin,
    PretrainedConfig,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .kda_attention_v3 import KDAAttentionV3
from .attention_residual import BlockAttentionResidualWrapper


# ================================================================
# MiniMind Config
# ================================================================

class MiniMindConfig(PretrainedConfig):

    model_type = "minimind"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe

        self.dropout = kwargs.get(
            "dropout",
            0.0,
        )

        self.vocab_size = kwargs.get(
            "vocab_size",
            6400,
        )

        self.bos_token_id = kwargs.get(
            "bos_token_id",
            1,
        )

        self.eos_token_id = kwargs.get(
            "eos_token_id",
            2,
        )

        self.pad_token_id = kwargs.get(
            "pad_token_id",
            self.eos_token_id,
        )

        self.flash_attn = kwargs.get(
            "flash_attn",
            True,
        )

        self.num_attention_heads = kwargs.get(
            "num_attention_heads",
            8,
        )

        self.num_key_value_heads = kwargs.get(
            "num_key_value_heads",
            4,
        )

        self.head_dim = kwargs.get(
            "head_dim",
            self.hidden_size
            // self.num_attention_heads,
        )

        self.hidden_act = kwargs.get(
            "hidden_act",
            "silu",
        )

        self.intermediate_size = kwargs.get(
            "intermediate_size",
            math.ceil(
                hidden_size * math.pi / 64
            ) * 64,
        )

        self.max_position_embeddings = kwargs.get(
            "max_position_embeddings",
            32768,
        )

        self.rms_norm_eps = kwargs.get(
            "rms_norm_eps",
            1e-6,
        )

        self.rope_theta = kwargs.get(
            "rope_theta",
            1e6,
        )

        self.tie_word_embeddings = kwargs.get(
            "tie_word_embeddings",
            True,
        )

        self.inference_rope_scaling = kwargs.get(
            "inference_rope_scaling",
            False,
        )

        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn",
        } if self.inference_rope_scaling else None

        # ------------------------------------------------------------
        # MoE
        # ------------------------------------------------------------

        self.num_experts = kwargs.get(
            "num_experts",
            4,
        )

        self.num_experts_per_tok = kwargs.get(
            "num_experts_per_tok",
            1,
        )

        self.moe_intermediate_size = kwargs.get(
            "moe_intermediate_size",
            self.intermediate_size,
        )

        self.norm_topk_prob = kwargs.get(
            "norm_topk_prob",
            True,
        )

        self.router_aux_loss_coef = kwargs.get(
            "router_aux_loss_coef",
            5e-4,
        )

        # ------------------------------------------------------------
        # Attention Backend
        # ------------------------------------------------------------

        self.use_kda = kwargs.get(
            "use_kda",
            True,
        )

        # ------------------------------------------------------------
        # Block Attention Residual
        # ------------------------------------------------------------

        self.use_attn_res = kwargs.get(
            "use_attn_res",
            True,
        )

        self.attn_res_block_size = kwargs.get(
            "attn_res_block_size",
            2,
        )

        self.attn_res_max_blocks = kwargs.get(
            "attn_res_max_blocks",
            8,
        )

        self.attn_res_dropout = kwargs.get(
            "attn_res_dropout",
            0.0,
        )


# ================================================================
# RMSNorm
# ================================================================

class RMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def norm(self, x):

        return x * torch.rsqrt(
            x.pow(2).mean(
                -1,
                keepdim=True,
            )
            + self.eps
        )

    def forward(self, x):

        return (
            self.weight
            * self.norm(x.float())
        ).type_as(x)


# ================================================================
# RoPE utilities
# ================================================================

def precompute_freqs_cis(
    dim,
    end=int(32 * 1024),
    rope_base=1e6,
    rope_scaling=None,
):

    freqs = (
        1.0
        / (
            rope_base
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                )[
                    : dim // 2
                ].float()
                / dim
            )
        )
    )

    attn_factor = 1.0

    if rope_scaling is not None:

        (
            orig_max,
            factor,
            beta_fast,
            beta_slow,
            attn_factor,
        ) = (
            rope_scaling.get(
                "original_max_position_embeddings",
                2048,
            ),
            rope_scaling.get(
                "factor",
                16,
            ),
            rope_scaling.get(
                "beta_fast",
                32.0,
            ),
            rope_scaling.get(
                "beta_slow",
                1.0,
            ),
            rope_scaling.get(
                "attention_factor",
                1.0,
            ),
        )

        if end / orig_max > 1.0:

            inv_dim = lambda b: (
                dim
                * math.log(
                    orig_max
                    / (
                        b
                        * 2
                        * math.pi
                    )
                )
                / (
                    2
                    * math.log(rope_base)
                )
            )

            low = max(
                math.floor(
                    inv_dim(beta_fast)
                ),
                0,
            )

            high = min(
                math.ceil(
                    inv_dim(beta_slow)
                ),
                dim // 2 - 1,
            )

            ramp = torch.clamp(
                (
                    torch.arange(
                        dim // 2,
                    ).float()
                    - low
                )
                / max(
                    high - low,
                    0.001,
                ),
                0,
                1,
            )

            freqs = freqs * (
                1
                - ramp
                + ramp / factor
            )

    t = torch.arange(end)

    freqs = torch.outer(
        t,
        freqs,
    ).float()

    freqs_cos = (
        torch.cat(
            [
                torch.cos(freqs),
                torch.cos(freqs),
            ],
            dim=-1,
        )
        * attn_factor
    )

    freqs_sin = (
        torch.cat(
            [
                torch.sin(freqs),
                torch.sin(freqs),
            ],
            dim=-1,
        )
        * attn_factor
    )

    return freqs_cos, freqs_sin


# ================================================================
# RoPE
# ================================================================

def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    unsqueeze_dim=1,
):

    def rotate_half(x):

        return torch.cat(
            (
                -x[
                    ...,
                    x.shape[-1] // 2:
                ],
                x[
                    ...,
                    : x.shape[-1] // 2
                ],
            ),
            dim=-1,
        )

    q_embed = (
        q * cos.unsqueeze(
            unsqueeze_dim
        )
        +
        rotate_half(q)
        * sin.unsqueeze(
            unsqueeze_dim
        )
    ).to(q.dtype)

    k_embed = (
        k * cos.unsqueeze(
            unsqueeze_dim
        )
        +
        rotate_half(k)
        * sin.unsqueeze(
            unsqueeze_dim
        )
    ).to(k.dtype)

    return q_embed, k_embed


# ================================================================
# Repeat KV
#
# Current cache layout:
#
#     [B, seq_len, KV_heads, head_dim]
#
# After repeat:
#
#     [B, seq_len, attention_heads, head_dim]
# ================================================================

def repeat_kv(
    x: torch.Tensor,
    n_rep: int,
):

    bs, slen, num_key_value_heads, head_dim = (
        x.shape
    )

    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(
            bs,
            slen,
            num_key_value_heads,
            n_rep,
            head_dim,
        )
        .reshape(
            bs,
            slen,
            num_key_value_heads * n_rep,
            head_dim,
        )
    )


# ================================================================
# Standard Transformer Attention
#
# IMPORTANT:
#
# Cache layout:
#
#     K = [B, T, KV_heads, D]
#     V = [B, T, KV_heads, D]
#
# NOT:
#
#     [B, KV_heads, T, D]
#
# This matches the existing MiniMind implementation and TEST 11
# cache-size calculation.
# ================================================================

class Attention(nn.Module):

    def __init__(
        self,
        config: MiniMindConfig,
    ):
        super().__init__()

        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )

        self.n_local_heads = (
            config.num_attention_heads
        )

        self.n_local_kv_heads = (
            self.num_key_value_heads
        )

        if (
            self.n_local_heads
            % self.n_local_kv_heads
            != 0
        ):
            raise ValueError(
                "num_attention_heads must be divisible "
                "by num_key_value_heads."
            )

        self.n_rep = (
            self.n_local_heads
            // self.n_local_kv_heads
        )

        self.head_dim = config.head_dim

        self.is_causal = True

        # ------------------------------------------------------------
        # QKV projections
        # ------------------------------------------------------------

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads
            * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads
            * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads
            * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads
            * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        # ------------------------------------------------------------
        # Q/K RMSNorm
        # ------------------------------------------------------------

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        self.k_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        # ------------------------------------------------------------
        # Dropout
        # ------------------------------------------------------------

        self.attn_dropout = nn.Dropout(
            config.dropout
        )

        self.resid_dropout = nn.Dropout(
            config.dropout
        )

        self.dropout = config.dropout

        # ------------------------------------------------------------
        # SDPA
        # ------------------------------------------------------------

        self.flash = (
            hasattr(
                torch.nn.functional,
                "scaled_dot_product_attention",
            )
            and config.flash_attn
        )

    def forward(
        self,
        x,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):

        bsz, seq_len, _ = x.shape

        # ============================================================
        # Q / K / V projection
        # ============================================================

        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # ============================================================
        # [B, T, hidden]
        #
        # ->
        #
        # Q [B, T, H, D]
        # K [B, T, KV, D]
        # V [B, T, KV, D]
        # ============================================================

        xq = xq.view(
            bsz,
            seq_len,
            self.n_local_heads,
            self.head_dim,
        )

        xk = xk.view(
            bsz,
            seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )

        xv = xv.view(
            bsz,
            seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )

        # ============================================================
        # Q/K normalization
        # ============================================================

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # ============================================================
        # RoPE
        #
        # position_embeddings MUST already contain the correct
        # absolute positions.
        #
        # Prefill:
        #
        #     positions = [0 ... T-1]
        #
        # Cached decode:
        #
        #     positions = [past_length ... past_length+T-1]
        # ============================================================

        cos, sin = position_embeddings

        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            cos,
            sin,
        )

        # ============================================================
        # Append KV cache
        #
        # Current layout:
        #
        #     [B, T, KV_heads, D]
        #
        # Sequence dimension = dim 1
        # ============================================================

        if past_key_value is not None:

            past_key = past_key_value[0]
            past_value = past_key_value[1]

            if past_key.ndim != 4:
                raise ValueError(
                    "Invalid key cache shape: "
                    f"{tuple(past_key.shape)}"
                )

            if past_value.ndim != 4:
                raise ValueError(
                    "Invalid value cache shape: "
                    f"{tuple(past_value.shape)}"
                )

            xk = torch.cat(
                [
                    past_key,
                    xk,
                ],
                dim=1,
            )

            xv = torch.cat(
                [
                    past_value,
                    xv,
                ],
                dim=1,
            )

        # ============================================================
        # Save cache BEFORE KV repetition
        #
        # This is important for GQA:
        #
        # cache:
        #     [B, T, KV_heads, D]
        #
        # attention:
        #     [B, H, T, D]
        # ============================================================

        past_kv = (
            (xk, xv)
            if use_cache
            else None
        )

        # ============================================================
        # Convert Q to:
        #
        #     [B, H, Tq, D]
        # ============================================================

        xq = xq.transpose(
            1,
            2,
        )

        # ============================================================
        # Repeat K/V:
        #
        # [B, T, KV, D]
        #
        # ->
        #
        # [B, T, H, D]
        #
        # ->
        #
        # [B, H, T, D]
        # ============================================================

        xk = repeat_kv(
            xk,
            self.n_rep,
        ).transpose(
            1,
            2,
        )

        xv = repeat_kv(
            xv,
            self.n_rep,
        ).transpose(
            1,
            2,
        )

        # ============================================================
        # Attention
        # ============================================================

        has_past = (
            past_key_value is not None
        )

        # ------------------------------------------------------------
        # Case A:
        #
        # Prefill
        #
        # q_len > 1
        # no cache
        #
        # SDPA causal attention is mathematically equivalent to the
        # normal causal attention used by the Transformer.
        # ------------------------------------------------------------

        if (
            self.flash
            and seq_len > 1
            and not has_past
            and (
                attention_mask is None
                or torch.all(
                    attention_mask == 1
                )
            )
        ):

            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=(
                    self.dropout
                    if self.training
                    else 0.0
                ),
                is_causal=True,
            )

        # ------------------------------------------------------------
        # Case B:
        #
        # Full attention with an existing cache and q_len > 1.
        #
        # This is uncommon in TEST 11, but we support it correctly.
        #
        # The query corresponds to the final seq_len positions.
        # ------------------------------------------------------------

        elif (
            seq_len > 1
        ):

            scores = (
                xq
                @ xk.transpose(
                    -2,
                    -1,
                )
            ).float()

            scores = (
                scores
                / math.sqrt(
                    self.head_dim
                )
            )

            total_kv_len = xk.shape[-2]

            past_len = (
                total_kv_len
                - seq_len
            )

            if attention_mask is not None:

                # ----------------------------------------------------
                # Standard attention mask:
                #
                # [B, total_kv_len]
                # ----------------------------------------------------

                if (
                    attention_mask.ndim == 2
                    and attention_mask.shape[-1]
                    == total_kv_len
                ):

                    mask = (
                        1.0
                        - attention_mask[
                            :, None, None, :
                        ]
                    ) * -1e9

                    scores = (
                        scores
                        + mask
                    )

                else:

                    scores = (
                        scores
                        + attention_mask
                    )

            # --------------------------------------------------------
            # Causal mask for query positions:
            #
            # query position:
            #
            #     past_len + i
            #
            # key position:
            #
            #     j
            #
            # valid when:
            #
            #     j <= past_len + i
            # --------------------------------------------------------

            query_positions = (
                torch.arange(
                    seq_len,
                    device=xq.device,
                )
                + past_len
            )

            key_positions = torch.arange(
                total_kv_len,
                device=xq.device,
            )

            causal_mask = (
                key_positions.unsqueeze(0)
                > query_positions.unsqueeze(1)
            )

            scores = scores.masked_fill(
                causal_mask[
                    None,
                    None,
                    :,
                    :,
                ],
                float("-inf"),
            )

            output = (
                F.softmax(
                    scores,
                    dim=-1,
                ).type_as(xq)
                @ xv
            )

        # ------------------------------------------------------------
        # Case C:
        #
        # Single-token cached decoding.
        #
        # THIS IS THE IMPORTANT TEST 11 PATH.
        #
        # Query:
        #
        #     q_len = 1
        #
        # Keys:
        #
        #     [past tokens + current token]
        #
        # All keys are valid.
        #
        # Therefore:
        #
        #     NO causal mask.
        #
        # Applying a q_len x q_len mask here is unnecessary and can
        # introduce subtle cache-path inconsistencies.
        # ------------------------------------------------------------

        else:

            scores = (
                xq
                @ xk.transpose(
                    -2,
                    -1,
                )
            ).float()

            scores = (
                scores
                / math.sqrt(
                    self.head_dim
                )
            )

            # --------------------------------------------------------
            # Attention mask
            # --------------------------------------------------------

            if attention_mask is not None:

                if (
                    attention_mask.ndim == 2
                    and attention_mask.shape[-1]
                    == xk.shape[-2]
                ):

                    scores = (
                        scores
                        + (
                            1.0
                            - attention_mask[
                                :, None, None, :
                            ]
                        )
                        * -1e9
                    )

                elif (
                    attention_mask.ndim
                    == 4
                ):

                    scores = (
                        scores
                        + attention_mask
                    )

            # --------------------------------------------------------
            # IMPORTANT:
            #
            # No causal mask here.
            #
            # Current query is at the end of the cache.
            # Every cached key is causally visible.
            # --------------------------------------------------------

            output = (
                F.softmax(
                    scores,
                    dim=-1,
                ).type_as(xq)
                @ xv
            )

        # ============================================================
        # Merge attention heads
        # ============================================================

        output = (
            output.transpose(
                1,
                2,
            )
            .contiguous()
            .reshape(
                bsz,
                seq_len,
                -1,
            )
        )

        # ============================================================
        # Output projection
        # ============================================================

        output = self.resid_dropout(
            self.o_proj(
                output
            )
        )

        return (
            output,
            past_kv,
        )


# ================================================================
# FeedForward
# ================================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        config: MiniMindConfig,
        intermediate_size=None,
    ):
        super().__init__()

        intermediate_size = (
            intermediate_size
            or config.intermediate_size
        )

        self.gate_proj = nn.Linear(
            config.hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            intermediate_size,
            bias=False,
        )

        self.act_fn = ACT2FN[
            config.hidden_act
        ]

    def forward(self, x):

        return self.down_proj(
            self.act_fn(
                self.gate_proj(x)
            )
            * self.up_proj(x)
        )


# ================================================================
# MoE
# ================================================================

class MOEFeedForward(nn.Module):

    def __init__(
        self,
        config: MiniMindConfig,
    ):
        super().__init__()

        self.config = config

        self.gate = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
        )

        self.experts = nn.ModuleList(
            [
                FeedForward(
                    config,
                    intermediate_size=(
                        config.moe_intermediate_size
                    ),
                )
                for _ in range(
                    config.num_experts
                )
            ]
        )

        self.act_fn = ACT2FN[
            config.hidden_act
        ]

        self.aux_loss = torch.tensor(
            0.0
        )

    def forward(self, x):

        batch_size, seq_len, hidden_dim = (
            x.shape
        )

        x_flat = x.reshape(
            -1,
            hidden_dim,
        )

        scores = F.softmax(
            self.gate(x_flat),
            dim=-1,
        )

        topk_weight, topk_idx = torch.topk(
            scores,
            k=self.config.num_experts_per_tok,
            dim=-1,
            sorted=False,
        )

        if self.config.norm_topk_prob:

            if (
                self.config.num_experts_per_tok
                > 1
            ):

                topk_weight = (
                    topk_weight
                    / (
                        topk_weight.sum(
                            dim=-1,
                            keepdim=True,
                        )
                        + 1e-20
                    )
                )

            else:

                top1 = torch.topk(
                    F.softmax(
                        self.gate(
                            x_flat.detach()
                        ),
                        dim=-1,
                    ),
                    k=1,
                    dim=-1,
                    sorted=False,
                )[0]

                topk_weight = (
                    top1
                    - top1.detach()
                    + 1.0
                )

        y = torch.zeros_like(
            x_flat
        )

        for i, expert in enumerate(
            self.experts
        ):

            mask = (
                topk_idx == i
            )

            if mask.any():

                token_idx = (
                    mask.any(
                        dim=-1
                    )
                    .nonzero(
                        as_tuple=False
                    )
                    .flatten()
                )

                weight = (
                    topk_weight[mask]
                    .view(-1, 1)
                )

                expert_output = expert(
                    x_flat[
                        token_idx
                    ]
                )

                y.index_add_(
                    0,
                    token_idx,
                    (
                        expert_output
                        * weight
                    ).to(y.dtype),
                )

            elif self.training:

                y[0, 0] += (
                    0
                    * sum(
                        p.sum()
                        for p in expert.parameters()
                    )
                )

        if (
            self.training
            and self.config.router_aux_loss_coef
            > 0
        ):

            load = (
                F.one_hot(
                    topk_idx,
                    self.config.num_experts,
                )
                .float()
                .mean(0)
            )

            self.aux_loss = (
                (
                    load
                    * scores.mean(0)
                ).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )

        else:

            self.aux_loss = scores.new_zeros(
                1
            ).squeeze()

        return y.view(
            batch_size,
            seq_len,
            hidden_dim,
        )


# ================================================================
# MiniMind Block
#
# KDA V3
# +
# Block Attention Residual
# +
# MLP
# ================================================================

class MiniMindBlock(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: MiniMindConfig,
    ):
        super().__init__()

        self.layer_id = layer_id

        # ============================================================
        # Attention Backend
        # ============================================================

        if config.use_kda:

            self.self_attn = KDAAttentionV3(
                config
            )

        else:

            self.self_attn = Attention(
                config
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MOEFeedForward(config)
        )

        # ============================================================
        # Block Attention Residual
        # ============================================================

        if config.use_attn_res:

            self.attn_residual = (
                BlockAttentionResidualWrapper(
                    hidden_size=config.hidden_size,
                    max_blocks=(
                        config.attn_res_max_blocks
                    ),
                    dropout=(
                        config.attn_res_dropout
                    ),
                )
            )

        else:

            self.attn_residual = None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
        previous_block_states=None,
    ):

        # ============================================================
        # Attention
        # ============================================================

        residual = hidden_states

        (
            hidden_states,
            present_key_value,
        ) = self.self_attn(
            self.input_layernorm(
                hidden_states
            ),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = (
            hidden_states
            + residual
        )

        # ============================================================
        # Block Attention Residual
        # ============================================================

        attn_res_weights = None

        if (
            self.attn_residual is not None
            and previous_block_states is not None
            and len(previous_block_states) > 0
        ):

            (
                hidden_states,
                attn_res_weights,
            ) = self.attn_residual(
                hidden_states,
                previous_block_states,
            )

        # ============================================================
        # MLP
        # ============================================================

        hidden_states = (
            hidden_states
            + self.mlp(
                self.post_attention_layernorm(
                    hidden_states
                )
            )
        )

        return (
            hidden_states,
            present_key_value,
            attn_res_weights,
        )


# ================================================================
# MiniMind Model
# ================================================================

class MiniMindModel(nn.Module):

    def __init__(
        self,
        config: MiniMindConfig,
    ):
        super().__init__()

        self.config = config

        self.vocab_size = (
            config.vocab_size
        )

        self.num_hidden_layers = (
            config.num_hidden_layers
        )

        # ============================================================
        # Block Attention Residual
        # ============================================================

        self.attn_res_block_size = max(
            int(
                config.attn_res_block_size
            ),
            1,
        )

        self.attn_res_num_blocks = math.ceil(
            self.num_hidden_layers
            / self.attn_res_block_size
        )

        # ============================================================
        # Embedding
        # ============================================================

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        # ============================================================
        # Transformer layers
        # ============================================================

        self.layers = nn.ModuleList(
            [
                MiniMindBlock(
                    layer_id=i,
                    config=config,
                )
                for i in range(
                    self.num_hidden_layers
                )
            ]
        )

        # ============================================================
        # Final norm
        # ============================================================

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # ============================================================
        # RoPE
        # ============================================================

        freqs_cos, freqs_sin = (
            precompute_freqs_cis(
                dim=config.head_dim,
                end=config.max_position_embeddings,
                rope_base=config.rope_theta,
                rope_scaling=config.rope_scaling,
            )
        )

        self.register_buffer(
            "freqs_cos",
            freqs_cos,
            persistent=False,
        )

        self.register_buffer(
            "freqs_sin",
            freqs_sin,
            persistent=False,
        )

    def _get_past_length(
        self,
        past_key_values,
    ):
        """
        Get the cached sequence length.

        IMPORTANT:

        Standard MiniMind Attention cache layout is:

            [B, T, KV_heads, head_dim]

        Therefore:

            sequence_length = shape[1]

        NOT shape[2].
        """

        if past_key_values is None:
            return 0

        if hasattr(
            past_key_values,
            "layers",
        ):
            return 0

        for past_layer in past_key_values:

            if past_layer is None:
                continue

            if (
                isinstance(
                    past_layer,
                    (tuple, list),
                )
                and len(past_layer) == 2
                and torch.is_tensor(
                    past_layer[0]
                )
                and torch.is_tensor(
                    past_layer[1]
                )
            ):

                cache_key = (
                    past_layer[0]
                )

                if cache_key.ndim != 4:
                    continue

                # ----------------------------------------------------
                # Standard Attention:
                #
                # [B, T, KV_heads, D]
                #
                # KDA:
                #
                # its internal cache is handled by KDA itself.
                #
                # We only use this value for Standard Attention.
                # ----------------------------------------------------

                return int(
                    cache_key.shape[1]
                )

        return 0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):

        batch_size, seq_length = (
            input_ids.shape
        )

        # ============================================================
        # DynamicCache compatibility
        # ============================================================

        if hasattr(
            past_key_values,
            "layers",
        ):

            past_key_values = None

        # ============================================================
        # Normalize cache
        # ============================================================

        if past_key_values is None:

            past_key_values = [
                None
                for _ in self.layers
            ]

        else:

            past_key_values = list(
                past_key_values
            )

            if len(
                past_key_values
            ) != len(
                self.layers
            ):

                raise ValueError(
                    "Number of cache states "
                    f"({len(past_key_values)}) does not match "
                    f"number of layers ({len(self.layers)})."
                )

        # ============================================================
        # Embedding
        # ============================================================

        hidden_states = self.dropout(
            self.embed_tokens(
                input_ids
            )
        )

        # ============================================================
        # Device-safe RoPE
        # ============================================================

        if (
            self.freqs_cos.device
            != hidden_states.device
        ):

            self.freqs_cos = (
                self.freqs_cos.to(
                    hidden_states.device
                )
            )

            self.freqs_sin = (
                self.freqs_sin.to(
                    hidden_states.device
                )
            )

        # ============================================================
        # IMPORTANT KV-CACHE ROPE FIX
        #
        # Prefill:
        #
        #     past_length = 0
        #
        # Decode:
        #
        #     past_length = cached sequence length
        #
        # Example:
        #
        # context = 128
        #
        # cached decode token:
        #
        #     position = 128
        #
        # NOT:
        #
        #     position = 0
        # ============================================================

        past_length = 0

        # ------------------------------------------------------------
        # Only Standard Attention uses these RoPE positions.
        #
        # KDA V3 accepts position_embeddings for interface
        # compatibility but its recurrence does not use the RoPE
        # tensors internally.
        # ------------------------------------------------------------

        if (
            not self.config.use_kda
            and past_key_values is not None
        ):

            past_length = (
                self._get_past_length(
                    past_key_values
                )
            )

        # ============================================================
        # Position boundary check
        # ============================================================

        if (
            past_length
            + seq_length
            >
            self.freqs_cos.shape[0]
        ):

            raise ValueError(
                "RoPE position exceeds "
                "max_position_embeddings: "
                f"past_length={past_length}, "
                f"seq_length={seq_length}, "
                f"max={self.freqs_cos.shape[0]}"
            )

        # ============================================================
        # Correct absolute RoPE positions
        # ============================================================

        position_embeddings = (
            self.freqs_cos[
                past_length:
                past_length + seq_length
            ],
            self.freqs_sin[
                past_length:
                past_length + seq_length
            ],
        )

        # ============================================================
        # Cache outputs
        # ============================================================

        presents = []

        # ============================================================
        # Block Attention Residual state
        # ============================================================

        previous_block_states = []

        current_block_outputs = []

        attn_res_weights = []

        # ============================================================
        # Transformer layers
        # ============================================================

        for layer_idx, (
            layer,
            past_key_value,
        ) in enumerate(
            zip(
                self.layers,
                past_key_values,
            )
        ):

            # --------------------------------------------------------
            # Completed previous blocks
            # --------------------------------------------------------

            if self.config.use_attn_res:

                block_states_for_layer = (
                    previous_block_states
                )

            else:

                block_states_for_layer = None

            # --------------------------------------------------------
            # Layer forward
            # --------------------------------------------------------

            (
                hidden_states,
                present,
                layer_attn_res,
            ) = layer(
                hidden_states,
                position_embeddings,
                past_key_value=(
                    past_key_value
                ),
                use_cache=use_cache,
                attention_mask=attention_mask,
                previous_block_states=(
                    block_states_for_layer
                ),
            )

            # --------------------------------------------------------
            # Save cache
            # --------------------------------------------------------

            presents.append(
                present
            )

            # --------------------------------------------------------
            # Attention residual weights
            # --------------------------------------------------------

            if layer_attn_res is not None:

                attn_res_weights.append(
                    layer_attn_res
                )

            # --------------------------------------------------------
            # Save current block output
            # --------------------------------------------------------

            current_block_outputs.append(
                hidden_states
            )

            # --------------------------------------------------------
            # Block boundary
            # --------------------------------------------------------

            is_block_end = (
                (
                    layer_idx + 1
                )
                % self.attn_res_block_size
                == 0
            )

            is_last_layer = (
                layer_idx
                == self.num_hidden_layers - 1
            )

            if (
                is_block_end
                or is_last_layer
            ):

                block_summary = (
                    current_block_outputs[-1]
                )

                previous_block_states.append(
                    block_summary
                )

                current_block_outputs = []

        # ============================================================
        # Final normalization
        # ============================================================

        hidden_states = self.norm(
            hidden_states
        )

        # ============================================================
        # MoE auxiliary loss
        # ============================================================

        aux_loss = hidden_states.new_zeros(
            ()
        )

        for layer in self.layers:

            if isinstance(
                layer.mlp,
                MOEFeedForward,
            ):

                aux_loss = (
                    aux_loss
                    + layer.mlp.aux_loss
                )

        # ============================================================
        # Cache API
        # ============================================================

        if not use_cache:

            presents = None

        # ============================================================
        # Return
        # ============================================================

        return (
            hidden_states,
            presents,
            aux_loss,
            attn_res_weights,
        )


# ================================================================
# MiniMind Causal LM
# ================================================================

class MiniMindForCausalLM(
    PreTrainedModel,
    GenerationMixin,
):

    config_class = MiniMindConfig

    _tied_weights_keys = {
        "lm_head.weight":
            "model.embed_tokens.weight"
    }

    def __init__(
        self,
        config: MiniMindConfig = None,
    ):

        self.config = (
            config
            or MiniMindConfig()
        )

        super().__init__(
            self.config
        )

        self.model = MiniMindModel(
            self.config
        )

        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )

        # ============================================================
        # Weight tying
        # ============================================================

        if self.config.tie_word_embeddings:

            self.model.embed_tokens.weight = (
                self.lm_head.weight
            )

        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        **kwargs,
    ):

        (
            hidden_states,
            present_key_values,
            aux_loss,
            attn_res_weights,
        ) = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        # ============================================================
        # logits_to_keep
        # ============================================================

        if isinstance(
            logits_to_keep,
            int,
        ):

            if logits_to_keep > 0:

                slice_indices = slice(
                    -logits_to_keep,
                    None,
                )

            else:

                slice_indices = slice(
                    None
                )

        else:

            slice_indices = (
                logits_to_keep
            )

        logits = self.lm_head(
            hidden_states[
                :,
                slice_indices,
                :,
            ]
        )

        # ============================================================
        # Causal LM loss
        # ============================================================

        loss = None

        if labels is not None:

            if logits.size(1) > 1:

                shift_logits = logits[
                    ...,
                    :-1,
                    :,
                ].contiguous()

                shift_labels = labels[
                    ...,
                    1:,
                ].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(
                        -1,
                        shift_logits.size(-1),
                    ),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        # ============================================================
        # HuggingFace output
        # ============================================================

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=(
                present_key_values
            ),
            hidden_states=(
                hidden_states
            ),
            attentions=(
                attn_res_weights
            ),
        )

    # ================================================================
    # Custom autoregressive generation
    # ================================================================

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        **kwargs,
    ):

        # ============================================================
        # Input
        # ============================================================

        input_ids = kwargs.pop(
            "input_ids",
            inputs,
        )

        if input_ids is None:

            raise ValueError(
                "generate() requires "
                "`inputs` or `input_ids`."
            )

        if input_ids.dim() != 2:

            raise ValueError(
                "`input_ids` must have shape "
                "[batch, sequence_length]."
            )

        # ============================================================
        # Repeat input
        # ============================================================

        if num_return_sequences > 1:

            input_ids = input_ids.repeat_interleave(
                num_return_sequences,
                dim=0,
            )

            if attention_mask is not None:

                attention_mask = (
                    attention_mask.repeat_interleave(
                        num_return_sequences,
                        dim=0,
                    )
                )

        # ============================================================
        # Existing cache
        # ============================================================

        past_key_values = kwargs.pop(
            "past_key_values",
            None,
        )

        prompt_processed = (
            past_key_values is not None
        )

        # ============================================================
        # Finished sequences
        # ============================================================

        finished = torch.zeros(
            input_ids.shape[0],
            dtype=torch.bool,
            device=input_ids.device,
        )

        # ============================================================
        # Stream prompt
        # ============================================================

        if streamer is not None:

            streamer.put(
                input_ids.cpu()
            )

        # ============================================================
        # Generation loop
        # ============================================================

        for step in range(
            max_new_tokens
        ):

            # --------------------------------------------------------
            # Prefill
            # --------------------------------------------------------

            if not prompt_processed:

                model_input_ids = (
                    input_ids
                )

            # --------------------------------------------------------
            # Cached decode
            # --------------------------------------------------------

            else:

                model_input_ids = (
                    input_ids[:, -1:]
                )

            # --------------------------------------------------------
            # Forward
            # --------------------------------------------------------

            outputs = self.forward(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                logits_to_keep=1,
                **kwargs,
            )

            # --------------------------------------------------------
            # Update cache
            # --------------------------------------------------------

            if use_cache:

                past_key_values = (
                    outputs.past_key_values
                )

            else:

                past_key_values = None

            prompt_processed = True

            # --------------------------------------------------------
            # Update attention mask
            # --------------------------------------------------------

            if attention_mask is not None:

                attention_mask = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(
                            (
                                attention_mask.shape[
                                    0
                                ],
                                1,
                            )
                        ),
                    ],
                    dim=-1,
                )

            # --------------------------------------------------------
            # Last-token logits
            # --------------------------------------------------------

            logits = (
                outputs.logits[:, -1, :]
            )

            # --------------------------------------------------------
            # Temperature
            # --------------------------------------------------------

            if temperature is None:

                temperature = 1.0

            if temperature <= 0:

                temperature = 1.0

            logits = (
                logits
                / temperature
            )

            # --------------------------------------------------------
            # Repetition penalty
            # --------------------------------------------------------

            if (
                repetition_penalty is not None
                and repetition_penalty != 1.0
            ):

                for i in range(
                    input_ids.shape[0]
                ):

                    seen = torch.unique(
                        input_ids[i]
                    )

                    score = logits[
                        i,
                        seen,
                    ]

                    logits[
                        i,
                        seen,
                    ] = torch.where(
                        score > 0,
                        score
                        / repetition_penalty,
                        score
                        * repetition_penalty,
                    )

            # --------------------------------------------------------
            # Top-k
            # --------------------------------------------------------

            if (
                top_k is not None
                and top_k > 0
            ):

                k = min(
                    int(top_k),
                    logits.shape[-1],
                )

                kth = torch.topk(
                    logits,
                    k,
                    dim=-1,
                ).values[
                    ...,
                    -1,
                    None,
                ]

                logits = torch.where(
                    logits < kth,
                    torch.full_like(
                        logits,
                        -float("inf"),
                    ),
                    logits,
                )

            # --------------------------------------------------------
            # Top-p
            # --------------------------------------------------------

            if (
                top_p is not None
                and 0.0 < top_p < 1.0
            ):

                sorted_logits, sorted_indices = (
                    torch.sort(
                        logits,
                        descending=True,
                        dim=-1,
                    )
                )

                sorted_probs = torch.softmax(
                    sorted_logits,
                    dim=-1,
                )

                cumulative_probs = (
                    torch.cumsum(
                        sorted_probs,
                        dim=-1,
                    )
                )

                sorted_mask = (
                    cumulative_probs
                    > top_p
                )

                sorted_mask[
                    ...,
                    1:,
                ] = sorted_mask[
                    ...,
                    :-1,
                ].clone()

                sorted_mask[
                    ...,
                    0,
                ] = False

                remove_mask = torch.zeros_like(
                    sorted_mask
                )

                remove_mask.scatter_(
                    1,
                    sorted_indices,
                    sorted_mask,
                )

                logits = logits.masked_fill(
                    remove_mask,
                    -float("inf"),
                )

            # --------------------------------------------------------
            # Sampling / greedy
            # --------------------------------------------------------

            if do_sample:

                probs = torch.softmax(
                    logits,
                    dim=-1,
                )

                probs = torch.nan_to_num(
                    probs,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                probs_sum = probs.sum(
                    dim=-1,
                    keepdim=True,
                )

                invalid_rows = (
                    probs_sum <= 0
                )

                if invalid_rows.any():

                    probs = torch.where(
                        invalid_rows,
                        torch.softmax(
                            outputs.logits[
                                :,
                                -1,
                                :,
                            ],
                            dim=-1,
                        ),
                        probs,
                    )

                probs = (
                    probs
                    / probs.sum(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(
                        1e-12
                    )
                )

                next_token = (
                    torch.multinomial(
                        probs,
                        num_samples=1,
                    )
                )

            else:

                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )

            # --------------------------------------------------------
            # Finished sequences
            # --------------------------------------------------------

            if eos_token_id is not None:

                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full(
                        (
                            next_token.shape[
                                0
                            ],
                            1,
                        ),
                        eos_token_id,
                    ),
                    next_token,
                )

            # --------------------------------------------------------
            # Append
            # --------------------------------------------------------

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token,
                ],
                dim=-1,
            )

            # --------------------------------------------------------
            # Stream
            # --------------------------------------------------------

            if streamer is not None:

                streamer.put(
                    next_token.cpu()
                )

            # --------------------------------------------------------
            # EOS
            # --------------------------------------------------------

            if eos_token_id is not None:

                finished |= (
                    next_token
                    .squeeze(-1)
                    .eq(eos_token_id)
                )

                if finished.all():

                    break

        # ============================================================
        # End streamer
        # ============================================================

        if streamer is not None:

            streamer.end()

        # ============================================================
        # Optional cache return
        # ============================================================

        if kwargs.get(
            "return_kv",
            False,
        ):

            return {
                "generated_ids": input_ids,
                "past_kv": past_key_values,
            }

        return input_ids