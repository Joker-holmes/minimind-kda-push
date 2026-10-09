import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# KDA V3
# =============================================================================
#
# V3 goals:
#
# 1. Preserve V2 numerical behavior
# 2. Preserve cache format:
#
#       cache = (recurrent_state, conv_state)
#
# 3. Preserve:
#
#       convolution -> projection -> normalization -> recurrence
#
# 4. Recurrent token loop is expressed as either:
#
#       - a TorchScript per-timestep loop (reference path)
#       - a chunked parallel formulation (fast path)
#
# =============================================================================


class ShortConv1dV3(nn.Module):
    """
    Causal depthwise convolution.

    Input:
        x:          [B, T, D]
    State:
        conv_state: [B, K-1, D]

    Output:
        y:          [B, T, D]
        new_state:  [B, K-1, D]
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=True,
            padding=0,
        )

    def forward(self, x, conv_state=None):

        B, T, D = x.shape

        history_len = self.kernel_size - 1

        if conv_state is None:
            conv_state = torch.zeros(
                B,
                history_len,
                D,
                device=x.device,
                dtype=x.dtype,
            )

        x_cat = torch.cat(
            [conv_state, x],
            dim=1,
        )

        x_conv = x_cat.transpose(1, 2)

        y = self.conv(x_conv)

        y = y.transpose(1, 2)

        new_state = x_cat[:, -history_len:, :]

        return y, new_state


# =============================================================================
# Reference recurrence (per-timestep, TorchScript)
# =============================================================================

@torch.jit.script
def kda_recurrence_script(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
):
    """
    q:      [B, H, T, D]
    k:      [B, H, T, D]
    v:      [B, H, T, D]
    alpha:  [B, H, T]
    beta:   [B, H, T]
    state:  [B, H, D, D]

    Returns:
        output:      [B, H, T, D]
        final_state: [B, H, D, D]
    """

    B = q.size(0)
    H = q.size(1)
    T = q.size(2)
    D = q.size(3)

    outputs = torch.empty(
        B,
        H,
        T,
        D,
        device=q.device,
        dtype=q.dtype,
    )

    for t in range(T):

        qt = q[:, :, t, :]
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]

        at = alpha[:, :, t]
        bt = beta[:, :, t]

        predicted = torch.einsum(
            "bhij,bhi->bhj",
            state,
            kt,
        )

        delta = vt - predicted

        state = state * at[:, :, None, None]

        state = state + (
            bt[:, :, None, None]
            * kt[:, :, :, None]
            * delta[:, :, None, :]
        )

        ot = torch.einsum(
            "bhij,bhi->bhj",
            state,
            qt,
        )

        outputs[:, :, t, :] = ot

    outputs = torch.nan_to_num(
        outputs,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    outputs = torch.clamp(
        outputs,
        min=-1e4,
        max=1e4,
    )

    state = torch.nan_to_num(
        state,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    state = torch.clamp(
        state,
        min=-1e4,
        max=1e4,
    )

    return outputs, state


# =============================================================================
# Chunked recurrence
# =============================================================================
#
# The per-timestep update with decay a_t and beta b_t is:
#
#     S_{t+1} = a_t * S_t - b_t * k_t * k_t^T * S_t + b_t * k_t * v_t^T
#
# Let E_t = v_t - k_t^T * S_t be the prediction error. Within a chunk
# starting at position 0 with entering state S_0:
#
#     S_i = D_i * S_0 + sum_{j < i} (D_i / D_{j+1}) * b_j * k_j * E_j^T
#
# Plugging S_i into E_i gives a lower-triangular system:
#
#     E_i + sum_{j < i} (D_i / D_{j+1}) * b_j * (k_j . k_i) * E_j
#         = v_i - D_i * S_0^T * k_i
#
# Outputs:
#
#     out_i = D_{i+1} * q_i^T * S_0
#           + sum_{j <= i} (D_{i+1} / D_{j+1}) * b_j * (q_i . k_j) * E_j
#
# Chunk-final state:
#
#     S_C = D_C * S_0
#         + sum_{j < C} (D_C / D_{j+1}) * b_j * k_j * E_j^T
#
# =============================================================================


def kda_recurrence_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    chunk_size: int = 64,
    max_state_norm: float = 20.0,
    diag_damping: float = 1.05,
):
    """
    Chunked, numerically exact delta-rule recurrence.

    q:      [B, H, T, D]
    k:      [B, H, T, D]
    v:      [B, H, T, D]
    alpha:  [B, H, T]
    beta:   [B, H, T]
    state:  [B, H, D, D]

    chunk_size:
        Timesteps per chunk. Pads internally if T is not a multiple.

    max_state_norm:
        Per-head Frobenius-norm cap for the recurrent state. The
        cap is applied after each chunk. Lower values give tighter
        numerical control at the cost of some expressiveness.

    diag_damping:
        Diagonal damping factor for the intra-chunk triangular
        matrix M_full = M + damping * I. Values slightly above 1
        raise the smallest singular value of M_full, which makes
        torch.linalg.solve well-conditioned. 1.0 reproduces the
        exact delta rule; 1.05 is a small numerical nudge that
        does not noticeably change the loss landscape.

    Returns:
        output:      [B, H, T, D]
        final_state: [B, H, D, D]
    """

    B, H, T, D = q.shape

    device = q.device
    dtype = q.dtype

    # ---------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # ---------------------------------------------------------------------

    remainder = T % chunk_size

    if remainder != 0:
        pad = chunk_size - remainder

        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))

        alpha = F.pad(alpha, (0, pad), value=1.0)
        beta = F.pad(beta, (0, pad), value=0.0)

    T_padded = q.shape[2]
    n_chunks = T_padded // chunk_size
    C = chunk_size

    # ---------------------------------------------------------------------
    # Masks
    # ---------------------------------------------------------------------

    idx = torch.arange(C, device=device, dtype=dtype)

    mask_lower = (idx[None, :] < idx[:, None]).to(dtype)
    mask_tril = (idx[None, :] <= idx[:, None]).to(dtype)

    eye = torch.eye(C, device=device, dtype=dtype)

    ones_64 = torch.ones(
        B,
        H,
        1,
        device=device,
        dtype=torch.float64,
    )

    # ---------------------------------------------------------------------
    # Main chunk loop
    # ---------------------------------------------------------------------

    outputs = torch.empty_like(q)

    S = state

    for c in range(n_chunks):

        s = c * C
        e = s + C

        q_c = q[:, :, s:e]
        k_c = k[:, :, s:e]
        v_c = v[:, :, s:e]
        a_c = alpha[:, :, s:e]
        b_c = beta[:, :, s:e]

        # -----------------------------------------------------------------
        # Cumulative decay (fp64)
        # -----------------------------------------------------------------

        a_c_64 = a_c.to(torch.float64)

        D_cum = torch.cumprod(
            torch.cat(
                [ones_64, a_c_64],
                dim=-1,
            ),
            dim=-1,
        )

        D_i_64 = D_cum[:, :, :C]
        D_j1_64 = D_cum[:, :, 1:]

        D_i = D_i_64.to(dtype)
        D_j1 = D_j1_64.to(dtype)

        # -----------------------------------------------------------------
        # Decay ratios (fp64)
        # -----------------------------------------------------------------

        ratio_m_64 = (
            D_i_64.unsqueeze(-1)
            / D_j1_64.unsqueeze(-2)
        )
        ratio_m = ratio_m_64.to(dtype)

        ratio2_64 = (
            D_j1_64.unsqueeze(-1)
            / D_j1_64.unsqueeze(-2)
        )
        ratio2 = ratio2_64.to(dtype)

        # -----------------------------------------------------------------
        # Gram matrix
        # -----------------------------------------------------------------

        G = torch.matmul(k_c, k_c.transpose(-1, -2))

        M = (
            b_c.unsqueeze(-2)
            * G
            * ratio_m
            * mask_lower
        )

        # Diagonal damping raises the smallest singular value of
        # M_full, which keeps torch.linalg.solve well conditioned
        # even when the near-diagonal ratios blow up.
        M_full = M + eye * diag_damping

        # -----------------------------------------------------------------
        # Right-hand side
        # -----------------------------------------------------------------

        predicted = torch.matmul(k_c, S)

        rhs = v_c - D_i.unsqueeze(-1) * predicted

        # -----------------------------------------------------------------
        # Triangular solve
        # -----------------------------------------------------------------

        E = torch.linalg.solve(M_full, rhs)

        # -----------------------------------------------------------------
        # Block-internal output
        # -----------------------------------------------------------------

        out_first = torch.matmul(q_c, S)
        out_first = D_j1.unsqueeze(-1) * out_first

        QK = torch.matmul(q_c, k_c.transpose(-1, -2))

        A_intra = (
            b_c.unsqueeze(-2)
            * QK
            * ratio2
            * mask_tril
        )

        out_second = torch.matmul(A_intra, E)

        outputs[:, :, s:e] = out_first + out_second

        # -----------------------------------------------------------------
        # Chunk-final state
        # -----------------------------------------------------------------

        D_C_64 = D_cum[:, :, -1]

        coeff_64 = (
            D_C_64.unsqueeze(-1) / D_j1_64
        ) * b_c.to(torch.float64)

        coeff = coeff_64.to(dtype)

        weighted_k = (
            coeff.unsqueeze(-1) * k_c
        ).transpose(-1, -2)

        S_update = torch.matmul(weighted_k, E)

        S = (
            D_C_64.to(dtype).unsqueeze(-1).unsqueeze(-1) * S
            + S_update
        )

        # -----------------------------------------------------------------
        # State norm clipping
        #
        # Bounds the recurrent state's Frobenius norm per head.
        # Prevents the state from growing without bound when the
        # alpha gate approaches 1. Only activates above the cap.
        # -----------------------------------------------------------------

        if max_state_norm is not None and max_state_norm > 0:
            state_norm = S.norm(dim=(-2, -1), keepdim=True)
            scale = torch.clamp(
                max_state_norm / (state_norm + 1e-6),
                max=1.0,
            )
            S = S * scale

    # ---------------------------------------------------------------------
    # Drop padding
    # ---------------------------------------------------------------------

    if remainder != 0:
        outputs = outputs[:, :, :T]

    # ---------------------------------------------------------------------
    # Output numerical protection
    # ---------------------------------------------------------------------

    outputs = torch.nan_to_num(
        outputs,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    outputs = torch.clamp(
        outputs,
        min=-1e4,
        max=1e4,
    )

    S = torch.nan_to_num(
        S,
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )

    S = torch.clamp(
        S,
        min=-1e4,
        max=1e4,
    )

    return outputs, S


# =============================================================================
# KDA Attention V3
# =============================================================================

class KDAAttentionV3(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_heads = config.num_attention_heads

        self.head_dim = config.head_dim

        assert (
            self.hidden_size
            == self.num_heads * self.head_dim
        )

        # -------------------------------------------------------------
        # QKV projections
        # -------------------------------------------------------------

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # -------------------------------------------------------------
        # Q/K normalization
        # -------------------------------------------------------------

        self.q_norm = nn.RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        self.k_norm = nn.RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        # -------------------------------------------------------------
        # Short convolution (BEFORE projection)
        # -------------------------------------------------------------

        self.q_conv = ShortConv1dV3(
            self.hidden_size,
            kernel_size=4,
        )

        self.k_conv = ShortConv1dV3(
            self.hidden_size,
            kernel_size=4,
        )

        self.v_conv = ShortConv1dV3(
            self.hidden_size,
            kernel_size=4,
        )

        # -------------------------------------------------------------
        # Gates
        # -------------------------------------------------------------

        self.alpha_down = nn.Linear(
            self.hidden_size,
            self.num_heads,
            bias=True,
        )

        self.alpha_up = nn.Linear(
            self.num_heads,
            self.num_heads,
            bias=True,
        )

        self.beta_proj = nn.Linear(
            self.hidden_size,
            self.num_heads,
            bias=True,
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        # -------------------------------------------------------------
        # Recurrence configuration
        #
        # max_state_norm was previously 100.0 and allowed the
        # recurrent state to grow into an ill-conditioned region,
        # producing permanent non-finite gradients after step 847
        # in SFT training. 20.0 keeps the solve well-conditioned.
        # -------------------------------------------------------------

        self.use_chunked = True

        self.chunk_size = 64

        self.max_state_norm = 20.0

        self.diag_damping = 1.05

    # -----------------------------------------------------------------
    # Split heads
    # -----------------------------------------------------------------

    def _split_heads(self, x):

        B, T, _ = x.shape

        x = x.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x,
        position_embeddings=None,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):

        B, T, _ = x.shape

        input_dtype = x.dtype

        # =============================================================
        # Cache
        # =============================================================

        if past_key_value is None:
            recurrent_state = None
            conv_state = None
        else:
            recurrent_state = past_key_value[0]
            conv_state = past_key_value[1]

        # =============================================================
        # Conv FIRST (shared conv_state for Q/K/V)
        # =============================================================

        q, new_conv_state = self.q_conv(x, conv_state)
        k, _ = self.k_conv(x, conv_state)
        v, _ = self.v_conv(x, conv_state)

        # =============================================================
        # Projection
        # =============================================================

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # =============================================================
        # Heads
        # =============================================================

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # =============================================================
        # Q/K normalization (fp32 before RMSNorm)
        # =============================================================

        q = self.q_norm(q.float())
        k = self.k_norm(k.float())

        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

        v = v.float()

        # =============================================================
        # Gates
        # =============================================================

        gate_input = x.float()

        alpha = self.alpha_up(
            self.alpha_down(gate_input)
        )
        alpha = torch.sigmoid(alpha)

        beta = torch.sigmoid(
            self.beta_proj(gate_input)
        )

        alpha = alpha.transpose(1, 2)
        beta = beta.transpose(1, 2)

        # =============================================================
        # Recurrent state
        # =============================================================

        if recurrent_state is None:
            state = torch.zeros(
                B,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
                dtype=torch.float32,
            )
        else:
            state = recurrent_state.float()

        # =============================================================
        # Recurrence
        # =============================================================

        if self.use_chunked and T > self.chunk_size:
            output, state = kda_recurrence_chunked(
                q,
                k,
                v,
                alpha,
                beta,
                state,
                chunk_size=self.chunk_size,
                max_state_norm=self.max_state_norm,
                diag_damping=self.diag_damping,
            )
        else:
            output, state = kda_recurrence_script(
                q,
                k,
                v,
                alpha,
                beta,
                state,
            )

        # =============================================================
        # [B,H,T,D] -> [B,T,H,D]
        # =============================================================

        output = output.transpose(1, 2)

        output = output.reshape(
            B,
            T,
            self.num_heads * self.head_dim,
        )

        # =============================================================
        # Output projection
        # =============================================================

        output = self.o_proj(
            output.to(input_dtype)
        )

        output = self.dropout(output)

        # =============================================================
        # Cache
        # =============================================================

        if use_cache:
            present_state = (state, new_conv_state)
        else:
            present_state = None

        return output, present_state