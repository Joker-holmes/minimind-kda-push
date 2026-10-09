import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockAttentionResidual(nn.Module):
    """
    Block Attention Residuals

    A practical block-level implementation of Attention Residuals.

    Instead of using only:

        h_l = h_{l-1} + residual

    we attend over previous block representations:

        h_l = attention(
            current_state,
            previous_block_states
        )

    Input:
        hidden_states:
            [B, T, D]

        previous_states:
            list of tensors
            each tensor has shape [B, T, D]

    Output:
        output:
            [B, T, D]

        attention_weights:
            [B, T, N]
            where N = number of previous block states
    """

    def __init__(
        self,
        hidden_size: int,
        max_blocks: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_blocks = max_blocks

        self.query = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.key = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.value = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

        self.scale = hidden_size ** -0.5

    def forward(
        self,
        hidden_states,
        previous_states,
    ):
        """
        Args:
            hidden_states:
                [B, T, D]

            previous_states:
                list[
                    [B, T, D]
                ]

        Returns:
            output:
                [B, T, D]

            attention_weights:
                [B, T, N]
        """

        if len(previous_states) == 0:
            return hidden_states, None

        if len(previous_states) > self.max_blocks:
            raise ValueError(
                f"Number of previous states "
                f"{len(previous_states)} exceeds "
                f"max_blocks={self.max_blocks}"
            )

        # --------------------------------------------------
        # 1. Query
        # --------------------------------------------------

        q = self.query(hidden_states)

        # [B, T, D]
        q = q * self.scale

        keys = []
        values = []

        # --------------------------------------------------
        # 2. Build block keys / values
        # --------------------------------------------------

        for state in previous_states:

            k = self.key(state)
            v = self.value(state)

            keys.append(k)
            values.append(v)

        # --------------------------------------------------
        # [B, T, N, D]
        # --------------------------------------------------

        k = torch.stack(keys, dim=2)

        v = torch.stack(values, dim=2)

        # --------------------------------------------------
        # 3. Attention score
        #
        # q: [B, T, D]
        # k: [B, T, N, D]
        #
        # score: [B, T, N]
        # --------------------------------------------------

        scores = torch.einsum(
            "btd,btnd->btn",
            q,
            k,
        )

        # --------------------------------------------------
        # 4. Softmax over blocks
        # --------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # --------------------------------------------------
        # 5. Weighted block representation
        #
        # weights: [B,T,N]
        # values:  [B,T,N,D]
        #
        # output:  [B,T,D]
        # --------------------------------------------------

        output = torch.einsum(
            "btn,btnd->btd",
            attention_weights,
            v,
        )

        output = self.out_proj(output)

        return output, attention_weights


class BlockAttentionResidualWrapper(nn.Module):
    """
    Convenience wrapper.

    Combines:

        current hidden state
        +
        Block Attention Residual

    """

    def __init__(
        self,
        hidden_size: int,
        max_blocks: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn_residual = BlockAttentionResidual(
            hidden_size=hidden_size,
            max_blocks=max_blocks,
            dropout=dropout,
        )

        self.norm = nn.RMSNorm(
            hidden_size,
        )

    def forward(
        self,
        hidden_states,
        previous_states,
    ):

        residual = hidden_states

        normalized = self.norm(
            hidden_states
        )

        attended, weights = self.attn_residual(
            normalized,
            previous_states,
        )

        output = residual + attended

        return output, weights