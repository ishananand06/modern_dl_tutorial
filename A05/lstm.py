"""
lstm.py — Character-level LSTM implemented from scratch using PyTorch.

You implement all four LSTM gates manually inside forward().
nn.RNN, nn.LSTM, nn.GRU are NOT allowed anywhere in this file.
You may use: nn.Embedding, nn.Linear, torch operations, autograd.

Architecture:
    Embedding → manual LSTM cell (loop over time) → Linear → logits

LSTM cell equations at each time step t:
    Combined input: z_t = x_t @ W_x.T + h_{t-1} @ W_h.T + b

    where W_x ∈ R^{4H × E} and W_h ∈ R^{4H × H} stack all four gate
    weight matrices into a single operation for efficiency:

        [f_pre, i_pre, g_pre, o_pre] = z_t.chunk(4, dim=-1)

    Forget gate:     f_t = sigmoid(f_pre)
    Input gate:      i_t = sigmoid(i_pre)
    Candidate mem:   g_t = tanh(g_pre)
    Output gate:     o_t = sigmoid(o_pre)
    Cell state:      c_t = f_t * c_{t-1} + i_t * g_t
    Hidden state:    h_t = o_t * tanh(c_t)

Using a single fused linear (4H × E and 4H × H) is the standard PyTorch
pattern — it is one matrix multiply instead of four, and it means you only
define two weight matrices (W_x, W_h) and one bias (b) rather than twelve.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """
    Character-level language model using a manually implemented LSTM cell.

    The LSTM has two state vectors:
        h_t  — hidden state (short-term memory), shape (batch, hidden_dim)
        c_t  — cell state  (long-term memory),  shape (batch, hidden_dim)

    Both are passed between chunks during truncated BPTT. Both must be
    detached before the next chunk to cut the gradient graph.

    Args:
        vocab_size: number of unique characters (V)
        embed_dim:  character embedding dimension
        hidden_dim: LSTM hidden/cell state dimension (H)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Fused LSTM cell weights — do NOT use nn.LSTM
        # W_x maps the input embedding to all four gate pre-activations at once
        # W_h maps the previous hidden state to all four gate pre-activations
        # b   is the shared bias for all four gates
        # TODO: define fused LSTM gate weights (W_x, W_h, b) and output
        # projection fc. Use xavier init for weights. Initialise forget-gate
        # bias to 1.0 (first hidden_dim entries of b).
        raise NotImplementedError

    def forward(
        self,
        x:  torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass over a sequence chunk.

        Args:
            x:  integer tensor of shape (batch, seq_len) — character indices
            hc: tuple (h, c) of initial hidden and cell states,
                each shape (batch, hidden_dim).
                If None, both are initialised to zeros.

        Returns:
            logits: float tensor of shape (batch, seq_len, vocab_size)
            (h, c): final hidden and cell states, shape (batch, hidden_dim) each
                    — detach both before the next chunk
        """
        # TODO: implement forward pass.
        #   - Embed x → (batch, seq_len, embed_dim)
        #   - Loop over t: compute fused gates, update c and h, collect h
        #   - Apply fc to stacked hidden states → logits
        # Do NOT use nn.LSTM. The time-step loop is mandatory.

        raise NotImplementedError

    def init_hidden(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero hidden and cell states of the correct shape."""
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
        return zeros, zeros.clone()