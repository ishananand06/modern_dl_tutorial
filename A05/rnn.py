"""
rnn.py — Character-level RNN implemented from scratch using PyTorch.

You implement the Elman RNN cell manually inside forward().
nn.RNN, nn.LSTM, nn.GRU are NOT allowed anywhere in this file.
You may use: nn.Embedding, nn.Linear, torch operations, autograd.

Architecture:
    Embedding → manual RNN cell (loop over time) → Linear → logits

The RNN cell at each time step t:
    a_t = x_t @ W_xh.T + h_{t-1} @ W_hh.T + b_h
    h_t = tanh(a_t)

where x_t is the embedded input at step t (shape: batch × embed_dim).
The output layer maps each hidden state to vocabulary logits:
    logits_t = h_t @ W_hy.T + b_y
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """
    Character-level language model using a manually implemented RNN cell.

    The forward pass processes a full sequence chunk and returns logits
    for every time step. Cross-entropy loss is computed outside this class
    (in train.py) so that the same model can be used for generation.

    Args:
        vocab_size: number of unique characters (V)
        embed_dim:  character embedding dimension
        hidden_dim: RNN hidden state dimension (m)
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

        # Input embedding: integer index → dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # TODO: define W_xh, W_hh, b_h as nn.Parameter
        # and fc as nn.Linear (hidden → vocab). Use xavier init for weights.
        raise NotImplementedError

    def forward(
        self,
        x:    torch.Tensor,
        h:    torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over a sequence chunk.

        Args:
            x: integer tensor of shape (batch, seq_len) — character indices
            h: initial hidden state, shape (batch, hidden_dim).
               If None, initialise to zeros.

        Returns:
            logits: float tensor of shape (batch, seq_len, vocab_size)
            h:      final hidden state, shape (batch, hidden_dim)
                    — detach before passing to the next chunk (truncated BPTT)
        """
        # TODO: implement forward pass.
        #   - Embed x → (batch, seq_len, embed_dim)
        #   - Loop over t: compute a_t, h_t = tanh(a_t), collect h
        #   - Apply fc to stacked hidden states → logits
        # Do NOT use nn.RNN. The time-step loop is mandatory.

        raise NotImplementedError

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero hidden state of the correct shape."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)