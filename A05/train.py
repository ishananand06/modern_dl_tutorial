"""
train.py — Train CharRNN or CharLSTM on tinyshakespeare.

Usage:
    python train.py --model rnn  --epochs 10 --hidden_dim 256
    python train.py --model lstm --epochs 10 --hidden_dim 256

Saves best checkpoint (lowest val loss) and final checkpoint to
    outputs/checkpoints/{model}_best.pt
    outputs/checkpoints/{model}_last.pt

Saves training curves to
    outputs/plots/{model}_curves.png

After both models are trained, also saves
    outputs/plots/comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data  import get_dataloaders
from rnn   import CharRNN
from lstm  import CharLSTM
from utils import (
    get_device,
    set_seed,
    save_checkpoint,
    plot_loss_curves,
    plot_comparison,
    compute_perplexity,
)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    seed       = 42,
    epochs     = 10,
    hidden_dim = 256,
    embed_dim  = 64,
    seq_len    = 100,
    batch_size = 64,
    lr         = 3e-3,
    clip_norm  = 5.0,
)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    clip_norm:  float,
    is_lstm:    bool,
) -> float:
    """
    Run one full pass over the training DataLoader.

    Truncated BPTT: hidden state is carried across batches within one epoch
    but detached at every batch boundary to cut the gradient graph.

    Args:
        model:     CharRNN or CharLSTM
        loader:    training DataLoader
        criterion: nn.CrossEntropyLoss()
        optimizer: torch.optim.Adam
        device:    target device
        clip_norm: max gradient norm for clipping (torch.nn.utils.clip_grad_norm_)
        is_lstm:   True for CharLSTM (two state tensors), False for CharRNN (one)

    Returns:
        avg_loss: mean cross-entropy loss per character across the epoch
    """
    model.train()
    total_loss = 0.0
    total_chars = 0

    # Initialise hidden state at the start of the epoch
    batch_size = loader.batch_size
    hidden = model.init_hidden(batch_size, device)

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)   # (batch, seq_len)
        y_batch = y_batch.to(device)   # (batch, seq_len)

        # TODO: implement the training step
        #   - Zero gradients, forward pass, compute cross-entropy loss,
        #     backward, clip gradients, step optimizer.
        #   - Detach hidden state before the next batch (truncated BPTT).
        #   - Accumulate loss for epoch-level reporting.
        #   CrossEntropyLoss expects logits (N, V) and targets (N,)
        #   where N = batch_size * seq_len.

        raise NotImplementedError

    return total_loss / total_chars


# ---------------------------------------------------------------------------
# One validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    is_lstm:   bool,
) -> float:
    """
    Evaluate the model on a DataLoader (validation or test).
    No gradient computation. Hidden state not carried across batches.

    Returns:
        avg_loss: mean cross-entropy loss per character
    """
    model.eval()
    total_loss  = 0.0
    total_chars = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        batch_size = x_batch.size(0)
        hidden     = model.init_hidden(batch_size, device)

        if is_lstm:
            logits, _ = model(x_batch, hidden)
        else:
            logits, _ = model(x_batch, hidden)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y_batch.view(-1),
        )
        total_loss  += loss.item() * y_batch.numel()
        total_chars += y_batch.numel()

    return total_loss / total_chars


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device  = get_device()
    is_lstm = args.model == "lstm"

    print(f"\n{'='*60}")
    print(f"Training {'CharLSTM' if is_lstm else 'CharRNN'}")
    print(f"  device     : {device}")
    print(f"  hidden_dim : {args.hidden_dim}")
    print(f"  embed_dim  : {args.embed_dim}")
    print(f"  seq_len    : {args.seq_len}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  epochs     : {args.epochs}")
    print(f"  lr         : {args.lr}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, _, vocab = get_dataloaders(
        seq_len    = args.seq_len,
        batch_size = args.batch_size,
    )
    print(f"Vocabulary size: {vocab.size}")

    # Model
    ModelClass = CharLSTM if is_lstm else CharRNN
    model = ModelClass(
        vocab_size = vocab.size,
        embed_dim  = args.embed_dim,
        hidden_dim = args.hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters     : {n_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Paths
    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{args.model}_best.pt"
    last_path = ckpt_dir / f"{args.model}_last.pt"

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.clip_norm, is_lstm,
        )
        val_loss = eval_epoch(model, val_loader, criterion, device, is_lstm)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f}  PPL {compute_perplexity(train_loss):6.2f} | "
            f"val loss {val_loss:.4f}  PPL {compute_perplexity(val_loss):6.2f}"
        )

        # Save best checkpoint (with full loss histories up to this epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, str(best_path),
                train_losses=train_losses, val_losses=val_losses,
            )

    # Save last checkpoint with complete histories
    save_checkpoint(
        model, optimizer, args.epochs, val_losses[-1], str(last_path),
        train_losses=train_losses, val_losses=val_losses,
    )

    # Save per-model plot
    plot_loss_curves(
        train_losses, val_losses,
        model_name = args.model.upper(),
        save_path  = f"outputs/plots/{args.model}_curves.png",
    )

    # Generate comparison.png if both model checkpoints exist with stored histories
    rnn_path  = ckpt_dir / "rnn_last.pt"
    lstm_path = ckpt_dir / "lstm_last.pt"
    if rnn_path.exists() and lstm_path.exists():
        rnn_ckpt  = torch.load(str(rnn_path),  map_location="cpu")
        lstm_ckpt = torch.load(str(lstm_path), map_location="cpu")
        rnn_val   = rnn_ckpt.get("val_losses",  [])
        lstm_val  = lstm_ckpt.get("val_losses", [])
        rnn_train = rnn_ckpt.get("train_losses",  [])
        lstm_train = lstm_ckpt.get("train_losses", [])
        if rnn_val and lstm_val:
            plot_comparison(
                rnn_train, rnn_val, lstm_train, lstm_val,
                save_path="outputs/plots/comparison.png",
            )
        else:
            print("Note: loss histories missing from checkpoints — comparison.png skipped.")

    print(f"\nDone. Best val PPL: {compute_perplexity(best_val_loss):.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CharRNN or CharLSTM on tinyshakespeare")
    p.add_argument("--model",      type=str,   default="rnn",
                   choices=["rnn", "lstm"],     help="which model to train")
    p.add_argument("--epochs",     type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--hidden_dim", type=int,   default=DEFAULTS["hidden_dim"])
    p.add_argument("--embed_dim",  type=int,   default=DEFAULTS["embed_dim"])
    p.add_argument("--seq_len",    type=int,   default=DEFAULTS["seq_len"])
    p.add_argument("--batch_size", type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",         type=float, default=DEFAULTS["lr"])
    p.add_argument("--clip_norm",  type=float, default=DEFAULTS["clip_norm"])
    p.add_argument("--seed",       type=int,   default=DEFAULTS["seed"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())