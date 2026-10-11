
"""
train.py - Training Script for MyOwnLLM
=======================================
Trains the decoder-only Transformer from scratch using PyTorch.

Key Features:
- Designed to run on a standard laptop CPU.
- Uses AdamW optimizer with weight decay.
- Evaluates train and validation loss at regular intervals.
- Saves model weights and hyperparameters to 'my_own_llm.pt'.
- Saves vocabulary to 'vocab.json'.
- Handles Ctrl+C interruption.
- Reports completion only when training finishes successfully.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn

from tokenizer import CharTokenizer
from model import MyOwnLLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MyOwnLLM on your Windows laptop."
    )
    parser.add_argument(
        "--data_path", type=str, default="dataset.txt",
        help="Path to text dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--context_length", type=int, default=64,
        help="Maximum sequence context length (default: 64)"
    )
    parser.add_argument(
        "--d_model", type=int, default=128,
        help="Embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--n_heads", type=int, default=4,
        help="Number of attention heads (default: 4)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=3,
        help="Number of Transformer blocks (default: 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for AdamW (default: 0.001)"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="my_own_llm.pt",
        help="Where to save model weights"
    )
    parser.add_argument(
        "--vocab_path", type=str, default="vocab.json",
        help="Where to save vocabulary mapping"
    )
    return parser.parse_args()


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
):
    """
    Samples random chunks of length context_length from data.
    Input x contains tokens to read.
    Target y contains the next token to predict.
    """
    max_idx = len(data) - context_length - 1

    if max_idx <= 0:
        raise ValueError(
            f"Dataset is too short ({len(data)} tokens) "
            f"for context length {context_length}."
        )

    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack(
        [data[i.item(): i.item() + context_length] for i in ix]
    )
    y = torch.stack(
        [
            data[i.item() + 1: i.item() + context_length + 1]
            for i in ix
        ]
    )

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: MyOwnLLM,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_iters: int = 15,
):
    """Computes average loss on training and validation subsets."""
    model.eval()
    losses = {}

    for split, data_split in [
        ("train", train_data),
        ("val", val_data),
    ]:
        total_loss = 0.0

        for _ in range(eval_iters):
            x, y = get_batch(
                data_split, batch_size, context_length, device
            )
            _, loss = model(x, y)
            total_loss += loss.item()

        losses[split] = total_loss / eval_iters

    model.train()
    return losses


def main():
    args = parse_args()

    print("=" * 65)
    print("        MyOwnLLM: Decoder-Only Transformer Training")
    print("=" * 65)

    # 1. Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"[Device] CUDA GPU detected: "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        device = torch.device("cpu")
        print(
            "[Device] Running on CPU "
            "(configured for laptop execution)."
        )

    # 2. Check dataset file
    if not os.path.exists(args.data_path):
        print(
            f"[ERROR] Dataset file '{args.data_path}' was not found!",
            file=sys.stderr,
        )
        print(
            "Please ensure 'dataset.txt' is in the current directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[Data] Reading text dataset from '{args.data_path}'...")

    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    if len(text) < 200:
        print(
            "[ERROR] Dataset is too small. "
            "Please provide at least 200 characters.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[Data] Loaded {len(text):,} characters.")

    # 3. Fit Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    tokenizer.save(args.vocab_path)

    token_ids = tokenizer.encode(text)
    full_tensor = torch.tensor(token_ids, dtype=torch.long)

    # 4. Train / Validation split
    split_idx = int(0.9 * len(full_tensor))
    train_data = full_tensor[:split_idx]
    val_data = full_tensor[split_idx:]

    print(
        f"[Data] Split into {len(train_data):,} training and "
        f"{len(val_data):,} validation tokens."
    )

    # Validate dataset lengths before training
    if (
        len(train_data) <= args.context_length + 1
        or len(val_data) <= args.context_length + 1
    ):
        print(
            "[ERROR] Training or validation data is too short "
            "for the selected context length. Use a larger dataset "
            "or a smaller --context_length.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 5. Initialize Model
    print("\n[Model] Initializing Transformer architecture...")

    model = MyOwnLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_length=args.context_length,
        dropout=0.1,
    ).to(device)

    total_params = model.count_parameters()
    print(f"[Model] Total Trainable Parameters: {total_params:,}")
    print(
        f"[Model] Dimensions: d_model={args.d_model}, "
        f"heads={args.n_heads}, layers={args.n_layers}, "
        f"context={args.context_length}"
    )

    # 6. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # 7. Training loop setup
    steps_per_epoch = max(
        10,
        len(train_data) // (args.batch_size * args.context_length),
    )
    total_steps = args.epochs * steps_per_epoch

    print(
        f"\n[Training] Starting {args.epochs} epochs "
        f"(~{total_steps} total gradient steps)..."
    )
    print("-" * 65)
    print(
        f"{'Epoch':<8} {'Step':<8} {'Train Loss':<14} "
        f"{'Val Loss':<14} {'Time Elapsed'}"
    )
    print("-" * 65)

    start_time = time.time()
    best_val_loss = float("inf")
    completed_successfully = False

    try:
        step = 0

        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0

            for _ in range(steps_per_epoch):
                step += 1
                model.train()

                xb, yb = get_batch(
                    train_data,
                    args.batch_size,
                    args.context_length,
                    device,
                )

                _, loss = model(xb, yb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()

                epoch_loss += loss.item()

            # Evaluate after each epoch
            eval_losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.context_length,
                device,
            )

            elapsed = time.time() - start_time

            print(
                f"{epoch:<8} {step:<8} "
                f"{eval_losses['train']:<14.4f} "
                f"{eval_losses['val']:<14.4f} "
                f"{elapsed:.1f}s"
            )

            # Save the best model checkpoint
            if eval_losses["val"] < best_val_loss:
                best_val_loss = eval_losses["val"]

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": tokenizer.vocab_size,
                    "config": {
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "context_length": args.context_length,
                    },
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                }

                torch.save(checkpoint, args.checkpoint_path)

        completed_successfully = True

    except KeyboardInterrupt:
        print("\n[Notice] Training interrupted by user (Ctrl+C).")

        if os.path.exists(args.checkpoint_path):
            print(
                "The best checkpoint saved prior to interruption "
                f"is available at '{args.checkpoint_path}'."
            )
        sys.exit(0)

    # 8. Final Status Verification
    print("-" * 65)
    total_time = time.time() - start_time

    if completed_successfully:
        print(
            f"[STATUS: TRAINED] Training successfully completed "
            f"in {total_time:.1f} seconds!"
        )
        print(f"Model checkpoint saved to: '{args.checkpoint_path}'")
        print(f"Vocabulary saved to:       '{args.vocab_path}'")
        print(f"Best Validation Loss:      {best_val_loss:.4f}")
        print("\nYou can now test text generation with:")
        print('    python generate.py --prompt "Once upon a time"')
    else:
        print(
            "[STATUS: INCOMPLETE] Training did not finish all epochs.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
