
"""
generate.py - Text Generation Script for MyOwnLLM
==================================================
Loads a trained model checkpoint and vocabulary, prompts the user (or uses CLI args),
and generates text autoregressively.

Error Handling:
- Verifies 'my_own_llm.pt' exists. If not, alerts user to run train.py first.
- Verifies 'vocab.json' exists.
- Safely handles unknown characters in prompts.
"""

import argparse
import os
import sys
import torch

from tokenizer import CharTokenizer
from model import MyOwnLLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using your trained MyOwnLLM model."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Starting prompt for generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Number of characters/tokens to generate (default: 150)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Sampling temperature (lower=more deterministic, higher=more creative, default: 0.75)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="Top-K token filter (default: 8, use 0 to disable)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="my_own_llm.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="vocab.json",
        help="Path to vocabulary file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("        MyOwnLLM: Autoregressive Text Generation")
    print("=" * 60)

    # 1. Error check: Checkpoint existence
    if not os.path.exists(args.checkpoint_path):
        print(
            f"[ERROR] Model checkpoint '{args.checkpoint_path}' not found!",
            file=sys.stderr,
        )
        print(
            "You must train the model first before generating text.",
            file=sys.stderr,
        )
        print(
            "\nRun this command to train the model:",
            file=sys.stderr,
        )
        print("    python train.py --epochs 20", file=sys.stderr)
        sys.exit(1)

    # 2. Error check: Vocabulary existence
    if not os.path.exists(args.vocab_path):
        print(
            f"[ERROR] Vocabulary file '{args.vocab_path}' not found!",
            file=sys.stderr,
        )
        print(
            "Please run training first to generate 'vocab.json'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Load Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.load(args.vocab_path)

    # 4. Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[Inference] Running on device: {device}")

    # 5. Load Model weights and architecture config
    print(
        f"[Inference] Loading checkpoint from '{args.checkpoint_path}'..."
    )
    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device,
    )
    config = checkpoint["config"]

    model = MyOwnLLM(
        vocab_size=checkpoint["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        context_length=config["context_length"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_val_loss = checkpoint.get("best_val_loss", "N/A")
    if isinstance(best_val_loss, (int, float)):
        best_val_loss = f"{best_val_loss:.4f}"

    print(
        f"[Inference] Model loaded successfully! "
        f"(Best validation loss: {best_val_loss})"
    )

    # 6. Encode prompt
    prompt = args.prompt
    if not prompt:
        prompt = "Once upon a time"

    print(f"[Inference] Prompt: '{prompt}'")
    print(
        f"[Inference] Settings: Temperature={args.temperature}, "
        f"Top-K={args.top_k}, Max Tokens={args.max_tokens}"
    )
    print("-" * 60)
    print("Generating text...\n")

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)  # (1, T)

    # 7. Generate tokens
    top_k_arg = args.top_k if args.top_k > 0 else None
    output_tensor = model.generate(
        input_tensor,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=top_k_arg,
    )

    # 8. Decode to text
    generated_ids = output_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    print(generated_text)
    print("\n" + "-" * 60)
    print(
        "[Note] This is a small experimental character-level language model trained"
    )
    print(
        "on a few kilobytes of text. It demonstrates Transformer token prediction,"
    )
    print(
        "not ChatGPT-level reasoning or world knowledge."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
