
"""
test_smoke.py - Smoke Test for MyOwnLLM
=======================================
Quick checks for PyTorch, tokenizer, model, forward pass,
loss, gradients, and text generation.
"""

import sys
import torch

from tokenizer import CharTokenizer
from model import MyOwnLLM


def run_smoke_test():
    print("=" * 60)
    print("        MyOwnLLM: Running Quick Smoke Test")
    print("=" * 60)

    # Step 1: Check PyTorch
    print(f"[1/6] PyTorch Version: {torch.__version__}")
    print(f"      CUDA Available: {torch.cuda.is_available()}")

    # Step 2: Test Tokenizer
    print("[2/6] Testing CharTokenizer encode & decode...")
    sample_text = "The quick brown fox jumps over the lazy dog. 12345!"
    tokenizer = CharTokenizer()
    tokenizer.fit(sample_text)

    encoded = tokenizer.encode("The quick fox")
    decoded = tokenizer.decode(encoded)
    assert decoded == "The quick fox", (
        f"Decoded text '{decoded}' does not match original!"
    )
    print(f"      Tokenizer passed! Vocab size: {tokenizer.vocab_size}")

    # Step 3: Test Model Instantiation
    print("[3/6] Instantiating MyOwnLLM Transformer...")
    vocab_size = 50
    d_model = 64
    n_heads = 4
    n_layers = 2
    context_length = 32

    model = MyOwnLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        context_length=context_length,
        dropout=0.0,
    )

    param_count = model.count_parameters()
    print(f"      Model created with {param_count:,} parameters.")
    assert param_count > 0, "Model has 0 parameters!"

    # Step 4: Test Forward Pass
    print("[4/6] Testing Forward Pass tensor shapes...")
    batch_size = 2
    seq_len = 16

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(dummy_input, dummy_targets)
    expected_shape = (batch_size, seq_len, vocab_size)

    assert logits.shape == expected_shape, (
        f"Expected logits shape {expected_shape}, got {logits.shape}"
    )
    assert loss is not None and loss.item() > 0, (
        f"Loss must be positive, got {loss}"
    )
    print(
        f"      Forward pass passed! Output shape: {logits.shape}, "
        f"Loss: {loss.item():.4f}"
    )

    # Step 5: Test Backward Pass
    print("[5/6] Testing Backward Pass (gradient propagation)...")
    loss.backward()

    grad_found = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, (
                f"Parameter {name} has no gradient!"
            )
            if param.grad.abs().sum() > 0:
                grad_found = True

    assert grad_found, "Gradients are all zero!"
    print("      Backward pass passed! Gradients computed cleanly.")

    # Step 6: Test Text Generation
    print("[6/6] Testing Autoregressive generate() method...")
    seed_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated = model.generate(
        seed_tokens,
        max_new_tokens=5,
        temperature=1.0,
    )

    assert generated.shape == (1, 8), (
        f"Expected generated shape (1, 8), got {generated.shape}"
    )
    print(f"      Generation passed! Output shape: {generated.shape}")

    print("\nSmoke test completed")
    print("  >>> ALL SMOKE TESTS PASSED! <<<")
    print("  Your environment and Transformer implementation are ready.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
