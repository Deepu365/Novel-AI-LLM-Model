"""
tokenizer.py - Character-Level Tokenizer for MyOwnLLM
======================================================
Builds a vocabulary of unique characters from training text.
Features:
- Encodes strings into lists of token IDs (integers).
- Decodes lists of token IDs back into readable text.
- Fallback for unseen characters (<unk> token).
- Saves and loads vocabulary to JSON for reproducible inference.
"""

import json
import os
from typing import List, Dict


class CharTokenizer:
    """
    Lightweight character-level tokenizer.
    Ideal for beginners: zero complex C++ dependencies, 100% transparent math,
    and fast to train on small datasets.
    """
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.unk_token = "<unk>"
        self.unk_id: int = 0

    def fit(self, text: str) -> None:
        """Build vocabulary from a string of text."""
        if not text:
            raise ValueError("Cannot train tokenizer on empty text.")

        # Extract all unique characters in sorted order for reproducibility
        unique_chars = sorted(list(set(text)))

        # Reserve index 0 for unknown character fallback
        self.char_to_id = {self.unk_token: 0}
        self.id_to_char = {0: self.unk_token}

        for idx, char in enumerate(unique_chars, start=1):
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char

        self.vocab_size = len(self.char_to_id)
        self.unk_id = 0
        print(f"[Tokenizer] Fitted vocabulary with {self.vocab_size} unique tokens.")

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of integer token IDs."""
        if not self.char_to_id:
            raise RuntimeError("Tokenizer has not been fitted or loaded yet. Call fit() or load() first.")

        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of integer token IDs back into a readable string."""
        if not self.id_to_char:
            raise RuntimeError("Tokenizer has not been fitted or loaded yet. Call fit() or load() first.")

        chars = []
        for tid in token_ids:
            if tid in self.id_to_char:
                # If it's the unknown token, display as ? for clean reading
                ch = self.id_to_char[tid]
                chars.append("?" if ch == self.unk_token else ch)
            else:
                chars.append("?")
        return "".join(chars)

    def save(self, filepath: str) -> None:
        """Save vocabulary mapping to a JSON file."""
        data = {
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Vocabulary saved to '{filepath}'.")

    def load(self, filepath: str) -> None:
        """Load vocabulary mapping from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file '{filepath}' not found.")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char_to_id = data["char_to_id"]
        self.vocab_size = data["vocab_size"]
        # Reconstruct reverse mapping
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.unk_id = self.char_to_id.get(self.unk_token, 0)
        print(f"[Tokenizer] Loaded vocabulary ({self.vocab_size} tokens) from '{filepath}'.")


if __name__ == "__main__":
    # Quick self-test when run directly
    sample = "Hello, Transformer! 123"
    tok = CharTokenizer()
    tok.fit(sample)
    encoded = tok.encode("Hello 123!")
    decoded = tok.decode(encoded)
    print("Original :", "Hello 123!")
    print("Encoded  :", encoded)
    print("Decoded  :", decoded)
    assert decoded == "Hello 123!", "Self-test failed!"
    print("Tokenizer self-test passed!")
