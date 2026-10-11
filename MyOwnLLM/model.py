"""
model.py - MyOwnLLM: Decoder-Only Transformer Language Model
============================================================
A small, self-contained Transformer language model implemented from scratch
in PyTorch for educational experiments and laptop CPU training.

Architecture:
- Character / Token Embedding + Learned Positional Embedding
- Pre-LayerNorm Transformer Blocks (Residual connections around attention and MLP)
- Multi-Head Causal Self-Attention with strictly masked future tokens
- Feed-Forward MLP (Linear -> GELU -> Linear)
- Final LayerNorm + Linear LM Head (Language Modeling Head)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.
    Ensures token at position t can only attend to positions <= t.
    """
    def __init__(self, d_model: int, n_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context_length = context_length

        # Fused Query, Key, Value linear projection: 3 * d_model
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output linear projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

        # Causal mask: Lower triangular matrix of ones.
        # Register as a buffer so it moves with the model device (CPU/GPU) without being a trainable parameter.
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, SequenceLength, d_model)
        B, T, C = x.shape

        # Compute Q, K, V in one matrix multiplication, then split
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (Batch, n_heads, SequenceLength, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product: (Q @ K^T) / sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        # Apply causal mask: mask positions where causal_mask is 0 with -infinity
        # Slice mask up to current sequence length T
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over last dimension (keys) to get attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted combination of values: (scores @ V)
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)

        # Recombine heads: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class MLP(nn.Module):
    """
    Two-layer position-wise Feed-Forward Network with GELU non-linearity.
    Standard Transformer expansion factor is 4x.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer Decoder Block with Pre-LayerNorm architecture.
    Pre-LN provides stable gradient flow during training on CPUs.
    """
    def __init__(self, d_model: int, n_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around self-attention
        x = x + self.attn(self.ln1(x))
        # Residual connection around MLP
        x = x + self.mlp(self.ln2(x))
        return x


class MyOwnLLM(nn.Module):
    """
    Full Decoder-Only Transformer Language Model.
    Designed to be small, fast, and trainable on a regular laptop CPU.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        context_length: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        # Token embedding lookup table
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        # Learned positional embedding lookup table
        self.pos_embeddings = nn.Embedding(context_length, d_model)

        self.drop = nn.Dropout(dropout)

        # Stack of Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, context_length, dropout)
            for _ in range(n_layers)
        ])

        # Final Layer Normalization
        self.ln_final = nn.LayerNorm(d_model)

        # Language model head: maps hidden representations to vocabulary logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between input token embeddings and output head (improves small model convergence)
        self.lm_head.weight = self.token_embeddings.weight

        # Initialize weights with standard small normal distribution
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: Tensor of shape (Batch, SequenceLength) containing token indices.
            targets: Optional tensor of shape (Batch, SequenceLength) for loss calculation.
        Returns:
            logits: (Batch, SequenceLength, vocab_size)
            loss: scalar Cross-Entropy loss if targets provided, else None.
        """
        B, T = idx.shape
        if T > self.context_length:
            raise ValueError(f"Input sequence length {T} exceeds model context length {self.context_length}")

        # Compute embeddings
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        tok_emb = self.token_embeddings(idx)                             # (B, T, d_model)
        pos_emb = self.pos_embeddings(positions)                        # (1, T, d_model)

        x = self.drop(tok_emb + pos_emb)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross_entropy
            # logits: (B*T, vocab_size), targets: (B*T)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 10,
    ) -> torch.Tensor:
        """
        Autoregressively generates new tokens given a starting prompt tensor idx (1, T).
        Uses temperature scaling and top-k filtering.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop current sequence if it exceeds the context length
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]

            # Forward pass to get logits for the current sequence
            logits, _ = self(idx_cond)

            # Pluck the logits at the very last position
            logits = logits[:, -1, :]  # shape: (1, vocab_size)

            # Apply temperature scaling (higher = more random, lower = more deterministic)
            if temperature > 0:
                logits = logits / max(temperature, 1e-5)
            else:
                # Greedy argmax if temperature is zero
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, next_token), dim=1)
                continue

            # Optional Top-K sampling: crop logits to top k highest values
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Softmax to convert logits into valid probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Append sampled token to running sequence
            idx = torch.cat((idx, next_token), dim=1)

        return idx
