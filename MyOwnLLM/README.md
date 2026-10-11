# MyOwnLLM: Build a Decoder-Only Transformer from Scratch

**MyOwnLLM** is an educational, beginner-friendly Python project that implements a miniature decoder-only Transformer language model from scratch in PyTorch without using any external AI APIs or pretrained weights.

It is specifically tuned to train quickly (in ~1 to 3 minutes) on a standard **Windows laptop CPU**.

---

## 📂 Project Structure

```text
MyOwnLLM/
│── dataset.txt        # Small training text corpus (stories, dialogues)
│── tokenizer.py       # Character-level tokenizer with save/load functionality
│── model.py           # Decoder-only Transformer architecture in PyTorch
│── train.py           # Training loop with AdamW, validation loss & checkpoint saving
│── generate.py        # Autoregressive generation script (temperature & top-k)
│── test_smoke.py      # Quick automated sanity test
│── requirements.txt   # Minimal dependencies (PyTorch)
│── run_windows.bat    # Optional 1-click Windows batch script
└── README.md          # Complete documentation and Windows guide
```

---

## 💻 Exact Windows CMD Setup & Execution Guide

Follow these exact steps in **Command Prompt (CMD)** on your Windows laptop:

### 1. Open Windows Command Prompt
Press `Win + R`, type `cmd`, and press `Enter`.

### 2. Verify Python Installation
```cmd
python --version
```
*(You should see Python 3.9, 3.10, 3.11, or 3.12. If not installed, download from python.org and check "Add Python to PATH" during install).*

### 3. Create Project Directory
```cmd
mkdir C:\MyOwnLLM
cd C:\MyOwnLLM
```
*(Save the project files into this folder `C:\MyOwnLLM`).*

### 4. Create and Activate a Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```
*(Your CMD prompt will now show `(venv)` at the beginning of the line).*

### 5. Install PyTorch (CPU Edition for Windows)
Running on CPU avoids huge multi-gigabyte CUDA downloads. Run:
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 6. Run the Quick Smoke Test
Before training, verify everything works:
```cmd
python test_smoke.py
```
You should see: `>>> ALL SMOKE TESTS PASSED! <<<`.

### 7. Train the Model
```cmd
python train.py --epochs 20
```
- Trains in approximately **90–120 seconds** on an ordinary laptop CPU.
- Displays epoch, step count, train loss, and validation loss.
- Saves the trained model to `my_own_llm.pt` and vocabulary to `vocab.json`.

### 8. Generate Text
Once training is finished, generate text using your model:
```cmd
python generate.py --prompt "Once upon a time"
```

Custom parameters:
```cmd
python generate.py --prompt "Deep in the digital kingdom" --temperature 0.7 --top_k 5 --max_tokens 200
```

---

## 🧠 Architecture Overview

### 1. Tokenization (`tokenizer.py`)
- Character-level mapping: converts text characters into integer tokens.
- Includes `<unk>` fallback token for unseen characters.
- Stores vocabulary mapping to `vocab.json`.

### 2. Causal Self-Attention (`model.py`)
- Standard Scaled Dot-Product Attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
- Uses a lower-triangular causal mask $M$ where future token positions are filled with $-\infty$, preventing tokens from "cheating" by peeking ahead.

### 3. Transformer Block
- Pre-LayerNorm configuration for numerical stability.
- Multi-Head Attention followed by a 2-layer MLP with GELU activation.
- Residual connections (`x = x + sublayer(x)`).

### 4. Language Model Head & Next-Token Prediction
- Maps the final representations to vocabulary logits.
- Computes Cross-Entropy loss between logits and target shifted tokens.
- Autoregressive generation via temperature scaling and Top-K multinomial sampling.

---

## ⚠️ Realistic Expectations: Toy LLM vs ChatGPT

This project is an **educational laboratory model**, NOT a production assistant:

| Metric / Aspect | MyOwnLLM | ChatGPT (GPT-4 / Claude) |
| :--- | :--- | :--- |
| **Parameters** | ~250,000 (~0.25M) | >1,000,000,000,000 (>1 Trillion) |
| **Training Data** | ~18 KB (~15,000 chars) | >15 Trillion tokens (Entire Internet) |
| **Training Hardware** | 1 Laptop CPU (2 minutes) | 25,000 H100 GPUs (several months, $50M+) |
| **Tokenization** | Character-level (80 vocab) | Byte-Pair Encoding BPE (100k vocab) |
| **Output Behavior** | Learns local spelling, quotes & simple rhyme patterns | High-level reasoning, code synthesis, multi-turn dialogue |
| **Alignment** | Pure raw autoregressive sampling | Supervised Fine-Tuning (SFT) + RLHF / DPO |

### Why build this?
Building this from scratch is the fastest way to understand the core mathematics that power **every modern decoder-only LLM** (GPT-4, Llama 3, Gemini, Mistral) without needing server farms.
