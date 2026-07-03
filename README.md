# Agentic SLM

A **from-scratch agentic AI system** that autonomously organizes a messy filesystem. It generates its own training data, trains a custom BPE tokenizer, builds a ~3M parameter Transformer (Small Language Model), trains it on agentic reasoning sequences, and then deploys the model as an autonomous agent that can **observe, think, and act** in a sandboxed file system — creating directories, moving files, and deleting junk entirely on its own.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1 — Synthetic Data Generation](#stage-1--synthetic-data-generation-data_genpy)
  - [Stage 2 — Custom BPE Tokenizer](#stage-2--custom-bpe-tokenizer-tokenizerpy)
  - [Stage 3 — Transformer Model](#stage-3--transformer-model-modelpy)
  - [Stage 4 — Training](#stage-4--training-trainpy)
  - [Stage 5 — Agent Deployment](#stage-5--agent-deployment-agentpy)
- [The Agentic Loop](#the-agentic-loop)
- [KV-Cache Optimization](#kv-cache-optimization)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)

---

## Overview

Most AI agents wrap API calls around a large cloud-hosted model. **Agentic Janitor takes a fundamentally different approach** — the entire intelligence pipeline is built from the ground up:

1. **Generate** 200,000 synthetic agentic reasoning sequences (Observe → Think → Act)
2. **Train** a custom Byte-Pair Encoding tokenizer with agentic special tokens
3. **Build** a decoder-only Transformer with causal self-attention and KV-caching
4. **Train** the model via next-token prediction on the synthetic dataset
5. **Deploy** the model as an autonomous agent in a sandboxed directory

The agent learns three core actions:
| Action   | Description                                        |
|----------|----------------------------------------------------|
| `MOVE`   | Move a file to the correct categorized directory   |
| `MKDIR`  | Create a missing directory before moving a file    |
| `DELETE` | Remove junk/temporary files (`.tmp`, `.bak`, `.cache`) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agentic Janitor                           │
│                                                                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐               │
│  │  data_gen   │──▶│ tokenizer  │──▶│   model    │               │
│  │ (200K seqs) │   │ (BPE 5000) │   │ (~3M params)│              │
│  └────────────┘   └────────────┘   └─────┬──────┘               │
│                                          │                       │
│                                    ┌─────▼──────┐               │
│                                    │   train     │               │
│                                    │ (2000 steps)│               │
│                                    └─────┬──────┘               │
│                                          │                       │
│                                    ┌─────▼──────┐               │
│                                    │   agent     │               │
│                                    │ (sandbox)   │               │
│                                    └─────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1 — Synthetic Data Generation (`data_gen.py`)

Generates **200,000** structured agentic reasoning sequences. Each sequence follows the `<OBSERVE> → <THINK> → <ACT>` format that teaches the model how to reason about file organization.

**File type taxonomy** — the generator understands 8 file categories:

| Category         | Extensions               | Signal Words                          | Target Directory          |
|------------------|--------------------------|---------------------------------------|---------------------------|
| Python Code      | `.py`                    | `def main():`, `import os`, `torch.nn`| `/src/python`             |
| JavaScript Code  | `.js`, `.ts`             | `console.log`, `function()`, `const server` | `/src/js`          |
| Financial Docs   | `.pdf`, `.csv`           | `Q3 Revenue`, `Tax Return`            | `/docs/2026/financials`   |
| Log Files        | `.log`, `.txt`           | `[ERROR]`, `Traceback`, `kernel panic`| `/logs`                   |
| Generic Code     | `.py`, `.js`, `.html`    | `def main():`, `console.log`          | `/src`                    |
| Images           | `.png`, `.jpg`           | `Resolution: 1920x1080`, `RGB`        | `/media`                  |
| Documents        | `.md`, `.pdf`, `.txt`    | `Meeting notes`, `Dear team,`         | `/docs`                   |
| Junk             | `.tmp`, `.bak`, `.cache` | `temp data`, `cache dump`, `backup corrupted` | **DELETE**        |

**Key design decisions:**
- **Environment simulation** — each sequence includes an `Env_ls` snapshot showing which directories exist, training the model to check infrastructure before acting
- **MKDIR training** — randomly omits the target directory from the environment, forcing the model to learn the `MKDIR → MOVE` two-step pattern
- **Noise injection** — distractor words are added to file content so the model learns to identify signal words amidst noise
- **Separator token** — each sequence ends with `<|endoftext|>` to teach the model sequence boundaries

**Example generated sequence:**
```
<OBSERVE> Env_ls: [/media, /docs]. File: 'server_42.log'. Content: 'the and quick [ERROR] this is a test' </OBSERVE>
<THINK> Content indicates this is a log file. It belongs in /logs. Checking environment... /logs does not exist. I must create the folder first. </THINK>
<ACT> MKDIR /logs </ACT>
<|endoftext|>
```

---

### Stage 2 — Custom BPE Tokenizer (`tokenizer.py`)

Trains a **Byte-Pair Encoding** tokenizer from scratch on the generated dataset using the HuggingFace `tokenizers` library.

**Vocabulary size:** 5,000 tokens — deliberately small to force the model to learn semantic chunks rather than memorizing entire sentences.

**Special tokens (order matters):**

| Token             | Purpose                                  |
|-------------------|------------------------------------------|
| `<\|endoftext\|>` | End-of-sequence separator                |
| `[UNK]`           | Unknown token fallback                   |
| `<OBSERVE>`       | Opens the observation/perception block   |
| `</OBSERVE>`      | Closes the observation block             |
| `<THINK>`         | Opens the reasoning/chain-of-thought block |
| `</THINK>`        | Closes the reasoning block               |
| `<ACT>`           | Opens the action block                   |
| `</ACT>`          | Closes the action block                  |
| `<SUCCESS>`       | Reserved for real sandbox feedback       |
| `<ERROR>`         | Reserved for real sandbox feedback       |

The tokenizer is trained with whitespace pre-tokenization and preserves the special agentic tags as atomic tokens, ensuring the model can reliably open and close structured reasoning blocks.

---

### Stage 3 — Transformer Model (`model.py`)

A **decoder-only Transformer** (GPT-style) built entirely from scratch in PyTorch. No external model libraries — every layer is hand-written.

**Hyperparameters:**

| Parameter        | Value  | Rationale                                    |
|------------------|--------|----------------------------------------------|
| `d_model`        | 256    | Embedding dimension                          |
| `n_layers`       | 4      | Transformer blocks                           |
| `n_heads`        | 8      | Attention heads (256 / 8 = 32 dim per head)  |
| `max_seq_len`    | 256    | Maximum context window                       |
| `vocab_size`     | 5,000  | Matches the custom tokenizer                 |
| `dropout`        | 0.1    | Regularization                               |
| **Total params** | **~3M**| Small enough to train on a laptop CPU/GPU    |

**Model components:**

1. **`CausalSelfAttention`** — Multi-head attention with a causal (lower-triangular) mask. Supports **KV-caching** for efficient autoregressive generation: during inference, previously computed key-value pairs are cached and concatenated with new ones, so only the latest token needs to be processed.

2. **`FeedForward`** — Standard 2-layer MLP with GELU activation and 4× expansion factor.

3. **`Block`** — A single Transformer block: `LayerNorm → Attention → Residual → LayerNorm → MLP → Residual` (pre-norm architecture).

4. **`AgenticSLM`** — The complete model:
   - Token embeddings + learned positional embeddings
   - Stack of `n_layers` Transformer blocks
   - Final LayerNorm + linear language modeling head
   - **Weight tying** between the token embedding matrix and the output projection head (saves parameters)
   - GPT-style normal distribution weight initialization (mean=0, std=0.02)

---

### Stage 4 — Training (`train.py`)

Standard **next-token prediction** training loop with the following configuration:

| Parameter        | Value    |
|------------------|----------|
| Batch size       | 16       |
| Learning rate    | 3e-4     |
| Optimizer        | AdamW    |
| Max iterations   | 2,000    |
| Eval interval    | 200 steps|

**Key implementation details:**
- **Memory-safe data loading** — the dataset is encoded line-by-line to prevent OOM crashes in WSL environments, rather than loading the entire 65MB+ text file into memory at once
- **Random batch sampling** — random starting indices are selected from the tokenized dataset; input `x` is a sequence of tokens and target `y` is the same sequence shifted by one position
- The trained model weights are saved to `data/agentic_janitor.pt` (~15MB)

---

### Stage 5 — Agent Deployment (`agent.py`)

The trained model is deployed as an **autonomous agent** inside a sandboxed directory. This is where all the pieces come together.

**Sandbox setup** — three test files are created:

| File              | Content                                              | Expected Action        |
|-------------------|------------------------------------------------------|------------------------|
| `server_42.log`   | `This is a test file. [ERROR] connection timeout on port 8080.` | MKDIR `/logs` → MOVE   |
| `script_99.py`    | `import os\ndef main():\n    print('Hello World')`   | MKDIR `/src` → MOVE    |
| `vacation.bak`    | `cache dump ignore this corrupted data`              | DELETE                 |

**Generation** uses greedy decoding (argmax) with KV-caching for efficient token-by-token generation. Generation stops when the model produces a `</ACT>` or `<|endoftext|>` token.

**Command execution** includes bulletproof parsing:
- Regex extraction of `<ACT>...</ACT>` blocks
- Tokenizer space cleanup around slashes (e.g., `" / "` → `"/"`)
- Safe file operations within the sandbox directory only

---

## The Agentic Loop

The agent uses a **two-level loop** for true multi-step reasoning:

```
OUTER LOOP (scan directory)
│
├── Pick the first unsorted file
│
└── INNER LOOP (up to 3 steps per file)
    │
    ├── Step 1: OBSERVE environment → THINK → ACT (e.g., MKDIR /logs)
    │           System executes → SUCCESS: Created directory
    │           ↓ (loop continues — directory was created, file still needs moving)
    │
    ├── Step 2: OBSERVE updated environment → THINK → ACT (e.g., MOVE file /logs)
    │           System executes → SUCCESS: Moved file
    │           ↓ (break — file handled)
    │
    └── Next file...
```

This allows the agent to handle the common two-step pattern: **create the missing directory first, then move the file into it** — all autonomously without any hardcoded rules.

---

## KV-Cache Optimization

The model implements **KV-caching** in the attention layers. During autoregressive generation:

- **Without cache:** The entire sequence (prompt + all generated tokens so far) is re-processed at every step. Computation grows quadratically: O(n²).
- **With cache:** Previously computed key-value pairs are stored and reused. Only the single newest token is processed at each step. Computation grows linearly: O(n).

This optimization is critical for the agent loop where the model generates multiple tokens per action.


## Project Structure

```
agentic_janitor/
│
├── run_all.sh                  # One-command pipeline runner (stages 1-5)
├── README.md                   # This file
├── .gitignore                  # Excludes data artifacts, sandbox, and caches
│
├── src/
│   ├── data_gen.py             # Stage 1: Synthetic dataset generator (200K sequences)
│   ├── tokenizer.py            # Stage 2: Custom BPE tokenizer trainer (vocab=5000)
│   ├── model.py                # Stage 3: Decoder-only Transformer (~3M params)
│   ├── train.py                # Stage 4: Training loop (AdamW, 2000 steps)
│   └── agent.py                # Stage 5: Autonomous agent with sandbox execution
│
├── data/                       # Generated artifacts (gitignored)
│   ├── synthetic_dataset.txt   # ~65MB of training sequences
│   ├── custom_tokenizer.json   # Trained BPE tokenizer
│   └── agentic_janitor.pt      # Trained model weights (~15MB)
│
└── sandbox/                    # Agent's sandboxed workspace (gitignored)
    ├── server_42.log           # Test file: log with error signals
    └── script_99.py            # Test file: Python source code
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch (`torch`)
- HuggingFace `tokenizers`

### Install Dependencies

```bash
pip install torch tokenizers
```

### Run the Full Pipeline

The entire project can be run end-to-end with a single command:

```bash
chmod +x run_all.sh
./run_all.sh
```

This executes all 5 stages sequentially:

```
[1/5] Generating Synthetic Data...    → Creates data/synthetic_dataset.txt
[2/5] Training Tokenizer...           → Creates data/custom_tokenizer.json
[3/5] Compiling and Testing Model...  → Validates the model architecture
[4/5] Training the SLM...             → Creates data/agentic_janitor.pt
[5/5] Deploying Agent to Sandbox...   → Runs the autonomous agent
```

### Run Individual Stages

```bash
python3 src/data_gen.py       # Generate training data
python3 src/tokenizer.py      # Train the tokenizer
python3 src/model.py          # Test model compilation
python3 src/train.py          # Train the model
python3 src/agent.py          # Deploy the agent
```


---

## Configuration

Key hyperparameters can be tuned in the respective files:

| Parameter          | File          | Default  | Notes                                  |
|--------------------|---------------|----------|----------------------------------------|
| `num_samples`      | `data_gen.py` | 200,000  | More data → better generalization      |
| `vocab_size`       | `tokenizer.py`| 5,000    | Must match `ModelArgs.vocab_size`      |
| `d_model`          | `model.py`    | 256      | Larger → more capacity, slower         |
| `n_layers`         | `model.py`    | 4        | Deeper → better reasoning, slower      |
| `n_heads`          | `model.py`    | 8        | Must evenly divide `d_model`           |
| `max_seq_len`      | `model.py`    | 256      | Maximum context window                 |
| `BATCH_SIZE`       | `train.py`    | 16       | Larger → needs more VRAM              |
| `LEARNING_RATE`    | `train.py`    | 3e-4     | Standard for small Transformers        |
| `MAX_ITERS`        | `train.py`    | 2,000    | Increase to 5000+ for better results   |

---

> **Note:** All data artifacts (`synthetic_dataset.txt`, `custom_tokenizer.json`, `agentic_janitor.pt`) and the `sandbox/` directory are gitignored. Run the pipeline to regenerate them.
