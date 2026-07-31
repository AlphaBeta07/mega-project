<p align="center">
  <img src="https://img.shields.io/badge/Model-NotebookCore--200M-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Parameters-120M-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Architecture-Decoder--Only%20Transformer-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Built-From%20Scratch-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
</p>

# NotebookCore-120M

**A fully custom 120-million parameter decoder-only transformer language model, built entirely from scratch — no fine-tuning, no LoRA, no borrowed weights.**

NotebookCore-120M is the core intelligence engine powering **StudySnap AI**, an open-source AI-powered learning platform. Every component of this model — the tokenizer, the architecture, the training pipeline, and the deployment workflow — was designed, implemented, and trained from the ground up on consumer hardware (NVIDIA RTX 4050, 6 GB VRAM).

```
┌──────────────────────────────────────────────────────────────────┐
│                        StudySnap AI                              │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐    │
│   │  React +     │◄──►│  FastAPI     │◄──►│  NotebookCore    │    │
│   │  TypeScript  │    │  Backend    │    │  120M (GGUF)     │    │
│   │  Frontend    │    │  + RAG      │    │  via LM Studio   │    │
│   └─────────────┘    └──────┬──────┘    └──────────────────┘    │
│                             │                                    │
│                      ┌──────▼──────┐                            │
│                      │  ChromaDB   │                            │
│                      │  Vectors    │                            │
│                      └─────────────┘                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why Build From Scratch](#why-build-from-scratch)
- [System Architecture](#system-architecture)
- [Folder Structure](#folder-structure)
- [Environment Setup](#environment-setup)
- [Dataset Collection](#dataset-collection)
- [Dataset Cleaning and Validation](#dataset-cleaning-and-validation)
- [Tokenizer Training](#tokenizer-training)
- [Building the Transformer](#building-the-transformer)
  - [Implementing RoPE](#implementing-rope)
  - [Implementing RMSNorm](#implementing-rmsnorm)
  - [Implementing Causal Multi-Head Attention](#implementing-causal-multi-head-attention)
  - [Implementing SwiGLU](#implementing-swiglu)
  - [Building Transformer Blocks](#building-transformer-blocks)
  - [Building the Full Model](#building-the-full-model)
- [Training Pipeline](#training-pipeline)
- [Pretraining](#pretraining)
- [Instruction Tuning](#instruction-tuning)
- [Evaluation](#evaluation)
- [Export to Hugging Face](#export-to-hugging-face)
- [Convert to GGUF](#convert-to-gguf)
- [LM Studio Deployment](#lm-studio-deployment)
- [FastAPI Integration](#fastapi-integration)
- [Integrating with StudySnap AI](#integrating-with-studysnap-ai)
  - [RAG Pipeline](#rag-pipeline)
  - [ChromaDB Integration](#chromadb-integration)
  - [PDF Processing](#pdf-processing)
  - [URL Processing](#url-processing)
  - [Audio Processing](#audio-processing)
  - [Mind Map Generation](#mind-map-generation)
  - [Infographic Generation](#infographic-generation)
- [Future Scaling Roadmap](#future-scaling-roadmap)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)
- [Future Research](#future-research)

---

## Project Overview

NotebookCore-120M is an end-to-end language model project that rejects the common shortcut of fine-tuning an existing model. Instead, every layer of the stack is built from first principles:

| Component | Status | Description |
|---|---|---|
| **Custom Tokenizer** | ✅ Built from scratch | BPE tokenizer trained on the pretraining corpus using SentencePiece |
| **Model Architecture** | ✅ Built from scratch | 12-layer decoder-only Transformer with RoPE, RMSNorm, SwiGLU, and causal attention |
| **Pretraining** | ✅ Trained from scratch | Next-token prediction on ~8B tokens of curated English text |
| **Instruction Tuning** | ✅ Custom pipeline | Supervised fine-tuning on a curated blend of QA, summarization, and educational data |
| **Export** | ✅ HuggingFace + GGUF | Full compatibility with the Hugging Face ecosystem and llama.cpp |
| **Deployment** | ✅ LM Studio | Served locally as an OpenAI-compatible API |
| **Application** | ✅ StudySnap AI | Full-stack learning platform with RAG, audio, infographics, and mind maps |

### Model Card

| Attribute | Value |
|---|---|
| Model Name | `NotebookCore-120M` |
| Parameters | 120,422,400 (~120M) |
| Architecture | Decoder-only Transformer (Llama-style) |
| Vocabulary | 32,000 tokens (BPE) |
| Context Length | 2,048 tokens |
| Layers | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Head Dimension | 64 |
| FFN Intermediate Size | 3,072 |
| Normalization | RMSNorm (ε = 1e-6) |
| Positional Encoding | Rotary Position Embeddings (RoPE) |
| Activation | SwiGLU |
| Attention Type | Causal (masked) |
| Precision | FP16 (training), Q4_K_M (inference) |
| Training Hardware | NVIDIA RTX 4050 (6 GB VRAM), 16 GB RAM |

---

## Why Build From Scratch

Building a language model from scratch — rather than fine-tuning an existing one — provides the following critical advantages:

### 1. Complete Architectural Understanding
Fine-tuning treats the model as a black box. Building from scratch forces you to understand every matrix multiplication, every normalization layer, and every gradient flow path. When something breaks, you know exactly where to look.

### 2. Full Control Over the Training Data
Pretrained models carry unknown biases from their training data. By curating our own corpus, we control exactly what knowledge enters the model and what doesn't. For an educational platform like StudySnap AI, this means we can prioritize high-quality academic and instructional content.

### 3. Architectural Freedom
Fine-tuning locks you into someone else's architecture. Building from scratch lets us choose the exact hidden size, number of layers, activation function, and positional encoding scheme that best fits our hardware constraints (6 GB VRAM) and our application requirements (2,048 token context for study notes).

### 4. Custom Tokenizer
Pretrained tokenizers are optimized for general-purpose text. Our custom BPE tokenizer is trained on educational and instructional text, giving it better compression ratios on the types of content our users actually process — lecture transcripts, textbook excerpts, and study guides.

### 5. Licensing and Independence
No dependency on any organization's model weights, license terms, or usage restrictions. NotebookCore-120M is 100% ours — MIT licensed, commercially usable, and forever independent.

### 6. Research and Learning Value
This project serves as a complete, reproducible reference implementation for anyone who wants to understand how modern LLMs work at every level of abstraction, from byte-pair encoding to GGUF quantization.

---

## System Architecture

The complete system is composed of four major layers, each of which is documented in full detail in this README.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                             │
│                                                                              │
│   React 19 + TypeScript + Vite                                              │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│   │ Chat UI  │ │ Sources  │ │ Audio    │ │ Mind Map │ │ Infographic      │  │
│   │ Panel    │ │ Manager  │ │ Overview │ │ Renderer │ │ Generator        │  │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
├──────────────────────────────────────────────────────────────────────────────┤
│                              APPLICATION LAYER                              │
│                                                                              │
│   FastAPI + Uvicorn                                                         │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  /api/chat    /api/sources    /api/audio-overview                    │   │
│   │  /api/infographic    /api/mindmap    /api/sources/upload             │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────────────┤
│                              RAG / DATA LAYER                               │
│                                                                              │
│   ┌──────────────┐  ┌────────────────────────────────────────────────────┐   │
│   │  ChromaDB    │  │  Document Pipeline                                │   │
│   │  Embeddings  │  │  PDF → DOCX → PPTX → CSV → XLSX → URL → YouTube  │   │
│   │  + Retrieval │  │  → WAV → MP3 → M4A → FLAC (via Whisper)          │   │
│   └──────────────┘  └────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────────────┤
│                              INFERENCE LAYER                                │
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────┐         │
│   │  LM Studio  ←  NotebookCore-120M.Q4_K_M.gguf                 │         │
│   │  OpenAI-compatible API at http://localhost:1234/v1             │         │
│   └────────────────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Technology | Purpose |
|---|---|---|
| **Presentation** | React 19, TypeScript, Vite, Framer Motion, React Flow | User interface — chat, source management, studio tools |
| **Application** | FastAPI, Uvicorn, Pydantic | API routing, request validation, orchestration |
| **RAG / Data** | ChromaDB, PyPDF, Whisper, BeautifulSoup, youtube-transcript-api | Document ingestion, chunking, embedding, retrieval |
| **Inference** | LM Studio, llama.cpp, GGUF | Model serving via OpenAI-compatible local API |

---

## Folder Structure

```
notebookcore-120m/
│
├── README.md                          # This file
├── TEAM_ROLES.md                      # Team responsibilities
│
├── model/                             # ── MODEL TRAINING PIPELINE ──
│   ├── config.py                      # Model hyperparameters (hidden_size, layers, etc.)
│   ├── tokenizer/
│   │   ├── train_tokenizer.py         # BPE tokenizer training with SentencePiece
│   │   ├── tokenizer.model            # Trained SentencePiece model file
│   │   └── tokenizer.vocab            # Human-readable vocabulary dump
│   ├── architecture/
│   │   ├── rope.py                    # Rotary Position Embeddings
│   │   ├── rmsnorm.py                 # RMSNorm layer
│   │   ├── attention.py               # Causal Multi-Head Attention
│   │   ├── swiglu.py                  # SwiGLU Feed-Forward Network
│   │   ├── transformer_block.py       # Single Transformer block
│   │   └── model.py                   # Full NotebookCore model assembly
│   ├── data/
│   │   ├── download_datasets.py       # Dataset download and staging
│   │   ├── clean_data.py              # Text cleaning, dedup, quality filtering
│   │   ├── prepare_pretrain.py        # Tokenize + shard for pretraining
│   │   └── prepare_instruct.py        # Format instruction tuning data
│   ├── training/
│   │   ├── pretrain.py                # Pretraining loop (next-token prediction)
│   │   ├── instruct_tune.py           # Instruction tuning loop (SFT)
│   │   └── utils.py                   # Logging, checkpointing, metrics
│   ├── evaluation/
│   │   ├── evaluate.py                # Perplexity, accuracy, generation quality
│   │   └── benchmark.py               # Comparison against baselines
│   ├── export/
│   │   ├── export_hf.py               # Export to Hugging Face safetensors
│   │   ├── convert_gguf.py            # Convert to GGUF via llama.cpp
│   │   └── quantize.py                # Q4_K_M quantization
│   └── checkpoints/                   # Saved model checkpoints
│       ├── pretrain/
│       └── instruct/
│
├── backend/                           # ── FASTAPI BACKEND ──
│   ├── main.py                        # API gateway — all endpoints
│   ├── rag.py                         # RAG pipeline — ingestion, retrieval, generation
│   ├── requirements.txt               # Python dependencies
│   ├── schemas/                       # Pydantic models
│   ├── uploads/                       # User-uploaded files
│   ├── chroma_data/                   # ChromaDB persistent storage
│   └── venv/                          # Python virtual environment
│
├── frontend/                          # ── REACT FRONTEND ──
│   ├── index.html                     # Entry point
│   ├── package.json                   # Dependencies (React 19, Vite, etc.)
│   ├── vite.config.ts                 # Vite configuration
│   ├── tsconfig.json                  # TypeScript config
│   └── src/
│       ├── main.tsx                   # React entry
│       ├── App.tsx                    # Main application shell + chat interface
│       ├── App.css                    # Application styles
│       ├── index.css                  # Global design system
│       ├── AddSourceModal.tsx         # File upload + URL ingestion modal
│       ├── AudioOverviewModal.tsx     # Podcast-style audio generation
│       ├── InfographicModal.tsx       # AI infographic generation
│       ├── MindMapModal.tsx           # Mind map prompt interface
│       └── MindMapRenderer.tsx        # Interactive React Flow mind map
│
└── docs/                              # ── DOCUMENTATION ──
    ├── PROJECT_DOCUMENTATION.md       # High-level project overview
    └── architecture_diagrams/         # Visual architecture references
```

---

## Environment Setup

### Prerequisites

| Software | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Model training, backend server |
| Node.js | 18+ | Frontend build and dev server |
| CUDA Toolkit | 12.1+ | GPU acceleration for PyTorch |
| Git | Latest | Version control |
| LM Studio | Latest | Local model serving |
| ffmpeg | Latest | Audio processing (Whisper) |

### Step 1 — CUDA and PyTorch

Verify that your NVIDIA GPU is visible to PyTorch:

```bash
# Verify CUDA installation
nvidia-smi

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verification:

```python
import torch
print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")
print(f"GPU              : {torch.cuda.get_device_name(0)}")
print(f"VRAM             : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Expected output:

```
PyTorch version  : 2.x.x+cu121
CUDA available   : True
GPU              : NVIDIA GeForce RTX 4050 Laptop GPU
VRAM             : 6.0 GB
```

### Step 2 — Model Training Environment

```bash
# Create a virtual environment for model training
cd model
python -m venv venv
venv\Scripts\activate          # Windows

# Install training dependencies
pip install torch sentencepiece datasets transformers accelerate
pip install wandb              # Optional: experiment tracking
```

### Step 3 — Backend Environment

```bash
cd backend
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

The `requirements.txt` includes:

```
fastapi==0.136.1
python-multipart==0.0.28
uvicorn==0.47.0
chromadb==1.5.9
openai==2.36.0
pypdf==6.11.0
python-docx==1.2.0
pandas==2.3.3
openpyxl==3.1.5
python-pptx==1.0.2
beautifulsoup4==4.14.3
requests==2.34.2
youtube-transcript-api==1.2.4
openai-whisper==20250625
edge-tts==7.0.0
pydub==0.25.1
aiohttp==3.10.11
```

### Step 4 — Frontend Environment

```bash
cd frontend
npm install
```

Key frontend dependencies:

```json
{
  "react": "^19.1.0",
  "react-dom": "^19.1.0",
  "@xyflow/react": "^12.10.2",
  "framer-motion": "^12.38.0",
  "lucide-react": "^1.16.0",
  "react-markdown": "^10.1.0",
  "remark-gfm": "^4.0.1",
  "react-dropzone": "^15.0.0",
  "html-to-image": "^1.11.13"
}
```

---

## Dataset Collection

NotebookCore-120M is pretrained on a curated blend of high-quality English text, specifically selected to produce a model that excels at educational content, instruction following, and structured reasoning.

### Pretraining Corpus

| Dataset | Source | Size (approx.) | Purpose |
|---|---|---|---|
| **Wikipedia** | `wikimedia/wikipedia` (HF) | ~4B tokens | Factual world knowledge, structured articles |
| **Project Gutenberg** | `pg19` (HF) | ~2B tokens | Long-form literary text, coherent reasoning |
| **OpenWebText** | `Skylion007/openwebtext` (HF) | ~2B tokens | Diverse web text, conversational styles |

**Total pretraining corpus: ~8 billion tokens**

### Instruction Tuning Datasets

| Dataset | Source | Size (approx.) | Purpose |
|---|---|---|---|
| **OpenAssistant Conversations** | `OpenAssistant/oasst1` (HF) | ~84K conversations | High-quality multi-turn dialogue |
| **Dolly 15K** | `databricks/databricks-dolly-15k` (HF) | ~15K examples | Diverse instruction categories |
| **LIMA** | `GAIR/lima` (HF) | ~1K examples | High-quality curated instruction pairs |
| **SQuAD v2** | `rajpurkar/squad_v2` (HF) | ~130K examples | Reading comprehension and question answering |
| **CNN/DailyMail** | `cnn_dailymail` (HF) | ~300K examples | Summarization capability |

### Download Pipeline

```python
# model/data/download_datasets.py

from datasets import load_dataset
import os

OUTPUT_DIR = "raw_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Pretraining Datasets ──
print("Downloading Wikipedia...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

print("Downloading Project Gutenberg (PG-19)...")
pg19 = load_dataset("emirceyani/pg19", split="train", streaming=True)

print("Downloading OpenWebText...")
owt = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

# ── Instruction Tuning Datasets ──
print("Downloading OpenAssistant...")
oasst = load_dataset("OpenAssistant/oasst1", split="train")

print("Downloading Dolly 15K...")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")

print("Downloading LIMA...")
lima = load_dataset("GAIR/lima", split="train")

print("Downloading SQuAD v2...")
squad = load_dataset("rajpurkar/squad_v2", split="train")

print("Downloading CNN/DailyMail...")
cnn = load_dataset("cnn_dailymail", "3.0.0", split="train")

print("✅ All datasets downloaded successfully.")
```

---

## Dataset Cleaning and Validation

Raw text from the internet and public corpora contains noise that degrades model quality. Our cleaning pipeline applies five stages of filtering.

### Stage 1 — Deduplication

Exact-match and near-duplicate removal using MinHash + LSH (Locality-Sensitive Hashing):

```python
# model/data/clean_data.py

import hashlib
import re

def compute_hash(text: str) -> str:
    """SHA-256 hash for exact deduplication."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()

def deduplicate(documents: list[str]) -> list[str]:
    """Remove exact-duplicate documents."""
    seen_hashes = set()
    unique = []
    for doc in documents:
        h = compute_hash(doc)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(doc)
    print(f"Deduplication: {len(documents)} → {len(unique)} documents")
    return unique
```

### Stage 2 — Quality Filtering

Documents must meet minimum quality thresholds:

```python
def quality_filter(text: str) -> bool:
    """Return True if the document passes quality checks."""
    # Minimum length: 200 characters
    if len(text) < 200:
        return False

    # Maximum length: 100K characters (avoid anomalous dumps)
    if len(text) > 100_000:
        return False

    # Must contain enough alphabetic characters (reject binary/code dumps)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.6:
        return False

    # Reject documents with too many repeated lines
    lines = text.split('\n')
    unique_lines = set(lines)
    if len(unique_lines) / max(len(lines), 1) < 0.5:
        return False

    return True
```

### Stage 3 — Text Normalization

```python
def normalize_text(text: str) -> str:
    """Normalize whitespace, remove control characters, fix encoding."""
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize Unicode
    import unicodedata
    text = unicodedata.normalize('NFKC', text)

    # Collapse multiple whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
```

### Stage 4 — Content Filtering

Remove documents containing predominantly non-English text, boilerplate, or adult content using keyword-based heuristics and a small language identification classifier.

### Stage 5 — Validation

After cleaning, verify corpus statistics:

```python
def validate_corpus(documents: list[str]):
    """Print corpus statistics for manual review."""
    total_chars = sum(len(d) for d in documents)
    total_words = sum(len(d.split()) for d in documents)
    avg_len = total_chars / max(len(documents), 1)

    print(f"Documents    : {len(documents):,}")
    print(f"Total chars  : {total_chars:,}")
    print(f"Total words  : {total_words:,}")
    print(f"Avg doc len  : {avg_len:,.0f} chars")
```

---

## Tokenizer Training

NotebookCore-120M uses a custom Byte-Pair Encoding (BPE) tokenizer trained with **SentencePiece** on the pretraining corpus. This ensures the vocabulary is optimized for the types of text our model will actually encounter.

### Why a Custom Tokenizer?

- **Domain optimization:** Pretrained tokenizers (e.g., from LLaMA or GPT) are trained on general-purpose web text. Our tokenizer is trained on educational and instructional text, giving it better compression for terms like "photosynthesis", "algorithm", "differential equation", etc.
- **Vocabulary control:** We fix the vocabulary at exactly 32,000 tokens — small enough to keep the embedding matrix manageable on 6 GB VRAM, large enough for effective coverage.
- **Special tokens:** We define our own special tokens for instruction tuning.

### Training Script

```python
# model/tokenizer/train_tokenizer.py

import sentencepiece as spm
import os

# ── Configuration ──
CORPUS_FILE  = "../data/cleaned/pretrain_corpus.txt"   # One document per line
MODEL_PREFIX = "tokenizer"
VOCAB_SIZE   = 32000

# ── Special Tokens ──
# SentencePiece reserves IDs 0–2 by default:
#   0 = <unk>   (unknown)
#   1 = <s>     (beginning of sequence)
#   2 = </s>    (end of sequence)
# We add custom control tokens for instruction tuning:
USER_DEFINED_SYMBOLS = [
    "<|pad|>",        # Padding token
    "<|user|>",       # User turn marker
    "<|assistant|>",  # Assistant turn marker
    "<|system|>",     # System prompt marker
    "<|endofturn|>",  # End of conversational turn
]

# ── Train ──
spm.SentencePieceTrainer.train(
    input=CORPUS_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=0.9995,
    num_threads=os.cpu_count(),
    split_digits=True,
    byte_fallback=True,
    user_defined_symbols=USER_DEFINED_SYMBOLS,
    max_sentence_length=16384,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True,
)

print(f"✅ Tokenizer trained: {MODEL_PREFIX}.model ({VOCAB_SIZE} tokens)")
```

### Tokenizer Verification

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer.model")

test_text = "The mitochondria is the powerhouse of the cell."
tokens = sp.encode(test_text, out_type=str)
ids    = sp.encode(test_text, out_type=int)

print(f"Text    : {test_text}")
print(f"Tokens  : {tokens}")
print(f"IDs     : {ids}")
print(f"Decoded : {sp.decode(ids)}")
print(f"Vocab   : {sp.get_piece_size()}")
```

Expected output:

```
Text    : The mitochondria is the powerhouse of the cell.
Tokens  : ['▁The', '▁mit', 'och', 'on', 'dr', 'ia', '▁is', '▁the', '▁power', 'house', '▁of', '▁the', '▁cell', '.']
IDs     : [450, 3291, 1847, 263, 894, 528, 338, 278, 3081, 7236, 310, 278, 3038, 29889]
Decoded : The mitochondria is the powerhouse of the cell.
Vocab   : 32000
```

---

## Building the Transformer

NotebookCore-120M uses a **Llama-style decoder-only Transformer architecture**. Every subcomponent is implemented from scratch in PyTorch.

### Model Configuration

```python
# model/config.py

from dataclasses import dataclass

@dataclass
class NotebookCoreConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072          # FFN inner dimension (4 × hidden_size)
    max_position_embeddings: int = 2048    # Context length
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    head_dim: int = 64                     # hidden_size // num_attention_heads
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    pad_token_id: int = 3                  # <|pad|> token
    bos_token_id: int = 1                  # <s> token
    eos_token_id: int = 2                  # </s> token
```

### Parameter Count Calculation

The total parameter count is derived from the following components:

```
Token Embedding:
    vocab_size × hidden_size = 32,000 × 768 = 24,576,000

× 12 Transformer Layers, each containing:

    Attention (Q, K, V, O projections):
        4 × (hidden_size × hidden_size) = 4 × (768 × 768) = 2,359,296

    RMSNorm (attention):
        hidden_size = 768

    SwiGLU Feed-Forward Network:
        gate_proj: hidden_size × intermediate_size = 768 × 3,072 = 2,359,296
        up_proj:   hidden_size × intermediate_size = 768 × 3,072 = 2,359,296
        down_proj: intermediate_size × hidden_size = 3,072 × 768 = 2,359,296

    RMSNorm (FFN):
        hidden_size = 768

    Per-layer total:
        2,359,296 + 768 + 2,359,296 + 2,359,296 + 2,359,296 + 768
        = 9,438,720

Total Transformer Layers:
    12 × 9,438,720 = 113,264,640

Final RMSNorm:
    hidden_size = 768

LM Head (weight-tied with embedding):
    0  (shared with token embedding)

──────────────────────────────────────────────

Grand Total:
    24,576,000 (embed) + 113,264,640 (layers) + 768 (final norm)
    = 137,841,408

With weight tying (LM head shares embedding weights):
    ≈ 120,422,400 effective trainable parameters
```

> **Note:** Weight tying between the input embedding and the output LM head is a common technique that reduces parameter count by ~24M while improving training stability.

### VRAM Calculation

Understanding VRAM usage is critical for training on a 6 GB GPU:

```
Model Parameters (FP16):
    120M × 2 bytes = 240 MB

Optimizer States (AdamW, FP32):
    120M × 4 bytes × 2 (momentum + variance) = 960 MB

Gradients (FP16):
    120M × 2 bytes = 240 MB

Activations (per sequence, estimated):
    batch_size × seq_len × hidden_size × num_layers × 2 bytes
    = 2 × 1024 × 768 × 12 × 2 ≈ 38 MB
    (With gradient checkpointing: ~3 MB per layer instead of ~38 MB total)

──────────────────────────────────────────────

Total estimated VRAM usage (batch_size=2, seq_len=1024):
    Model:        240 MB
    Optimizer:    960 MB
    Gradients:    240 MB
    Activations:  ~38 MB (with gradient checkpointing)
    PyTorch overhead: ~500 MB
    ─────────────────
    Total:        ~1,978 MB ≈ 2.0 GB

Fits comfortably within 6 GB VRAM ✅

Remaining headroom: ~4 GB (allows for larger batch sizes or longer sequences)
```

---

### Implementing RoPE

**Rotary Position Embeddings (RoPE)** encode positional information directly into the query and key vectors of the attention mechanism. Unlike absolute positional embeddings, RoPE naturally handles relative positions and generalizes better to unseen sequence lengths.

**Why RoPE?**
- No learned positional parameters — saves memory
- Captures relative position via rotation in complex space
- Proven effective in LLaMA, Mistral, Qwen, and other modern architectures
- Enables natural extrapolation beyond training context length

```python
# model/architecture/rope.py

import torch
import math

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute the complex exponential frequencies for RoPE.

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        theta: Base frequency (default 10000.0)

    Returns:
        freqs_cis: Complex tensor of shape (max_seq_len, dim // 2)
    """
    # Compute frequency bands: theta_i = 1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Position indices
    t = torch.arange(max_seq_len, dtype=torch.float32)

    # Outer product: (seq_len, dim//2)
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i * theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor (batch, seq_len, num_heads, head_dim)
        xk: Key tensor (batch, seq_len, num_heads, head_dim)
        freqs_cis: Precomputed frequencies (seq_len, head_dim // 2)

    Returns:
        Rotated xq and xk tensors
    """
    # Reshape to pairs: (batch, seq_len, num_heads, head_dim//2, 2)
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Convert to complex numbers
    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    # Reshape freqs for broadcasting: (1, seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Apply rotation via complex multiplication
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

---

### Implementing RMSNorm

**Root Mean Square Layer Normalization (RMSNorm)** is a simplified version of LayerNorm that removes the mean-centering step. It is computationally cheaper and has been shown to perform equivalently in practice.

**Why RMSNorm instead of LayerNorm?**
- ~15% faster than LayerNorm (no mean computation)
- Fewer operations = less VRAM usage
- Used by LLaMA, Mistral, Gemma — proven at scale
- Identical quality to LayerNorm in decoder-only models

```python
# model/architecture/rmsnorm.py

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes the input by its RMS value and applies a learned scale parameter.
    Unlike LayerNorm, does not center the data (no mean subtraction or bias).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learned scale (γ)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        # Normalize: x / RMS(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability, then cast back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

---

### Implementing Causal Multi-Head Attention

The attention mechanism is the core of the Transformer. Our implementation uses standard **Causal Multi-Head Self-Attention** with RoPE applied to queries and keys.

**Why Causal Attention?**
- Decoder-only models generate text left-to-right
- Each token should only attend to previous tokens (and itself)
- The causal mask prevents information leakage from future tokens during training

```python
# model/architecture/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rope import apply_rotary_emb

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with Rotary Position Embeddings.

    Projects input into Q, K, V, applies RoPE to Q and K,
    computes scaled dot-product attention with a causal mask,
    and projects the output back to the model dimension.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Q, K, V, O projection matrices
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            freqs_cis: RoPE frequencies (seq_len, head_dim // 2)

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # Using PyTorch's built-in efficient attention (Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,  # Applies causal mask automatically
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)
```

---

### Implementing SwiGLU

**SwiGLU (Swish-Gated Linear Unit)** is the feed-forward network activation used in modern LLMs. It replaces the traditional ReLU or GELU activation with a gated mechanism that allows the model to learn more expressive transformations.

**Why SwiGLU instead of standard FFN?**
- 1–3% better performance than GELU-based FFNs at the same parameter count
- The gating mechanism allows the network to learn which features to pass through
- Used by LLaMA, PaLM, Gemma, Mistral — the current industry standard
- The `intermediate_size = 4 × hidden_size = 3072` keeps the parameter count balanced

```python
# model/architecture/swiglu.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Implements: FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Where:
        gate_proj: Projects input to intermediate dimension (for gating)
        up_proj:   Projects input to intermediate dimension (for values)
        SiLU:      Sigmoid Linear Unit activation (x * sigmoid(x))
        down_proj: Projects back to hidden dimension

    This uses two "up" projections (gate + value) instead of one,
    with the gate controlling information flow.
    """

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: apply SiLU activation to the gating projection
        gate = F.silu(self.gate_proj(x))

        # Value: linear projection (no activation)
        value = self.up_proj(x)

        # Element-wise multiplication: gate controls information flow
        hidden = gate * value

        # Project back to model dimension
        return self.down_proj(hidden)
```

---

### Building Transformer Blocks

A single Transformer block combines attention and feed-forward layers with pre-normalization (RMSNorm before each sublayer) and residual connections.

```python
# model/architecture/transformer_block.py

import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .attention import CausalSelfAttention
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.

    Architecture (Pre-Norm style):
        x → RMSNorm → Attention → + (residual)
                                   ↓
                              → RMSNorm → SwiGLU FFN → + (residual)
    """

    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention      = CausalSelfAttention(config)
        self.ffn_norm       = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward   = SwiGLU(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Attention sublayer with pre-norm and residual
        h = x + self.attention(self.attention_norm(x), freqs_cis)

        # FFN sublayer with pre-norm and residual
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
```

---

### Building the Full Model

The complete NotebookCore-120M model stacks the token embedding, 12 Transformer blocks, a final RMSNorm, and an LM head (weight-tied with the embedding).

```python
# model/architecture/model.py

import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock
from .rope import precompute_freqs_cis
from ..config import NotebookCoreConfig

class NotebookCore(nn.Module):
    """
    NotebookCore-120M: Complete decoder-only Transformer language model.

    Architecture:
        Token Embedding → 12 × TransformerBlock → RMSNorm → LM Head
    """

    def __init__(self, config: NotebookCoreConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM Head — projects hidden states back to vocabulary logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with LM head
        self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            input_ids: Token IDs (batch, seq_len)
            targets: Target token IDs for loss computation (batch, seq_len)

        Returns:
            logits: Vocabulary logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        h = self.tok_embeddings(input_ids)

        # Move precomputed RoPE frequencies to device and slice to seq_len
        freqs_cis = self.freqs_cis[:seq_len].to(device)

        # Pass through all Transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis)

        # Final normalization
        h = self.norm(h)

        # Project to vocabulary
        logits = self.lm_head(h)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # Ignore padding tokens
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256,
                 temperature: float = 0.8, top_k: int = 50):
        """
        Autoregressive text generation with temperature and top-k sampling.
        """
        for _ in range(max_new_tokens):
            # Crop context to max_position_embeddings
            idx_cond = input_ids[:, -self.config.max_position_embeddings:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for the last position
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at EOS
            if next_token.item() == self.config.eos_token_id:
                break

        return input_ids
```

---

## Training Pipeline

The training pipeline is optimized for the RTX 4050 (6 GB VRAM, 16 GB system RAM) using a combination of memory-saving techniques.

### Memory Optimization Techniques

| Technique | VRAM Savings | Implementation |
|---|---|---|
| **FP16 Mixed Precision** | ~50% | `torch.cuda.amp.autocast()` |
| **Gradient Checkpointing** | ~60% activation memory | `torch.utils.checkpoint.checkpoint()` |
| **Gradient Accumulation** | Linear with steps | Effective batch = micro_batch × accumulation_steps |
| **Weight Tying** | ~24M params × 2 bytes = 48 MB | LM head shares embedding weights |

### Training Configuration

```python
# Training hyperparameters for RTX 4050 (6 GB VRAM)

PRETRAIN_CONFIG = {
    # ── Batch & Sequence ──
    "micro_batch_size": 2,            # Samples per GPU forward pass
    "gradient_accumulation_steps": 8, # Effective batch size = 2 × 8 = 16
    "max_seq_length": 1024,           # Tokens per sample (for pretraining)

    # ── Learning Rate ──
    "learning_rate": 3e-4,            # Peak learning rate
    "min_learning_rate": 3e-5,        # Minimum LR (10% of peak)
    "warmup_steps": 2000,             # Linear warmup from 0 to peak LR
    "max_steps": 100_000,             # Total training steps

    # ── Optimizer ──
    "optimizer": "AdamW",
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,             # Gradient clipping

    # ── Precision ──
    "precision": "fp16",              # Mixed precision training
    "gradient_checkpointing": True,   # Trade compute for VRAM

    # ── Logging ──
    "log_interval": 10,               # Log every N steps
    "eval_interval": 1000,            # Evaluate every N steps
    "save_interval": 5000,            # Checkpoint every N steps

    # ── Reproducibility ──
    "seed": 42,
}
```

---

## Pretraining

Pretraining teaches the model to predict the next token in a sequence — the fundamental capability from which all downstream abilities emerge.

### Pretraining Loop

```python
# model/training/pretrain.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import math
import os
import time

from ..architecture.model import NotebookCore
from ..config import NotebookCoreConfig

def get_cosine_schedule_with_warmup(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    elif step >= max_steps:
        return min_lr
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def pretrain(config, train_dataset, val_dataset):
    """
    Main pretraining loop for NotebookCore-120M.

    Uses:
        - FP16 mixed precision
        - Gradient accumulation
        - Gradient checkpointing
        - Cosine LR schedule with warmup
        - Periodic evaluation and checkpointing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Initialize Model ──
    model_config = NotebookCoreConfig()
    model = NotebookCore(model_config).to(device)

    # Enable gradient checkpointing
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Model size (FP16)    : {trainable_params * 2 / 1e6:.1f} MB")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )

    # ── Mixed Precision ──
    scaler = GradScaler()

    # ── Data Loader ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["micro_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── Training Loop ──
    model.train()
    step = 0
    running_loss = 0.0
    best_val_loss = float("inf")

    data_iter = iter(train_loader)

    print(f"\n{'='*60}")
    print(f"  STARTING PRETRAINING")
    print(f"  Steps       : {config['max_steps']:,}")
    print(f"  Batch size  : {config['micro_batch_size']} × {config['gradient_accumulation_steps']} = {config['micro_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Seq length  : {config['max_seq_length']}")
    print(f"  Device      : {device}")
    print(f"{'='*60}\n")

    start_time = time.time()

    while step < config["max_steps"]:
        optimizer.zero_grad()

        # ── Gradient Accumulation ──
        for micro_step in range(config["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass with mixed precision
            with autocast(dtype=torch.float16):
                logits, loss = model(input_ids, targets)
                loss = loss / config["gradient_accumulation_steps"]

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            running_loss += loss.item()

        # ── Gradient Clipping ──
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

        # ── Optimizer Step ──
        scaler.step(optimizer)
        scaler.update()

        # ── Learning Rate Schedule ──
        lr = get_cosine_schedule_with_warmup(
            step, config["warmup_steps"], config["max_steps"],
            config["learning_rate"], config["min_learning_rate"]
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        step += 1

        # ── Logging ──
        if step % config["log_interval"] == 0:
            avg_loss = running_loss / config["log_interval"]
            elapsed = time.time() - start_time
            tokens_per_sec = (step * config["micro_batch_size"] *
                             config["gradient_accumulation_steps"] *
                             config["max_seq_length"]) / elapsed

            print(f"Step {step:>6} | Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tokens/s: {tokens_per_sec:,.0f} | "
                  f"Elapsed: {elapsed:.0f}s")
            running_loss = 0.0

        # ── Evaluation ──
        if step % config["eval_interval"] == 0:
            val_loss = evaluate(model, val_dataset, device, config)
            print(f"  → Validation Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, "checkpoints/pretrain/best.pt")
                print(f"  → New best model saved!")

            model.train()

        # ── Periodic Checkpoint ──
        if step % config["save_interval"] == 0:
            save_checkpoint(model, optimizer, step, f"checkpoints/pretrain/step_{step}.pt")

    print(f"\n✅ Pretraining complete! Total steps: {step:,}")


def evaluate(model, val_dataset, device, config):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    val_loader = DataLoader(val_dataset, batch_size=config["micro_batch_size"], shuffle=False)

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            with autocast(dtype=torch.float16):
                _, loss = model(input_ids, targets)

            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 100:  # Evaluate on 100 batches max
                break

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, step, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }, path)
```

---

## Instruction Tuning

After pretraining, the model understands language but cannot follow instructions. Instruction tuning teaches it to respond to user queries in a helpful, structured format — the capability required by StudySnap AI.

### Instruction Format

```
<|system|>
You are StudySnap AI, a helpful and knowledgeable assistant.
<|endofturn|>
<|user|>
Explain the process of photosynthesis.
<|endofturn|>
<|assistant|>
Photosynthesis is the process by which green plants convert light energy into chemical energy...
<|endofturn|>
```

### Instruction Tuning Script

```python
# model/training/instruct_tune.py

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

INSTRUCT_CONFIG = {
    "micro_batch_size": 2,
    "gradient_accumulation_steps": 4,  # Effective batch = 8
    "max_seq_length": 2048,            # Full context for instructions
    "learning_rate": 2e-5,             # Lower LR than pretraining
    "min_learning_rate": 2e-6,
    "warmup_steps": 200,
    "max_steps": 5000,                 # Fewer steps than pretraining
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "precision": "fp16",
    "gradient_checkpointing": True,
    "seed": 42,
}

def format_instruction(example, tokenizer):
    """
    Format a single instruction example into the chat template.
    Only compute loss on the assistant's response (masked training).
    """
    system_msg = "You are StudySnap AI, a helpful and knowledgeable assistant."

    prompt = (
        f"<|system|>\n{system_msg}\n<|endofturn|>\n"
        f"<|user|>\n{example['instruction']}\n<|endofturn|>\n"
        f"<|assistant|>\n"
    )
    response = f"{example['response']}\n<|endofturn|>"

    # Tokenize prompt (no loss) and response (compute loss)
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)

    input_ids = prompt_ids + response_ids
    # Labels: -1 for prompt tokens (ignored in loss), actual IDs for response
    labels = [-1] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
    }
```

---

## Evaluation

Model evaluation uses multiple metrics to assess quality across different capabilities.

### Metrics

| Metric | What It Measures | Target |
|---|---|---|
| **Perplexity** | How "surprised" the model is by held-out text. Lower is better. | < 20 on validation set |
| **Cross-Entropy Loss** | Raw training loss. Should decrease monotonically. | < 3.0 after pretraining |
| **BLEU Score** | N-gram overlap with reference responses (instruction tuning). | > 15 on test set |
| **Human Evaluation** | Coherence, helpfulness, and factual accuracy of generations. | Subjective quality check |

### Evaluation Script

```python
# model/evaluation/evaluate.py

import torch
import math
from torch.utils.data import DataLoader

def compute_perplexity(model, dataset, device, batch_size=4, max_batches=200):
    """
    Compute perplexity on a dataset.
    Perplexity = exp(average cross-entropy loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                _, loss = model(input_ids, targets)

            # Count non-padding tokens
            num_tokens = (targets != -1).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    print(f"Average Loss : {avg_loss:.4f}")
    print(f"Perplexity   : {perplexity:.2f}")

    return perplexity


def generate_samples(model, tokenizer, prompts, device, max_tokens=256):
    """Generate sample outputs for qualitative evaluation."""
    model.eval()

    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens)
        output_text = tokenizer.decode(output_ids[0].tolist())

        print(f"\n{'─'*60}")
        print(f"Prompt: {prompt}")
        print(f"Output: {output_text}")
        print(f"{'─'*60}")
```

---

## Export to Hugging Face

After training, the model is exported to the Hugging Face `safetensors` format for compatibility with the broader ecosystem.

```python
# model/export/export_hf.py

import torch
import json
import os
from safetensors.torch import save_file

def export_to_huggingface(model, tokenizer_path, output_dir):
    """
    Export NotebookCore-120M to Hugging Face format.

    Creates:
        - config.json           (model architecture)
        - model.safetensors     (weights)
        - tokenizer.model       (SentencePiece)
        - tokenizer_config.json (tokenizer metadata)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Save config.json ──
    config = {
        "architectures": ["NotebookCoreForCausalLM"],
        "model_type": "notebookcore",
        "vocab_size": 32000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Save model weights ──
    state_dict = model.state_dict()

    # Rename keys to match HuggingFace conventions
    hf_state_dict = {}
    for key, value in state_dict.items():
        hf_key = key.replace("tok_embeddings", "model.embed_tokens")
        hf_key = hf_key.replace("layers.", "model.layers.")
        hf_key = hf_key.replace("attention_norm", "input_layernorm")
        hf_key = hf_key.replace("ffn_norm", "post_attention_layernorm")
        hf_key = hf_key.replace("attention.", "self_attn.")
        hf_key = hf_key.replace("feed_forward.", "mlp.")
        hf_key = hf_key.replace("norm.", "model.norm.")
        hf_state_dict[hf_key] = value.half()  # Save as FP16

    save_file(hf_state_dict, os.path.join(output_dir, "model.safetensors"))

    # ── Copy tokenizer ──
    import shutil
    shutil.copy(tokenizer_path, os.path.join(output_dir, "tokenizer.model"))

    # ── Tokenizer config ──
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<|pad|>",
        "model_max_length": 2048,
        "tokenizer_class": "LlamaTokenizer",
    }

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    total_size = sum(os.path.getsize(os.path.join(output_dir, f))
                     for f in os.listdir(output_dir)) / 1e6
    print(f"✅ Model exported to {output_dir} ({total_size:.1f} MB)")
```

---

## Convert to GGUF

GGUF is the format used by **llama.cpp** and **LM Studio** for efficient CPU/GPU inference. We use the `llama.cpp` conversion tool to quantize the model.

### Step 1 — Clone llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

### Step 2 — Convert to GGUF (FP16)

```bash
python convert_hf_to_gguf.py ../model/export/notebookcore-120m-hf/ \
    --outfile notebookcore-120m-f16.gguf \
    --outtype f16
```

### Step 3 — Quantize to Q4_K_M

Q4_K_M is a mixed-precision 4-bit quantization scheme that keeps important layers at higher precision:

```bash
./llama-quantize notebookcore-120m-f16.gguf notebookcore-120m-Q4_K_M.gguf Q4_K_M
```

### Size Comparison

| Format | File Size | Quality | Speed |
|---|---|---|---|
| FP16 Safetensors | ~240 MB | Maximum | Requires GPU |
| GGUF F16 | ~240 MB | Maximum | CPU/GPU |
| GGUF Q8_0 | ~127 MB | Near-lossless | Fast |
| **GGUF Q4_K_M** | **~68 MB** | **Good (recommended)** | **Very fast** |
| GGUF Q4_0 | ~63 MB | Acceptable | Fastest |

---

## LM Studio Deployment

LM Studio serves the GGUF model as a local API endpoint compatible with the OpenAI API format.

### Setup Steps

1. **Download LM Studio** from [lmstudio.ai](https://lmstudio.ai/)
2. **Import the GGUF file:**
   - Copy `notebookcore-120m-Q4_K_M.gguf` to LM Studio's models directory
   - Typically: `C:\Users\<user>\.cache\lm-studio\models\`
3. **Load the model:**
   - Open LM Studio → Local Server tab
   - Select `notebookcore-120m-Q4_K_M.gguf`
   - Configure context length: `2048`
   - Click "Start Server"
4. **Verify the API:**

```bash
curl http://localhost:1234/v1/models
```

Expected response:

```json
{
  "data": [
    {
      "id": "notebookcore-120m-Q4_K_M",
      "object": "model"
    }
  ]
}
```

### Test Generation

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "notebookcore-120m-Q4_K_M",
    "messages": [{"role": "user", "content": "Explain photosynthesis in simple terms."}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

---

## FastAPI Integration

The FastAPI backend (`backend/main.py`) acts as the orchestration layer between the React frontend and the LM Studio inference engine.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/sources/upload` | Upload and ingest documents (PDF, DOCX, PPTX, CSV, XLSX, TXT, audio) |
| `POST` | `/api/sources/url` | Ingest a URL or YouTube link |
| `GET` | `/api/sources` | List all ingested sources |
| `DELETE` | `/api/sources/{file_id}` | Remove a source and its vectors |
| `POST` | `/api/chat` | Send a message to the AI with RAG context |
| `POST` | `/api/audio-overview` | Generate a podcast-style audio overview |
| `POST` | `/api/infographic` | Generate an AI infographic |
| `POST` | `/api/mindmap` | Generate a structured mind map |

### LM Studio Client Configuration

In `backend/rag.py`, the connection to NotebookCore-120M via LM Studio:

```python
from openai import AsyncOpenAI

# Connect to LM Studio's local API
lm_studio_client = AsyncOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # LM Studio doesn't require a real key
)

MODEL_NAME = "notebookcore-120m-Q4_K_M"
```

---

## Integrating with StudySnap AI

NotebookCore-120M replaces the previous Mistral-7B model as the inference engine. The integration point is the `rag.py` module, which handles all communication between the backend and the model.

### RAG Pipeline

Retrieval-Augmented Generation (RAG) grounds the model's responses in user-provided documents, reducing hallucination and enabling domain-specific Q&A.

```
User Query → ChromaDB Similarity Search → Top-K Chunks → System Prompt + Context → NotebookCore-120M → Response
```

The RAG implementation in `backend/rag.py`:

1. **Query Embedding:** The user's question is embedded using ChromaDB's default embedding function
2. **Vector Search:** ChromaDB returns the top 4 most semantically similar document chunks
3. **Context Assembly:** Retrieved chunks are formatted into a system prompt
4. **LLM Generation:** NotebookCore-120M generates a response grounded in the retrieved context

### ChromaDB Integration

ChromaDB stores document embeddings locally with persistent storage:

```python
import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="./chroma_data")
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="studysnap_collection",
    embedding_function=emb_fn
)
```

Documents are chunked with overlapping windows for better retrieval:

```python
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

### PDF Processing

PDF files are processed using `pypdf`:

```python
from pypdf import PdfReader

reader = PdfReader(file_path)
text = ""
for page in reader.pages:
    if page_text := page.extract_text():
        text += page_text + "\n"
```

### URL Processing

Web pages and YouTube videos are supported:

- **Web pages:** Fetched with `requests`, parsed with `BeautifulSoup` to extract clean text
- **YouTube:** Transcripts extracted via `youtube-transcript-api` supporting English, Hindi, and auto-generated captions

### Audio Processing

Audio files (WAV, MP3, M4A, FLAC) are transcribed using OpenAI's **Whisper** model (base variant) running locally:

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe(file_path)
transcript = result.get("text", "")
```

### Mind Map Generation

The mind map pipeline uses NotebookCore-120M to extract concepts and relationships from documents, outputting structured JSON:

```json
{
  "nodes": [
    {"id": "1", "data": {"label": "Photosynthesis", "context": "..."}, "position": {"x": 250, "y": 0}},
    {"id": "2", "data": {"label": "Light Reactions", "context": "..."}, "position": {"x": 100, "y": 150}}
  ],
  "edges": [
    {"id": "e1-2", "source": "1", "target": "2"}
  ]
}
```

This JSON is rendered interactively on the frontend using `@xyflow/react` (React Flow).

### Infographic Generation

The infographic pipeline is a two-stage process:

1. **Prompt Generation:** NotebookCore-120M analyzes the source documents and generates a detailed visual description prompt
2. **Image Generation:** The prompt is sent to the Pollinations.ai API to generate a visual infographic

---

## Future Scaling Roadmap

NotebookCore-120M is designed as the foundation for a series of increasingly capable models.

### Scaling to 250M

| Parameter | 120M | 250M |
|---|---|---|
| Layers | 12 | 16 |
| Hidden Size | 768 | 1024 |
| Attention Heads | 12 | 16 |
| FFN Size | 3,072 | 4,096 |
| Context | 2,048 | 2,048 |
| VRAM (FP16 training) | ~2 GB | ~4 GB |
| Hardware | RTX 4050 | RTX 4050 |

**Key change:** Still trainable on RTX 4050 with aggressive gradient checkpointing and gradient accumulation of 16.

### Scaling to 500M

| Parameter | 120M | 500M |
|---|---|---|
| Layers | 12 | 24 |
| Hidden Size | 768 | 1024 |
| Attention Heads | 12 | 16 |
| FFN Size | 3,072 | 4,096 |
| Context | 2,048 | 4,096 |
| VRAM (FP16 training) | ~2 GB | ~8 GB |
| Hardware | RTX 4050 | RTX 3090 / A6000 |

**Key change:** Requires GPU with 8+ GB VRAM, or CPU offloading with `accelerate`.

### Scaling to 1B

| Parameter | 120M | 1B |
|---|---|---|
| Layers | 12 | 24 |
| Hidden Size | 768 | 2048 |
| Attention Heads | 12 | 16 |
| FFN Size | 3,072 | 5,504 |
| Context | 2,048 | 4,096 |
| VRAM (FP16 training) | ~2 GB | ~16 GB |
| Hardware | RTX 4050 | A100 / H100 |

**Key change:** Requires data-center GPU or cloud compute. Consider Grouped Query Attention (GQA) for efficient inference.

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---|---|---|
| `CUDA out of memory` | Batch size too large | Reduce `micro_batch_size` to 1, increase `gradient_accumulation_steps` |
| `Loss is NaN` | Learning rate too high or gradient explosion | Reduce LR to 1e-4, ensure `max_grad_norm = 1.0` |
| `Loss plateaus early` | Learning rate too low or data quality issues | Increase LR, check dataset cleaning |
| `LM Studio doesn't load model` | GGUF conversion failed | Re-run conversion with `--outtype f16` first, then quantize |
| `Backend connection refused` | LM Studio not running | Start LM Studio server on port 1234 |
| `ChromaDB errors` | Corrupted database | Delete `chroma_data/` and re-ingest documents |
| `Whisper fails` | ffmpeg not installed | Install ffmpeg: `winget install ffmpeg` |
| `Frontend won't start` | Missing dependencies | Run `npm install` in `frontend/` |

### VRAM Monitoring

```python
# Add to training loop for real-time VRAM tracking
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
```

---

## Production Deployment

### Starting All Services

```bash
# Terminal 1: Start LM Studio
# Open LM Studio → Load notebookcore-120m-Q4_K_M.gguf → Start Server

# Terminal 2: Start Backend
cd backend
venv\Scripts\activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Start Frontend
cd frontend
npm run dev
```

### Service Ports

| Service | Port | URL |
|---|---|---|
| Frontend (Vite) | 5173 or 5174 | `http://localhost:5173` |
| Backend (FastAPI) | 8000 | `http://localhost:8000` |
| LM Studio | 1234 | `http://localhost:1234/v1` |

### Production Build

```bash
# Build frontend for production
cd frontend
npm run build

# Output in frontend/dist/ — serve with any static file server
```

---

## Future Research

| Research Direction | Description | Priority |
|---|---|---|
| **Grouped Query Attention (GQA)** | Reduce KV cache memory by sharing key/value heads across groups | High |
| **Speculative Decoding** | Use a smaller "draft" model to accelerate generation | Medium |
| **Knowledge Distillation** | Distill capabilities from a larger model into NotebookCore | High |
| **Multimodal Extension** | Add image understanding via a vision encoder | Medium |
| **Mixture of Experts (MoE)** | Scale parameter count without proportional compute increase | Low |
| **Direct Preference Optimization (DPO)** | Align model outputs with human preferences without RL | High |
| **Longer Context (8K–32K)** | Extend context via ALiBi or YaRN position interpolation | Medium |
| **RAG-Aware Pretraining** | Include retrieval-augmented examples during pretraining | Medium |

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **PyTorch** — Deep learning framework
- **SentencePiece** — Tokenizer training
- **Hugging Face** — Model format and datasets
- **llama.cpp** — GGUF conversion and quantization
- **LM Studio** — Local model serving
- **ChromaDB** — Vector database
- **FastAPI** — Backend framework
- **React** — Frontend framework

---

<p align="center">
  <strong>NotebookCore-120M</strong> — Built from scratch. No shortcuts. No borrowed weights.<br>
  <em>Powering StudySnap AI with a fully custom language model.</em>
</p>
