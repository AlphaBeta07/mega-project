<p align="center">
  <img src="https://img.shields.io/badge/StudySnap-AI-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Frontend-React%2019-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-8B%20%7C%203.8B-orange?style=for-the-badge" />
</p>

# StudySnap AI — Complete Project Documentation

**StudySnap AI** is an intelligent, full-stack application designed to enhance learning and productivity. It allows users to ingest various sources (documents, URLs), and leverages custom-trained AI models to perform Retrieval-Augmented Generation (RAG), synthesize audio overviews, generate visual infographics, and create structured mind maps.

---

## 1. Core Features

- **Document & Source Ingestion:** Upload documents or provide URLs/YouTube links. These are processed, vectorized, and stored in a local ChromaDB database.
- **Context-Aware Chat (RAG):** Chat directly with the AI using context pulled from ingested sources.
- **Audio Overviews:** Convert selected sources into engaging, podcast-style audio scripts and synthesize them into playable audio using TTS (Text-to-Speech) via Whisper/Edge TTS.
- **Infographic Generation:** Transform complex text into visually structured infographics (e.g., Bento Grid style) customized by detail level, powered by **ComfyUI** (Stable Diffusion).
- **Mind Map Generation:** Extract key concepts and relationships, converted into structured JSON, and rendered interactively on the frontend using React Flow.
- **Multi-language Support:** Dynamic output generation in various languages.

---

## 2. System Architecture

The complete system is composed of four major layers:

```text
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
│   ┌─────────────────────────────────┐ ┌──────────────────────────────────┐   │
│   │  LM Studio (Text Generation)    │ │  ComfyUI (Image Generation)      │   │
│   │  studysanpai (8B) or            │ │  Stable Diffusion 1.5 Workflows  │   │
│   │  educationalassistant (3.8B)    │ │                                  │   │
│   └─────────────────────────────────┘ └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Technology | Purpose |
|---|---|---|
| **Presentation** | React 19, TypeScript, Vite, Framer Motion, React Flow | User interface — chat, source management, studio tools |
| **Application** | FastAPI, Uvicorn, Pydantic | API routing, request validation, orchestration |
| **RAG / Data** | ChromaDB, PyPDF, Whisper, BeautifulSoup, youtube-transcript-api | Document ingestion, chunking, embedding, retrieval |
| **Inference** | LM Studio, llama.cpp, GGUF, ComfyUI | Model serving for LLM text and Stable Diffusion image generation |

---

## 3. Folder Structure

```text
mega_project/
│
├── README.md                          # This file
├── PROJECT_DOCUMENTATION.md           # High-level architecture overview
├── main-readme.md                     # studysanpai (8B) Model Documentation
├── new run.md                         # educationalassistant (3.8B) Model Training Pipeline
│
├── model/                             # ── MODEL TRAINING PIPELINE ──
│   ├── config.py                      # Model hyperparameters
│   ├── tokenizer/                     # Custom BPE tokenizer training
│   ├── architecture/                  # RoPE, RMSNorm, Attention, SwiGLU implementations
│   ├── data/                          # Dataset download, clean, and formatting
│   ├── training/                      # Pretraining and instruction tuning loops
│   ├── evaluation/                    # Benchmarking and model metrics
│   ├── export/                        # HF safetensors and GGUF conversions
│   └── checkpoints/                   # Saved model states
│
├── backend/                           # ── FASTAPI BACKEND ──
│   ├── main.py                        # API gateway — all endpoints
│   ├── rag.py                         # RAG pipeline — ingestion, retrieval, generation
│   ├── comfy_client.py                # ComfyUI client for image generation
│   ├── requirements.txt               # Python dependencies
│   ├── schemas/                       # Pydantic models
│   ├── uploads/                       # User-uploaded files
│   ├── chroma_data/                   # ChromaDB persistent storage
│   └── venv/                          # Python virtual environment
│
└── frontend/                          # ── REACT FRONTEND ──
    ├── index.html                     # Entry point
    ├── package.json                   # Dependencies (React 19, Vite, etc.)
    ├── vite.config.ts                 # Vite configuration
    ├── tsconfig.json                  # TypeScript config
    └── src/
        ├── main.tsx                   # React entry
        ├── App.tsx                    # Main application shell + chat interface
        ├── App.css                    # Application styles
        ├── index.css                  # Global design system
        ├── AddSourceModal.tsx         # File upload + URL ingestion modal
        ├── AudioOverviewModal.tsx     # Podcast-style audio generation
        ├── InfographicModal.tsx       # AI infographic generation
        ├── InfographicEditor/         # Infographic editing tools and renderer
        ├── MindMapModal.tsx           # Mind map prompt interface
        └── MindMapRenderer.tsx        # Interactive React Flow mind map
```

---

## 4. AI Models Integration

StudySnap AI supports two primary language models, each tailored for specific deployment and research needs:

### A. Production Engine: Custom 7B Model (Mistral LoRA)
For robust, high-quality answers, we fine-tuned a 7B model locally on an NVIDIA RTX 4050 (6 GB VRAM).
- **Base Model:** `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- **Dataset:** `yahma/alpaca-cleaned`
- **Training Setup:** LoRA (Low-Rank Adaptation) using a **Hybrid GPU + CPU RAM Strategy**, 4-bit NF4 Quantization, and Gradient Checkpointing to fit within strict VRAM constraints (5.5GB).
- **Export:** Converted to GGUF (Q4_K_M) for serving via LM Studio.

### B. Educational Engine: NotebookCore-120M
For deep architectural understanding, we built a 120-million parameter decoder-only transformer **entirely from scratch**.
- **Architecture:** 12-layer Llama-style decoder (RoPE, RMSNorm, SwiGLU).
- **Pretraining:** Next-token prediction on ~8B tokens (Wikipedia, Project Gutenberg, OpenWebText).
- **Instruction Tuning:** SFT on curated QA and summarization data (OpenAssistant, Dolly, SQuAD v2).
- **Tokenizer:** Custom BPE tokenizer (SentencePiece) optimized for educational text.

---

## 5. Environment Setup & Prerequisites

| Software | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Model training, backend server |
| Node.js | 18+ | Frontend build and dev server |
| CUDA Toolkit | 12.1+ | GPU acceleration for PyTorch |
| Git | Latest | Version control |
| LM Studio | Latest | Local model serving (Text generation) |
| ComfyUI | Latest | Local image generation for infographics |
| ffmpeg | Latest | Audio processing (Whisper) |

### Step 1: Clone and CUDA Check
```bash
git clone <your-repo-url>
cd mega_project
nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Backend Setup
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Frontend Setup
```bash
cd frontend
npm install
```

---

## 6. Execution Pipeline

### Starting StudySnap AI Services

You will need **3 separate terminals** (and ComfyUI running in the background for images):

```bash
# Terminal 1: Start LM Studio (Text Generation)
# Open LM Studio -> Local Server -> Load your GGUF model -> Start Server on port 1234

# Terminal 2: Backend FastAPI Server
cd backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Frontend React App
cd frontend
npm run dev
```

Navigate to `http://localhost:5173` to access the StudySnap AI dashboard. Verify backend endpoints at `http://localhost:8000/docs`.

### Training the 120M Model (Optional)
If you wish to run the custom 120M training pipeline from scratch:
```bash
# Full execution script
python -X utf8 run_pipeline.py
```
For granular control over training steps (e.g., tokenization, pretraining, instruct-tuning, export), see detailed commands in [new run.md](new run.md).
