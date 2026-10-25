<p align="center">
  <img src="https://img.shields.io/badge/StudySnap-AI-blueviolet?style=for-the-badge&logo=openai" alt="StudySnap AI" />
  <img src="https://img.shields.io/badge/Frontend-React%2019%20%7C%20TypeScript%20%7C%20Vite-61DAFB?style=for-the-badge&logo=react" alt="React 19" />
  <img src="https://img.shields.io/badge/Database-Supabase%20PostgreSQL%20%2B%20RLS-3ECF8E?style=for-the-badge&logo=supabase" alt="Supabase" />
  <img src="https://img.shields.io/badge/AI%20Engine-FastAPI%20%7C%20Python%203.12-009688?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-FF6F00?style=for-the-badge" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/LLM%20Engine-LM%20Studio%20%7C%20GGUF-8A2BE2?style=for-the-badge" alt="LM Studio" />
  <img src="https://img.shields.io/badge/Audio-Edge%20TTS%20%7C%20Whisper-0078D4?style=for-the-badge" alt="Edge TTS" />
  <img src="https://img.shields.io/badge/Tests-31%20Passing%20(100%25)-brightgreen?style=for-the-badge" alt="Tests" />
</p>

# 🎓 StudySnap AI — Full-Stack Academic Study Assistant & Lecture Intelligence Platform

**StudySnap AI** is a production-grade, privacy-first academic study platform engineered for students, researchers, and educators. It pairs **local, private LLM inference** with **Retrieval-Augmented Generation (RAG)**, transforming multi-format course materials into interactive study experiences without recurring API subscription costs or cloud data leakage.

The platform ingests lecture recordings, PDFs, slides, syllabus documents, and YouTube videos, then synthesizes them into:
- 🎙️ **2-Minute Podcast-Style Audio Overviews** (multi-host dialogues via Edge TTS & Whisper)
- 🗺️ **Interactive Mind Maps** (concept hierarchies rendered dynamically with React Flow)
- 📊 **Visual Bento Grid Infographics** (customizable detail levels via ComfyUI Stable Diffusion)
- 📝 **Live Student Notebook Workspace** (rich Markdown editor with 1200ms debounced autosave to Supabase)
- 🎯 **AI-Generated Quizzes & 3D Flip Flashcards** (instant grading and spaced repetition mastery)
- 📈 **PYQ & Exam Predictive Analysis** (topic frequency distribution from previous year questions)

> [!NOTE]
> **Database & Architecture:** StudySnap AI uses **Supabase** (PostgreSQL, Auth, Storage, and Row Level Security) as its primary application database, paired with **ChromaDB** for local vector embeddings. A thread-safe local fallback store ensures all features remain fully functional in offline/disconnected environments.

---

## 🤖 Custom AI Model & Local Inference

StudySnap AI is optimized to run fully offline using custom quantized GGUF models loaded directly into **LM Studio** on port `1234`.

### Primary Custom Model (`EducationalAssistant`)

The platform's primary reasoning engine is a custom fine-tuned LLaMA-based educational model exported in **GGUF Q4_K_M** format, tailored specifically for academic explanations, lecture summarization, RAG synthesis, and exam question formulation.

| Property | Value / Specification |
|:---|:---|
| **Model Identifier** | `my_custom_model_gguf/Anish/EducationalAssistant` |
| **GGUF File** | `my_custom_model.Q4_K_M.gguf` |
| **Format** | GGUF |
| **Quantization** | `Q4_K_M` (4-bit Medium Quantization) |
| **Architecture** | `llama` |
| **Domain** | `llm` (Educational Assistant / Academic Intelligence) |
| **Size on Disk** | **2.32 GB** |
| **Inference Engine** | LM Studio (OpenAI-compatible local server on port `1234`) |
| **Model Alias in Backend** | `MODEL_NAME=educationalassistant` |
| **VRAM Requirement** | 4 GB – 6 GB (runs smoothly on RTX 3050/4050 laptops or CPU) |

> [!IMPORTANT]
> ### 📥 Download Model Weights & GGUF Files
> **Google Drive Repository:**  
> 🔗 **[Download Model Weights & GGUF Files (Google Drive)](https://drive.google.com/drive/folders/1t8rqnwpjoVUJValpjHrmMkZs_c5lO6IT?usp=sharing)**
> 
> 1. Download `my_custom_model.Q4_K_M.gguf`.
> 2. Open **LM Studio** → Go to **My Models** (or drag & drop the file).
> 3. Navigate to **Local Server** (↔ icon) → Load `my_custom_model.Q4_K_M.gguf`.
> 4. Set identifier / alias to `educationalassistant`.
> 5. Click **Start Server** on port `1234`.

### Additional Supported Models

- **Mistral 7B Instruct LoRA (`studysanpai`):** 7-billion parameter adapter fine-tuned on the cleaned Alpaca educational instruction dataset using hybrid GPU/CPU memory allocation.
- **NotebookCore-120M:** A decoder-only transformer trained from scratch (RoPE positional embeddings, RMSNorm, SwiGLU feed-forward networks, and custom BPE tokenizer) with a complete 10-stage pretraining and instruction-tuning pipeline.

---

## ✨ Key Features & Study Modules

### 1. 📚 Multimodal Document & Lecture Ingestion
- **Document Formats:** Upload PDF, DOCX, PPTX slides, CSV, XLSX, TXT, and Markdown files.
- **YouTube Transcripts:** Paste any YouTube lecture URL to automatically extract, chunk, and index full video transcripts.
- **Web Research:** Ingest public web articles and documentation (hardened with server-side SSRF validation against loopback and internal cloud metadata addresses).
- **Audio Lectures:** Ingest recorded lecture audio (MP3, WAV, OGG) with automatic ASR transcription via Whisper.

### 2. 🔍 Context-Grounded RAG Chat with Verifiable Citations
- **ChromaDB Vector Store:** Chunks documents with configurable overlap and indexes text embeddings via `all-MiniLM-L6-v2`.
- **Exact Source Citations:** AI answers provide source badges displaying the originating document, exact page number, and verifiable excerpt.
- **Local SLM/LLM Normalizer:** Integrated `preprocessMarkdown` pipeline automatically cleans malformed `#Title##`, camelCase headers, and list tokens from local small language models.

### 3. 🎙️ 2-Minute Podcast Audio Overviews
- Generates a natural dialogue script between two AI hosts discussing the core ideas of uploaded materials.
- High-fidelity voice synthesis powered by **Microsoft Edge TTS** (`en-US-GuyNeural` & `en-US-JennyNeural`).
- In-browser playback with speed adjustment and seek controls.

### 4. 🗺️ Interactive Mind Maps
- Extracts concept taxonomies and hierarchical relationships into structured JSON.
- Dynamically rendered on the frontend using **`@xyflow/react` (React Flow)** with pan, zoom, auto-layout, and interactive node expansion.

### 5. 📊 Bento Grid Visual Infographics
- Condenses lecture notes into visually structured Bento Grid infographics.
- Integrated with local **ComfyUI** Stable Diffusion pipelines for visual styling and diagram generation.
- Three selectable depth levels: *Summary*, *Detailed*, and *Comprehensive*.

### 6. 📝 Student Studio & Debounced Autosave
- Unified workspace featuring a split-pane layout: sources, AI chat, and student notes.
- **1200ms debounced autosave** directly to Supabase PostgreSQL with visual saving states (`Cloud Saving...`, `Saved`, `Error`).
- Full Markdown editing toolbar (Bold, Italic, Code, Link, Image, Lists) and export options.

### 7. 🎯 AI Quizzes & 3D Flip Flashcards
- **Quiz Generator:** Dynamic multiple-choice questions with real-time score tracking, instant answer validation, and explanatory feedback.
- **Flashcards:** Interactive 3D flip card decks with spaced repetition mastery status (Easy, Medium, Hard).

### 8. 📈 Academic PYQ & Syllabus Analytics
- Ingest Previous Year Question (PYQ) papers to analyze exam trends.
- Automated topic frequency extraction, exam prediction, and syllabus coverage heatmaps.

### 9. 🎨 VS Code–Inspired Theme Engine
- Curated color themes inspired by popular code editors: **One Dark Pro**, **Dracula**, **GitHub Dark/Light**, **Tokyo Night**, **Monokai**, **SynthWave '84**, **Nord**, and **Cyberpunk**.
- Full-page canvas-based **interactive particle background** with mouse-gravity physics.

### 10. 🔐 Supabase Authentication & PostgreSQL RLS
- Complete authentication flow: Registration, Login, Demo Guest Access, Session Auto-Refresh, and Password Recovery.
- PostgreSQL **Row Level Security (RLS)** policies (`auth.uid() = owner_id`) ensure strict cross-user data isolation.
- Offline resilience store ensures zero downtime during local development or network outages.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  STUDYSNAP FRONTEND                                     │
│   React 19 • TypeScript • Vite • Lucide Icons • React Flow • Framer Motion              │
│   - Supabase Auth Integration (Login, Register, Demo, Password Recovery)                │
│   - Student Dashboard (Notebook History, Creation, Renaming, Statistics)                │
│   - Studio Workspace (Debounced Autosave, RAG Studio, Audio, MindMap, Infographics)     │
│   - Multi-Theme Engine & Canvas Particle Physics                                        │
└────────────────────────────┬───────────────────────────────┬────────────────────────────┘
                             │                               │
                             ▼ (Supabase JS + RLS)           ▼ (Bearer <Supabase JWT>)
┌────────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
│           SUPABASE CLOUD / LOCAL           │ │       FASTAPI AI ENGINE (Port 8000)      │
│   - Supabase Auth (JWT Provider)           │ │  - Supabase JWT Token Auth Verification  │
│   - PostgreSQL Database                    │ │  - Multi-Format Document Parsing         │
│     (profiles, notebooks, documents,       │ │  - ChromaDB Vector Store & Semantic RAG  │
│      notes, chat_sessions, messages)       │ │  - SSRF URL Validation & Safe Ingestion  │
│   - Row Level Security (RLS Policies)      │ │  - Edge TTS Audio Dialogue Generation    │
│   - Storage Bucket: studysnap-assets       │ │  - Resilient Local Fallback Store        │
└────────────────────────────────────────────┘ └───────────────────┬──────────────────────┘
                                                                   │
                                     ┌─────────────────────────────┴──────────────────────┐
                                     ▼                                                    ▼
                   ┌───────────────────────────────────┐               ┌───────────────────────────────────┐
                   │        LM STUDIO (Port 1234)      │               │        COMFYUI (Port 8188)        │
                   │  Model: my_custom_model.Q4_K_M    │               │  - Stable Diffusion Workflows     │
                   │  Arch: llama (2.32 GB GGUF)       │               │  - Visual Bento Grid Rendering    │
                   │  Endpoint: /v1/chat/completions   │               └───────────────────────────────────┘
                   └───────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
mega_project/
├── backend/                        # FastAPI AI & RAG Backend
│   ├── database/                   # Supabase client, models, and CRUD operations
│   │   ├── connection.py           # Supabase connection manager & RLS token propagation
│   │   ├── crud.py                 # Notebook, document, note, and chat persistence
│   │   ├── models.py               # Pydantic data schemas
│   │   └── config.py               # Database configuration settings
│   ├── tests/                      # Automated test suite (31 tests passing)
│   │   ├── test_api.py             # Health, SSRF validation, ChromaDB, and sources
│   │   ├── test_notebooks.py       # Notebook CRUD and autosave lifecycle
│   │   ├── test_prompt_registry.py # Prompt template and SLM output sanitizer
│   │   ├── test_student_profile.py # User profile, avatar storage, and isolation
│   │   └── test_supabase_features.py # JWT validation and PostgreSQL RLS isolation
│   ├── auth.py                     # Supabase JWT token verification
│   ├── comfy_client.py             # Local ComfyUI API client for infographics
│   ├── main.py                     # FastAPI routes, middleware, and lifespan handlers
│   ├── prompt_registry.py          # Academic prompt templates & output cleaners
│   ├── rag.py                      # ChromaDB embeddings, chunking, and LM Studio client
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Container definition for FastAPI engine
│
├── frontend/                       # React 19 + TypeScript + Vite Frontend
│   ├── src/
│   │   ├── components/             # UI views and interactive modules
│   │   │   ├── AudioOverviewPage.tsx # Multi-speaker podcast audio player
│   │   │   ├── AuthModal.tsx       # Supabase auth (login, signup, guest demo)
│   │   │   ├── FlashcardsPage.tsx  # 3D animated flip study flashcards
│   │   │   ├── LandingPage.tsx     # Public community notes and platform showcase
│   │   │   ├── MindMapPage.tsx     # React Flow interactive hierarchy graph
│   │   │   ├── NoteReaderModal.tsx # Fullscreen modal reader for notes
│   │   │   ├── QuizPage.tsx        # Dynamic academic quiz generator
│   │   │   ├── ReportsPage.tsx     # PYQ frequency and syllabus analytics
│   │   │   ├── StudentDashboard.tsx# User notebooks, study streak, and metrics
│   │   │   ├── ThemePickerModal.tsx# VS Code theme selector modal
│   │   │   └── UserProfileModal.tsx# Profile details, avatar upload, and college
│   │   ├── config/                 # Supabase client, API endpoints, themes, and features
│   │   ├── context/                # ThemeContext and state providers
│   │   ├── App.tsx                 # Core studio workspace and navigation
│   │   ├── main.tsx                # React DOM entrypoint
│   │   └── index.css               # Design system tokens and glassmorphism styles
│   ├── package.json                # Frontend dependencies
│   ├── vite.config.ts              # Vite configuration with React SWC/Babel
│   └── Dockerfile                  # Multi-stage production container
│
├── supabase/
│   └── migrations/                 # PostgreSQL schema migrations
│       └── 20261017000001_initial_studysnap_schema.sql # Tables, indexes & RLS policies
│
├── docs/                           # Technical documentation
│   ├── ARCHITECTURE.md             # System topology and service interactions
│   ├── DATABASE.md                 # Supabase schema, indexes, and RLS guide
│   ├── DEPLOYMENT.md               # Production deployment instructions
│   ├── PROJECT_AUDIT.md            # Comprehensive codebase audit and repair log
│   └── TROUBLESHOOTING.md          # Common errors and resolutions
│
├── docker-compose.yml              # Multi-container orchestration
├── run_pipeline.py                 # Pipeline runner for training custom models
└── README.md                       # Main documentation
```

---

## ⚡ Quick Start & Installation

### Prerequisites

- **Python:** 3.10, 3.11, or 3.12
- **Node.js:** v18.0.0 or higher (v20+ recommended) & npm
- **LM Studio:** [Download LM Studio](https://lmstudio.ai/)
- **GPU (Recommended):** NVIDIA GPU with 4GB+ VRAM (or CPU)
- **Supabase Account:** [Supabase Free Tier](https://supabase.com/) (or use the built-in offline fallback)

---

### Step 1: Clone Repository & Setup Environment

```bash
git clone https://github.com/AlphaBeta07/mega-project.git
cd mega_project
```

Create configuration files from examples:

```bash
# Frontend configuration
cp frontend/.env.example frontend/.env

# Backend configuration
cp backend/.env.example backend/.env
```

**`backend/.env` contents:**
```ini
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174

# LM Studio Configuration
LM_STUDIO_URL=http://localhost:1234/v1
MODEL_NAME=educationalassistant

# Supabase Credentials (optional for offline testing, required for cloud sync)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key
SUPABASE_SECRET_KEY=your-supabase-service-role-key
SUPABASE_JWT_SECRET=your-supabase-jwt-secret
SUPABASE_STORAGE_BUCKET=studysnap-assets
```

**`frontend/.env` contents:**
```ini
VITE_AI_BACKEND_URL=http://localhost:8000
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
VITE_SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key
```

---

### Step 2: Download Model & Start LM Studio

1. Download **`my_custom_model.Q4_K_M.gguf`** from [Google Drive](https://drive.google.com/drive/folders/1t8rqnwpjoVUJValpjHrmMkZs_c5lO6IT?usp=sharing).
2. Place the file in your LM Studio models directory (e.g., `C:\Users\<User>\.cache\lm-studio\models\my_custom_model_gguf\Anish\EducationalAssistant\my_custom_model.Q4_K_M.gguf`).
3. Open **LM Studio** → Select the **Local Server** tab (↔).
4. Load the model:
   - **File:** `my_custom_model.Q4_K_M.gguf`
   - **Quantization:** `Q4_K_M`
   - **Architecture:** `llama` (Size: 2.32 GB)
5. Set port to `1234` and click **Start Server**.

---

### Step 3: Run the FastAPI AI Backend

```powershell
# Navigate to backend directory
cd d:\mega_project\backend

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start FastAPI development server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*The backend starts at `http://localhost:8000` with interactive Swagger docs at `http://localhost:8000/docs`.*

---

### Step 4: Run the React 19 Frontend

Open a new terminal:

```powershell
# Navigate to frontend directory
cd d:\mega_project\frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
*Visit `http://localhost:5173` in your browser.*

---

### Alternative: Run with Docker Compose

To start the containerized backend, gateway, and frontend in one command:

```bash
docker compose up --build
```

---

## 🧪 Testing & Quality Assurance

### Backend Automated Test Suite (Pytest)

The backend includes a 31-test automated suite verifying all core functions, SSRF sanitization, ChromaDB vector indexing, notebook CRUD, and Supabase RLS security:

```powershell
cd d:\mega_project\backend
.\venv\Scripts\python -m pytest -v
```

**Test Execution Summary:**
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1
collected 31 items

tests/test_api.py::test_db_health PASSED                                 [  3%]
tests/test_api.py::test_system_status PASSED                             [  6%]
tests/test_api.py::test_ssrf_url_validation PASSED                       [  9%]
tests/test_api.py::test_sources_list PASSED                              [ 12%]
tests/test_api.py::test_text_chunking PASSED                             [ 16%]
tests/test_api.py::test_chromadb_persistence PASSED                      [ 19%]
tests/test_api.py::test_empty_upload_rejection PASSED                    [ 22%]
tests/test_api.py::test_chat_offline_resilience PASSED                   [ 25%]
tests/test_api.py::test_notes_read_resilience PASSED                     [ 29%]
tests/test_notebooks.py::test_notebook_crud_lifecycle PASSED             [ 32%]
tests/test_prompt_registry.py::test_detect_audio_to_notes_feature PASSED [ 35%]
tests/test_prompt_registry.py::test_sanitize_feature_title PASSED        [ 38%]
tests/test_prompt_registry.py::test_resolve_request_instruction_for_audionotes PASSED [ 41%]
tests/test_prompt_registry.py::test_resolve_request_instruction_explicit_feature PASSED [ 45%]
tests/test_prompt_registry.py::test_chat_endpoint_hides_system_prompt PASSED [ 48%]
tests/test_student_profile.py::test_get_student_profile PASSED           [ 51%]
tests/test_student_profile.py::test_update_student_profile PASSED        [ 54%]
tests/test_student_profile.py::test_update_theme_preference PASSED       [ 58%]
tests/test_student_profile.py::test_invalid_and_expired_token_rejection PASSED [ 61%]
tests/test_student_profile.py::test_student_profiles_isolation PASSED    [ 64%]
tests/test_student_profile.py::test_avatar_upload_valid_png PASSED       [ 67%]
tests/test_student_profile.py::test_avatar_upload_oversized_rejection PASSED [ 70%]
tests/test_student_profile.py::test_avatar_upload_unsupported_format_rejection PASSED [ 74%]
tests/test_student_profile.py::test_avatar_upload_empty_file_rejection PASSED [ 77%]
tests/test_student_profile.py::test_avatar_replacement_and_removal PASSED [ 80%]
tests/test_supabase_features.py::test_supabase_health_endpoint PASSED    [ 83%]
tests/test_supabase_features.py::test_expired_supabase_token_rejection PASSED [ 87%]
tests/test_supabase_features.py::test_malformed_token_rejection PASSED   [ 90%]
tests/test_supabase_features.py::test_supabase_notebook_rls_isolation PASSED [ 93%]
tests/test_supabase_features.py::test_supabase_notes_crud PASSED         [ 96%]
tests/test_supabase_features.py::test_supabase_chat_session_history PASSED [100%]

============================== 31 passed in 79.06s ==============================
```

### Frontend TypeScript Compilation & Build Validation

```powershell
cd d:\mega_project\frontend
npm run build
```
*Compiles cleanly with 0 TypeScript errors into `dist/`.*

---

## 📡 API Reference Overview

| HTTP Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/health/db` | Health check verifying Supabase PostgreSQL and fallback connectivity |
| `GET` | `/api/status` | Real-time connection status for LM Studio (`:1234`) and ComfyUI (`:8188`) |
| `GET` | `/api/sources` | Lists all ingested study sources in the active collection |
| `POST` | `/api/sources/upload` | Ingests PDF, DOCX, PPTX, TXT, or audio files with ChromaDB indexing |
| `POST` | `/api/sources/url` | Ingests web articles and YouTube video transcripts (SSRF protected) |
| `DELETE` | `/api/sources/{id}` | Deletes a source and removes associated vector embeddings |
| `POST` | `/api/chat` | Main conversational RAG endpoint with source citations and page numbers |
| `POST` | `/api/audio-overview` | Generates a 2-minute podcast dialogue and synthesizes Edge TTS audio |
| `POST` | `/api/mindmap` | Generates a hierarchical concept tree rendered via React Flow |
| `POST` | `/api/infographic` | Generates Bento Grid infographic data via ComfyUI Stable Diffusion |
| `POST` | `/api/quiz` | Auto-generates multiple choice questions with explanations |
| `POST` | `/api/flashcards` | Generates flashcard decks for spaced-repetition study |
| `GET` / `POST` | `/api/notebooks` | Fetches or creates user notebooks (enforcing RLS `owner_id`) |
| `PATCH` | `/api/notebooks/{id}` | Debounced autosave endpoint for student notes |
| `GET` / `PUT` | `/api/profile` | Reads or updates student profile metadata and theme preferences |
| `POST` | `/api/profile/avatar` | Uploads student avatar image to Supabase Storage (`studysnap-assets`) |

---

## 📖 Additional Documentation

- 🏛️ **[System Architecture Guide](docs/ARCHITECTURE.md)** — Architectural patterns, data flow, and service boundaries.
- 🗄️ **[Database & RLS Documentation](docs/DATABASE.md)** — Supabase PostgreSQL schema, indexes, and Row Level Security policies.
- 🚀 **[Production Deployment Guide](docs/DEPLOYMENT.md)** — Docker, reverse proxy setup, and production configurations.
- 🔍 **[Comprehensive System Audit](docs/PROJECT_AUDIT.md)** — Engineering audit, vulnerability mitigations, and feature roadmap.
- 🛠️ **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** — Solutions for common LM Studio, audio, or database issues.
- 🧠 **[7B Model Training & Architecture](main-readme.md)** — Hyperparameter details, LoRA configurations, and VRAM management.
- 🔬 **[NotebookCore-120M Pipeline](walkthrough.md)** — Step-by-step training walkthrough for the scratch-trained 120M model.

---

## 👥 Contributors & Acknowledgements

- **Anish** — Lead Full-Stack, AI Model Architecture & System Engineering
- **Team StudySnap AI** — Design, UI Components, and Evaluation

*Developed with ❤️ for students and educators worldwide.*
