<p align="center">
  <img src="https://img.shields.io/badge/StudySnap-AI-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Frontend-React%2019%20%7C%20TypeScript%20%7C%20Vite-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Database-Supabase%20PostgreSQL%20%2B%20RLS-3ECF8E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/AI%20Engine-FastAPI%20%7C%20Python%203.12-teal?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-Mistral%207B%20%7C%20120M%20Scratch-red?style=for-the-badge" />
</p>

# StudySnap AI — Complete Production-Ready AI Study Assistant (Supabase Edition)

**StudySnap AI** is a production-ready, full-stack academic study assistant engineered for students and educators. It uses **Supabase** for user authentication, PostgreSQL relational data persistence, and Row Level Security (RLS), combined with **ChromaDB** for high-precision Retrieval-Augmented Generation (RAG).

The system ingests lecture recordings, PDFs, slides, syllabus documents, and YouTube links, synthesizes 2-minute podcast-style Audio Overviews (via Edge TTS & Whisper), generates concept Mind Maps (React Flow), produces AI-rendered Infographics (ComfyUI), and persists student notebooks with debounced autosave.

> [!NOTE]
> **Supabase Only:** StudySnap AI uses Supabase as its primary database and auth provider. MongoDB is completely removed and no longer required.

---

## 🚀 Key Features

- **Supabase Authentication & RLS:** Secure email/password registration, login, logout, password recovery, session restoration across refreshes, and PostgreSQL Row Level Security (`auth.uid() = owner_id`).
- **Student Dashboard & Notebook Workspace:** Real-time notebook history, creation with UUIDs, debounced autosave (1200ms) with visible saving/saved/error state indicators, renaming, and deletion.
- **Document & Multimodal Ingestion:** Upload PDF, DOCX, PPTX, CSV, XLSX, Web URLs, and YouTube transcripts. Chunked and indexed into persistent ChromaDB.
- **Context-Grounded RAG Chat:** Ask questions and receive verified answers with source citations, page numbers, and exact excerpts.
- **2-Minute Audio Overviews:** Convert documents into natural multi-speaker podcast audio overviews synthesized with Edge TTS.
- **Interactive Mind Maps:** Concept hierarchies converted into JSON trees and rendered interactively with React Flow.
- **Visual Infographics:** Transform study notes into Bento Grid infographics using local ComfyUI Stable Diffusion workflows.
- **Zero-Downtime Resilience:** Resilient thread-safe local fallback store ensures all features remain operational during offline development and testing.

---

## 📥 AI Models & Weights Download

StudySnap AI supports local offline inference through LM Studio using our custom fine-tuned models:

> [!IMPORTANT]
> **Model Weights Download (Google Drive):**
> 📥 **[Download Model Weights & GGUF Files](https://drive.google.com/drive/folders/1t8rqnwpjoVUJValpjHrmMkZs_c5lO6IT?usp=sharing)**
>
> Download the `.gguf` file and load it into LM Studio on port `1234` for local GPU/CPU inference without external API costs or internet dependency.

- **Production LLM:** Custom fine-tuned Mistral 7B Instruct LoRA (`studysanpai` / Q4_K_M GGUF).
- **Architecture Research Engine:** `NotebookCore-120M`, a decoder-only transformer trained from scratch (RoPE, RMSNorm, SwiGLU, custom BPE tokenizer).

---

## 🏗️ Architecture Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND LAYER                                  │
│   React 19 • TypeScript • Vite • Lucide • React Flow                         │
│   - Supabase Auth Integration (Login, Register, Password Recovery)           │
│   - Student Dashboard (Real Notebook History, Creation, Rename, Delete)      │
│   - Studio Workspace (Debounced Autosave, RAG, Audio, MindMap, Infographics) │
└───────────────────────┬───────────────────────────────┬──────────────────────┘
                        │                               │
                        ▼ (Supabase JS Client + RLS)    ▼ (Bearer <Supabase JWT>)
┌───────────────────────────────────────┐ ┌────────────────────────────────────┐
│         SUPABASE CLOUD / LOCAL        │ │     AI INFERENCE ENGINE (FastAPI)  │
│  - Supabase Auth (JWT Provider)       │ │     Port: 8000                     │
│  - Supabase PostgreSQL                │ │  - Supabase JWT Token Auth         │
│    (profiles, notebooks, documents,   │ │  - Document Parsing & Chunking     │
│     notes, chat_sessions, messages)   │ │  - ChromaDB Vector Store & Query   │
│  - Row Level Security (RLS)           │ │  - LM Studio Client & RAG Context  │
│  - Storage Bucket: studysnap-assets   │ │  - Audio Synthesis (Edge TTS)      │
└───────────────────────────────────────┘ └─────────────────┬──────────────────┘
                                                            │
                                                            ▼
                                          ┌────────────────────────────────────┐
                                          │         LOCAL AI SERVERS           │
                                          │  - LM Studio (Port 1234)          │
                                          │  - ComfyUI (Port 8188)             │
                                          │  - ChromaDB (backend/chroma_data/) │
                                          └────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Supabase Database Setup
1. Open your [Supabase Dashboard](https://supabase.com/dashboard) and go to the **SQL Editor**.
2. Run the migration file:
   [`supabase/migrations/20261017000001_initial_studysnap_schema.sql`](supabase/migrations/20261017000001_initial_studysnap_schema.sql)
3. Copy your **Supabase URL**, **Anon/Publishable Key**, and **JWT Secret** into your `.env` files.

### 2. Configure Environment Files
Copy `.env.example` to `frontend/.env` and `backend/.env`:
```bash
# Frontend (frontend/.env)
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
VITE_SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key
VITE_AI_BACKEND_URL=http://localhost:8000

# Backend (backend/.env)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key
SUPABASE_SECRET_KEY=your-supabase-service-role-key
SUPABASE_JWT_SECRET=your-supabase-jwt-secret
LM_STUDIO_URL=http://localhost:1234/v1
```

### 3. Start Backend & Frontend

**Terminal 1 — FastAPI AI Backend:**
```powershell
cd d:\mega_project\backend
.\venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — React 19 Frontend:**
```powershell
cd d:\mega_project\frontend
npm run dev
```

Visit `http://localhost:5173` to access StudySnap AI!

---

## 🧪 Testing

Run backend pytest tests (including Supabase JWT token verification, cross-user RLS isolation, and CRUD operations):
```powershell
cd d:\mega_project\backend
.\venv\Scripts\python -m pytest -v
```
*All 16 tests passing 100%.*

Validate frontend TypeScript and bundle build:
```powershell
cd d:\mega_project\frontend
npm run build
```
*0 errors, production bundle compiled cleanly in `dist/`.*

---

## 📚 Documentation
- [Supabase Database & RLS Guide](docs/DATABASE.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Project Audit](docs/PROJECT_AUDIT.md)
