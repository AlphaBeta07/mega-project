# StudySnap AI — Comprehensive System Audit & Prioritized Repair Plan

**Document Version:** 1.0.0  
**Audit Date:** October 17, 2026  
**Auditor:** Senior Full-Stack, AI/ML, Database, DevOps & Application Security Engineer  
**Repository:** StudySnap AI (`d:\mega_project`)

---

## 1. Executive Summary

StudySnap AI is an advanced AI-powered academic study platform featuring:
- **Frontend:** React 19, TypeScript, Vite, Glassmorphic styling, Dark mode, and dynamic study modules (Audio Overview, Infographic Bento Grid, Mind Maps, Quizzes, Flashcards, Solved PYQs, Student Dashboard, Public Notes Catalog).
- **AI / RAG Engine (`backend/`):** FastAPI (Python 3.12), ChromaDB vector persistence, SentenceTransformers embeddings (`all-MiniLM-L6-v2`), LM Studio local inference (`llama-3.2-1b-instruct` / fine-tuned 7B/120M models), Edge TTS audio dialogue generation, ComfyUI image generator client, and multi-format document extractors (PDF, TXT, YouTube transcripts, web articles).
- **Application Gateway & Auth Backend (`main_backend/`):** Node.js, Express 4, JWT authentication, Bcrypt password hashing, Mongoose models (User, Note, PYQ, AudioFile, AIArtifact), aggregated Student Dashboard statistics, and an AI service proxy.

This audit evaluates the codebase across architecture, dependencies, security, data persistence, and API integration, establishing a concrete, prioritized repair roadmap.

---

## 2. Architecture & Service Topology

```
                          ┌─────────────────────────────┐
                          │   Frontend (React 19/Vite)  │
                          │   Port 5173 / 5174          │
                          └──────────────┬──────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
      ┌─────────────────────────┐                 ┌─────────────────────────┐
      │  Main Backend Gateway   │                 │   FastAPI AI Engine     │
      │  (Node.js / Express)    │                 │   (Python 3.12)         │
      │  Port 5000              │                 │   Port 8000             │
      └────────────┬────────────┘                 └────────────┬────────────┘
                   │                                           │
         ┌─────────┴─────────┐                       ┌─────────┴─────────┐
         ▼                   ▼                       ▼                   ▼
┌─────────────────┐ ┌─────────────────┐     ┌─────────────────┐ ┌─────────────────┐
│ MongoDB         │ │ Fallback Store  │     │ ChromaDB        │ │ Local AI        │
│ (Users, Notes,  │ │ (Standalone /   │     │ (Vector RAG     │ │ (LM Studio      │
│  PYQs, Auth)    │ │  Offline Mode)  │     │  Embeddings)    │ │  Port 1234)     │
└─────────────────┘ └─────────────────┘     └─────────────────┘ └─────────────────┘
```

---

## 3. Detailed Audit Findings by Component

### 3.1 Backend (`backend/` - FastAPI Port 8000)

| ID | Issue | Severity | Location | Impact | Status |
|---|---|---|---|---|---|
| **BE-01** | `audio_filename` undefined in `create_audio_overview` | **CRITICAL** | `backend/main.py:331` | Calling `/api/audio-overview` immediately crashes with `NameError: name 'audio_filename' is not defined`. The `generate_tts_audio` call was omitted during merge resolution. | **Identified for Immediate Fix** |
| **BE-02** | Unhandled MongoDB connection timeouts | **HIGH** | `backend/database/crud.py` | If MongoDB port 27017 is offline, calling CRUD methods causes long timeouts or unhandled exceptions. | **Identified for Immediate Fix** |
| **BE-03** | Insecure file deletion & directory traversal risk | **MEDIUM** | `backend/main.py:205` | `remove_source` checks `filename.startswith(file_id)` without validating against traversal paths. | **Identified for Immediate Fix** |
| **BE-04** | Hardcoded `http://localhost:8000` URLs in responses | **MEDIUM** | `backend/main.py:331` | Audio URL returned as hardcoded `localhost:8000`, breaking containerized/remote deployments. | **Identified for Immediate Fix** |
| **BE-05** | SSRF risk in URL source ingestion | **HIGH** | `backend/rag.py` | `ingest_document` fetches arbitrary URLs using `requests.get` without validating against private IP ranges (`127.0.0.1`, `169.254.169.254`, `10.x.x.x`). | **Identified for Immediate Fix** |

### 3.2 Main Backend (`main_backend/` - Express Port 5000)

| ID | Issue | Severity | Location | Impact | Status |
|---|---|---|---|---|---|
| **MB-01** | Missing `node_modules` | **CRITICAL** | `main_backend/` | Service cannot run until `npm install` is executed. | **RESOLVED** (`npm install` completed) |
| **MB-02** | Mongoose operation buffering timeout (10s freeze) | **HIGH** | `main_backend/config/db.js` | When MongoDB is unreachable, queries buffer for 10,000ms before returning 500 errors. | **Identified for Immediate Fix** |
| **MB-03** | Lack of offline/standalone database fallback | **MEDIUM** | `main_backend/controllers/` | When developing without a live MongoDB daemon, user registration, login, and dashboard endpoints fail. | **Identified for Immediate Fix** |
| **MB-04** | Missing rate limiting on auth endpoints | **MEDIUM** | `main_backend/routes/authRoutes.js` | No brute-force protection on `POST /api/auth/login` and `/register`. | **Identified for Immediate Fix** |

### 3.3 Frontend (`frontend/` - React 19/Vite)

| ID | Issue | Severity | Location | Impact | Status |
|---|---|---|---|---|---|
| **FE-01** | Hardcoded API backend URL | **HIGH** | `frontend/src/App.tsx:55` | `const backendUrl = "http://localhost:8000"` hardcoded; no `.env` or `.env.example` provided. | **Identified for Immediate Fix** |
| **FE-02** | `LandingPage.tsx` uses mock data directly | **HIGH** | `frontend/src/components/LandingPage.tsx` | Directly imports `MOCK_PUBLIC_NOTES` from `mockNotes.ts` rather than fetching `GET /api/notes/public`. | **Identified for Immediate Fix** |
| **FE-03** | `StudentDashboard.tsx` uses hardcoded mock state | **HIGH** | `frontend/src/components/StudentDashboard.tsx` | Notebooks and stats are hardcoded constants instead of consuming `/api/dashboard/stats` and `/api/dashboard/notebooks`. | **Identified for Immediate Fix** |
| **FE-04** | `AuthModal.tsx` has fake authentication | **HIGH** | `frontend/src/components/AuthModal.tsx` | Submitting the form or clicking "Demo Login" does not call `POST /api/auth/login`, `/register`, or `/demo-login`. | **Identified for Immediate Fix** |
| **FE-05** | Broken audio overview modal player reference | **MEDIUM** | `frontend/src/components/AudioOverviewPage.tsx` | Audio element lacks error handling when generation produces an unreachable URL. | **Identified for Immediate Fix** |

### 3.4 Security & Secrets

| ID | Issue | Severity | Location | Recommendation |
|---|---|---|---|---|
| **SEC-01** | Committed Gemini API Key in `.env` | **CRITICAL** | `backend/.env:4` | The active `GEMINI_API_KEY` was committed. **Recommended action:** Revoke and rotate this key immediately; keep `.env` in `.gitignore`. |
| **SEC-02** | Hardcoded JWT Secrets in `.env` | **MEDIUM** | `backend/.env`, `main_backend/.env` | Default secrets should be flagged with runtime warnings in production environments. |
| **SEC-03** | Untrusted document content prompt injection | **MEDIUM** | `backend/rag.py` | RAG retrieval inserts context directly into the prompt without delimiting source chunks. |

---

## 4. Frontend-to-Backend API Mapping Matrix

| Frontend Component | UI Action / Trigger | Target Endpoint | Backend Service | Status / Repair Needed |
|---|---|---|---|---|
| `AuthModal.tsx` | Log in | `POST /api/auth/login` | `main_backend:5000` | Currently simulated in memory; wire to real endpoint |
| `AuthModal.tsx` | Demo Login | `POST /api/auth/demo-login` | `main_backend:5000` | Currently simulated in memory; wire to real endpoint |
| `LandingPage.tsx` | Load public notes & filter | `GET /api/notes/public` | `main_backend:5000` | Currently imports `mockNotes.ts`; wire to real endpoint |
| `LandingPage.tsx` | Like note | `POST /api/notes/:id/like` | `main_backend:5000` | Wire up real counter update |
| `StudentDashboard.tsx` | Load stats | `GET /api/dashboard/stats` | `main_backend:5000` | Currently static constants; wire to real endpoint |
| `StudentDashboard.tsx` | Load notebooks | `GET /api/dashboard/notebooks` | `main_backend:5000` | Currently static constants; wire to real endpoint |
| `App.tsx` | List sources | `GET /api/sources` | `backend:8000` | Working |
| `AddSourceModal.tsx` | Upload document | `POST /api/sources/upload` | `backend:8000` | Working |
| `AddSourceModal.tsx` | Add URL/YouTube | `POST /api/sources/url` | `backend:8000` | Working (Streaming NDJSON) |
| `App.tsx` | Delete source | `DELETE /api/sources/:id` | `backend:8000` | Working |
| `App.tsx` | Chat with RAG | `POST /api/chat` | `backend:8000` | Working (with fallback) |
| `AudioOverviewPage.tsx` | Generate podcast | `POST /api/audio-overview` | `backend:8000` | Broken by BE-01; needs repair |
| `InfographicEditor.tsx` | Generate infographic | `POST /api/infographic` | `backend:8000` | Working |
| `MindMapPage.tsx` | Generate mindmap | `POST /api/mindmap` | `backend:8000` | Working |
| `QuizPage.tsx` | Generate quiz | `POST /api/quiz` | `backend:8000` | Working |
| `FlashcardsPage.tsx` | Generate flashcards | `POST /api/flashcards` | `backend:8000` | Working |

---

## 5. Prioritized Implementation Roadmap

### Phase A: Core Backend Stability (Immediate)
1. Fix `audio_filename = await generate_tts_audio(script_json)` in `backend/main.py`.
2. Sanitize file uploads and deletions in `backend/main.py`.
3. Harden `backend/rag.py` against SSRF when fetching user-supplied URLs.
4. Prevent unhandled MongoDB exceptions in `backend/database/crud.py` by adding graceful offline detection and fallback.

### Phase B: Gateway & Persistence Resilience
1. In `main_backend/config/db.js`, disable `bufferCommands` and add immediate offline detection.
2. Implement robust JSON/SQLite persistence fallback in `main_backend` so the system operates reliably both standalone and with MongoDB.
3. Wire and verify `test_backend.js` test suite.

### Phase C: Frontend API Wiring & Config Clean-up
1. Create `frontend/.env.example` and `frontend/.env` supporting `VITE_API_GATEWAY_URL` (Port 5000) and `VITE_AI_BACKEND_URL` (Port 8000).
2. Wire `AuthModal.tsx` to `POST /api/auth/login`, `register`, and `demo-login`, storing the JWT token in `localStorage`.
3. Wire `LandingPage.tsx` to fetch from `GET /api/notes/public` with graceful fallback to seeded notes if backend is unreachable.
4. Wire `StudentDashboard.tsx` to fetch real stats and notebooks from `/api/dashboard/stats` and `/api/dashboard/notebooks`.

### Phase D: Automated Testing & Verification
1. Add backend test suite (`pytest`) in `backend/tests/` covering endpoints, ChromaDB persistence, RAG query formatting, and error handling.
2. Execute and verify `main_backend` automated tests (`npm test`).
3. Execute and verify frontend build (`npm run build`).

### Phase E: Docker Deployment & Documentation
1. Create `docker-compose.yml` defining `mongodb`, `fastapi-backend`, `express-gateway`, and `frontend`.
2. Create production Dockerfiles for each tier.
3. Create comprehensive documentation (`docs/ARCHITECTURE.md`, `docs/DATABASE.md`, `docs/DEPLOYMENT.md`, `docs/TROUBLESHOOTING.md`).
4. Update `README.md` with complete installation and execution instructions.

---

## 6. Implementation Summary: Supabase Auth & MongoDB Primary Database

| Feature | Status | Verification | Notes |
|---|---|---|---|
| **Supabase Client in Frontend** | Verified | `frontend/src/config/supabase.ts` | Configured with `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`; auto session refresh |
| **Supabase Auth in AuthModal** | Verified | `frontend/src/components/AuthModal.tsx` | `signUpWithEmail`, `signInWithEmail`, error validation, friendly alerts |
| **Session Persistence across Reloads** | Verified | `frontend/src/App.tsx` | Supabase `onAuthStateChange` listener restored session on page reload |
| **MongoDB Notebooks Collection** | Verified | `backend/database/models.py` | Full schema with UUID, `user_id`, `title`, `content`, `sources`, `chat_history`, timestamps |
| **MongoDB Notebook CRUD** | Verified | `backend/database/crud.py` | Create, List, Get by ID, Update (Autosave), Delete with user ownership isolation |
| **Supabase JWT Verification** | Verified | `backend/auth.py` | Decodes Bearer JWT, validates expiration, extracts `sub` claim for MongoDB queries |
| **FastAPI Notebook Endpoints** | Verified | `backend/main.py` | `POST /api/notebooks`, `GET /api/notebooks`, `GET /api/notebooks/:id`, `PUT`, `DELETE` |
| **Dashboard New Notebook & List** | Verified | `frontend/src/components/StudentDashboard.tsx` | Real MongoDB creation, rename, delete, empty state, and live metrics |
| **Debounced Autosave in Studio** | Verified | `frontend/src/App.tsx` | 1200ms debounce, status indicators: `Saving...`, `Saved to MongoDB`, `Save pending` |
| **Backend Test Suite (Pytest)** | 100% Passed | `pytest tests/` (10 passed) | Tests Notebook CRUD, cross-user isolation, RAG, ChromaDB persistence |
| **Gateway Test Suite (Node.js)** | 100% Passed | `node test_backend.js` (17 passed) | Validates auth, notes CRUD, PYQ, dashboard |
| **Frontend Production Build** | 100% Passed | `npm run build` | Zero TypeScript errors, built in 11.73s |

