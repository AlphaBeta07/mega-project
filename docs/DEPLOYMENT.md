# StudySnap AI — Production Deployment Guide

## 1. Overview

StudySnap AI supports two deployment modalities:
1. **Containerized Production Deployment:** Orchestrated using Docker and Docker Compose (recommended for cloud and enterprise setups).
2. **Native Local Development / Multi-Process Deployment:** Run natively on Windows or Linux using Node.js, Python virtual environment, and local AI servers (LM Studio, ComfyUI).

---

## 2. Option A: Containerized Production Deployment (Docker Compose)

### Prerequisites
- Docker Engine 24.0+
- Docker Compose v2.20+
- At least 8 GB RAM and 15 GB available disk space.

### Step 1: Clone & Configure Environment
Clone the repository and inspect the environment template:
```bash
cp .env.example .env
```

### Step 2: Build & Launch Containers
Run the unified Docker Compose stack:
```bash
docker compose up -d --build
```

This launches four interconnected services:
- `studysnap-mongodb`: MongoDB 7.0 database (Port 27017)
- `studysnap-ai-engine`: FastAPI Python 3.12 AI Engine (Port 8000)
- `studysnap-gateway`: Express Node.js API Gateway (Port 5000)
- `studysnap-web`: Nginx-served React 19 Frontend (Port 3000)

### Step 3: Verify Container Health
Check the status of all services:
```bash
docker compose ps
```

Verify service endpoints:
- **Frontend Web App:** `http://localhost:3000`
- **Main API Gateway:** `http://localhost:5000/api/health`
- **AI Engine Health:** `http://localhost:8000/api/health`
- **FastAPI Interactive Docs:** `http://localhost:8000/docs`

### Step 4: Stopping & Graceful Teardown
```bash
docker compose down
```
To also purge database volumes:
```bash
docker compose down -v
```

---

## 3. Option B: Native Host Deployment (Windows PowerShell)

For local development and testing with full GPU acceleration for LM Studio and ComfyUI:

### Prerequisites
- Windows 10/11 64-bit
- Node.js 18+ (tested on Node v20/v22)
- Python 3.10 – 3.12 (with `ffmpeg` on PATH)
- NVIDIA GPU with CUDA 12.1+ (optional, for local LLM inference)

### Step 1: Prepare Python AI Engine (`backend/`)
Open **Terminal 1**:
```powershell
cd d:\mega_project\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Prepare Node.js Gateway (`main_backend/`)
Open **Terminal 2**:
```powershell
cd d:\mega_project\main_backend
npm install
npm test
npm start
```

### Step 3: Prepare Frontend (`frontend/`)
Open **Terminal 3**:
```powershell
cd d:\mega_project\frontend
npm install
npm run build
npm run dev
```

### Step 4: Local AI Models (LM Studio & ComfyUI)
1. **LM Studio:**
   - Download the model weights: [Google Drive Model Weights](https://drive.google.com/drive/folders/1t8rqnwpjoVUJValpjHrmMkZs_c5lO6IT?usp=sharing)
   - Open LM Studio -> Local Server tab -> Load model -> Click **Start Server** on port `1234`.
2. **ComfyUI (Optional for infographics):**
   - Start ComfyUI on default port `8188`.

---

## 4. Production Checklist & Security Hardening

- [x] **No Hardcoded Secrets:** JWT secrets, database connection strings, and API keys are strictly configured via `.env`.
- [x] **SSRF Protection:** External URL ingestion resolves DNS and blocks internal/private IP blocks.
- [x] **Directory Traversal Protection:** All file deletion and upload paths are validated with `os.path.commonpath`.
- [x] **Database Failover:** If MongoDB goes down, `storageService.js` prevents server crashes by failing over to the persistent ACID JSON store.
- [x] **Zero TypeScript Errors:** Frontend production bundle compiled and verified using `tsc -b && vite build`.
- [x] **Automated Test Coverage:** Automated test suites verify API endpoints, ChromaDB persistence, authentication, and SSRF defenses.
