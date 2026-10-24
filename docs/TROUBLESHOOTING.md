# StudySnap AI — Troubleshooting & Diagnostics Guide

This document lists common issues, root causes, diagnostic checks, and verified solutions across all components of StudySnap AI.

---

## 1. LM Studio & AI Inference Issues

### Symptom: `Failed to connect to LM Studio at http://localhost:1234/v1`
- **Cause:** LM Studio is either not running, the local server has not been started, or the port differs from `1234`.
- **Diagnosis:**
  ```powershell
  # Test if port 1234 is listening
  Test-NetConnection -ComputerName localhost -Port 1234
  ```
- **Solution:**
  1. Open LM Studio on your computer.
  2. Navigate to the **Local Server** (double-arrow icon) tab on the left sidebar.
  3. Select your model (e.g., `llama-3.2-1b-instruct` or the custom fine-tuned 7B model downloaded from [Google Drive](https://drive.google.com/drive/folders/1t8rqnwpjoVUJValpjHrmMkZs_c5lO6IT?usp=sharing)).
  4. Set port to `1234` and click **Start Server**.
  5. *Note:* Even if LM Studio is offline, StudySnap AI will still retrieve, display, and cite the relevant document chunks without crashing.

---

## 2. MongoDB & Database Failover

### Symptom: Application takes 10+ seconds to respond on initial request
- **Cause:** Previous versions of Mongoose had `bufferCommands: true`, which caused all requests to hang waiting for MongoDB to connect.
- **Resolution:**
  - This has been fixed in `main_backend/config/db.js` with `bufferCommands: false` and `serverSelectionTimeoutMS: 2000`.
  - When MongoDB is not running, the application immediately switches to `main_backend/data/store.json` with zero request delay.
- **To Start MongoDB:**
  ```powershell
  # If MongoDB is installed as a Windows service:
  net start MongoDB

  # If running via Docker:
  docker run -d --name studysnap-mongo -p 27017:27017 mongo:7.0
  ```

---

## 3. Audio Overview & Edge TTS / Whisper

### Symptom: `FileNotFoundError: ffmpeg not found` or audio generation hangs
- **Cause:** Whisper and `pydub` require `ffmpeg` to decode and convert audio streams.
- **Diagnosis:**
  ```powershell
  ffmpeg -version
  ```
- **Solution:**
  1. Install `ffmpeg` using `winget`:
     ```powershell
     winget install "Gyan.FFmpeg"
     ```
  2. Restart your terminal to refresh the system PATH environment variable.
  3. Verify with `ffmpeg -version`.

---

## 4. ChromaDB Vector Database

### Symptom: `chromadb.errors.InvalidCollectionException` or Vector lock error
- **Cause:** Multiple Python processes attempting to write simultaneously to `backend/chroma_data/` without shared access, or a corrupted SQLite file from an abrupt shutdown.
- **Solution:**
  1. Stop any running FastAPI processes.
  2. Check if any stray Python processes are locking the directory:
     ```powershell
     Get-Process python | Select-Object Id, ProcessName
     ```
  3. If indices are corrupted, ChromaDB will automatically recreate them upon re-uploading documents.
  4. Ensure `backend/chroma_data/` has read/write permissions for the current user.

---

## 5. Infographics & ComfyUI

### Symptom: `Failed to connect to ComfyUI at http://localhost:8188`
- **Cause:** ComfyUI is not launched on port 8188.
- **Solution:**
  1. Start ComfyUI via `run_nvidia_gpu.bat` (or your local ComfyUI launcher).
  2. Verify that `http://localhost:8188` loads in your web browser.
  3. If you do not have ComfyUI running, all other features (Chat, RAG, Audio Overviews, Mind Maps, and Notes) continue to operate normally.

---

## 6. Frontend Build & TypeScript Issues

### Symptom: `error TS6133: 'xyz' is declared but its value is never read`
- **Cause:** TypeScript `tsconfig.json` has `noUnusedLocals: true`.
- **Solution:**
  - Run `npm run build` in `frontend/` to view exact file and line numbers.
  - All unused variables have been cleaned up and tested. Ensure future modifications consume declared variables or prefix them with an underscore.

---

## 7. Diagnostic Quick Check Commands

Run this sequence in PowerShell to verify the health of all subsystems:

```powershell
# 1. Test Gateway Health
Invoke-RestMethod -Uri "http://localhost:5000/api/health"

# 2. Test AI Engine Health
Invoke-RestMethod -Uri "http://localhost:8000/api/health"

# 3. Test Gateway Backend Test Suite
cd d:\mega_project\main_backend
node test_backend.js

# 4. Test AI Engine Pytest Suite
cd d:\mega_project\backend
.\venv\Scripts\pytest -v tests/test_api.py
```
