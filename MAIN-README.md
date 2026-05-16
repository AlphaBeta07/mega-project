# StudySnap AI 

StudySnap AI is a self-hosted, highly capable, local AI research assistant. It allows you to ingest a massive variety of data sources, store them efficiently in a local vector database, and chat with them using your own locally-hosted Large Language Model (LLM).

## 🚀 Key Features

*   **Pixel-Perfect UI**: Built with React, Tailwind CSS, and Framer Motion.
*   **Fully Local & Private**: Powered entirely by local models via LM Studio. Your data never leaves your machine.
*   **Massive File Support**: Natively ingest and chunk PDFs, DOCX, TXT, CSV, XLSX, PPTX, MD, HTML, JSON, and XML files.
*   **Web & YouTube Processing**: Paste any web URL to automatically scrape text, or paste a YouTube URL to automatically download and process video transcripts (with fallback support for auto-generated captions).
*   **Local Audio Transcription**: Upload MP3, WAV, or M4A audio files directly. Uses a local instance of OpenAI Whisper (`base` model) to accurately transcribe audio into text for RAG.
*   **Source Toggling**: Individually check/uncheck specific sources to control exactly what knowledge the AI uses to answer your current question.
*   **Markdown Rendering**: Rich chat UI that automatically parses and renders structured markdown, headings, bullet points, code blocks, and tables.

## 🛠️ Technology Stack

**Frontend (Client):**
*   React 18 (TypeScript) via Vite
*   Tailwind CSS (Styling & Dark Mode)
*   Lucide React (Icons)
*   Framer Motion (Animations)
*   React-Markdown & Remark-GFM (Message formatting)
*   React-Dropzone (Drag & drop file uploads)

**Backend (Server):**
*   FastAPI (Python Web Framework)
*   ChromaDB (Local Vector Database)
*   OpenAI Python SDK (for communicating with LM Studio's OpenAI-compatible API)
*   Whisper (Local audio transcription)
*   YouTube-Transcript-API (Video subtitles)
*   PyPDF, python-docx, pandas, openpyxl, python-pptx (Document parsing)
*   BeautifulSoup4 (Web scraping)

---

## ⚙️ Getting Started

### Prerequisites
1. **Node.js** (v18+)
2. **Python** (v3.10+)
3. **FFmpeg** (Required by Whisper for audio transcription; must be added to your system PATH)
4. **LM Studio** (With a model loaded and the local server running)

### 1. LM Studio Configuration
1. Open LM Studio and load your preferred model (e.g., `my_custom_7B_model_gguf/Anish/StudySnap AI`).
2. Start the Local Server.
3. Ensure it is running on `http://localhost:1234/v1`.

### 2. Backend Setup
Navigate to the backend directory, activate your virtual environment, and install the 14 core dependencies:
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
pip install -r requirements.txt
```

Run the FastAPI server:
```bash
uvicorn main:app --reload
```
The backend will run on `http://localhost:8000`.

### 3. Frontend Setup
Open a new terminal, navigate to the frontend directory, install dependencies, and start the Vite dev server:
```bash
cd frontend
npm install
npm run dev
```
The frontend will run on `http://localhost:5173` (or 5174).

---

## 📂 Project Structure

```text
mega_project-v4/
├── backend/
│   ├── main.py              # FastAPI endpoints (Upload, Chat, Delete)
│   ├── rag.py               # RAG pipeline, Chunking, Extractors, LM Studio Logic
│   ├── requirements.txt     # Python dependencies
│   ├── chroma_data/         # Persistent local vector database storage
│   └── uploads/             # Temporary storage for ingested files
└── frontend/
    ├── index.html
    ├── package.json
    └── src/
        ├── App.tsx          # Main layout, Chat UI, Source toggling logic
        ├── index.css        # Global styles, Tailwind, Markdown themes
        └── AddSourceModal.tsx # Drag & Drop file uploader and URL input
```

## 🧠 How the RAG Pipeline Works
1. **Ingestion**: When a user uploads a file or URL, the backend routes it to the correct parser in `rag.py`. 
2. **Chunking**: The extracted text is split into chunks of 1000 characters with 200 characters of overlap.
3. **Embedding**: Chunks are embedded using ChromaDB's default embedding function and stored locally in the `chroma_data` folder.
4. **Querying**: When a user chats, the backend filters ChromaDB for the user's `selected_source_ids`, retrieves the top 4 most relevant chunks, and sends them to LM Studio as context alongside the user's prompt.
