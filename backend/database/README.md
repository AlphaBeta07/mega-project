# 🗄️ StudySnap AI - Complete MongoDB Database Module

This folder contains the complete, final **MongoDB database package** for **StudySnap AI - AI-Based Smart Lecture Notes Generator**, designed strictly according to the project's **Software Requirements Specification (SRS)** document.

---

## 📌 Module Architecture

```text
backend/database/
├── __init__.py          ← Package exports for 1-line imports
├── config.py            ← Environment settings & MongoDB configuration
├── connection.py        ← Motor async & PyMongo connection handlers
├── models.py            ← Pydantic & MongoDB models for all SRS entities
├── crud.py              ← Async CRUD query functions for all 8 collections
├── init_db.py           ← Setup CLI script to build indexes & test connectivity
├── seed_data.py         ← Demo dataset generator for team testing
└── README.md            ← Developer guide and usage documentation
```

---

## 📊 Database Collections & SRS Entity Mapping

| Collection Name | SRS Entity / Feature | Purpose |
|-----------------|----------------------|---------|
| `users` | **Users** | Stores student & lecturer accounts, bcrypt password hashes, and user roles. |
| `notes` | **Notes** | Stores generated lecture notes, chapter titles, Markdown content, summaries, and keywords. Includes text search index. |
| `audio_files` | **AudioFiles** | Metadata for uploaded/recorded lecture audio files, sizes, and file URLs. |
| `pyq_documents` | **PYQDocuments** | Stores Previous Year Question (PYQ) PDF metadata, extracted text, and topic frequency distributions. |
| `topic_mappings` | **TopicMapping** | Maps high-frequency PYQ topics to specific sections in generated Lecture Notes. |
| `chatbot_sessions` | **ChatbotSessions** | Stores doubt-solving chat sessions and complete message history. |
| `transcriptions` | **Transcription** | Stores raw Whisper ASR transcriptions, detected languages, and WER metadata. |
| `sources` | **RAG Sources** | Stores ingested study documents, text chunks count, and web links for RAG. |
| `ai_artifacts` | **AI Artifacts** | Stores generated Audio Podcasts, Infographics, and Mind Maps. |
| `user_settings` | **User Settings** | User preferences (response language, research mode). |

---

## 🚀 Quick Start Guide for Team Members

### 1. Install Dependencies
Make sure your virtual environment has the required packages installed:
```bash
pip install motor pymongo python-dotenv pydantic
```

### 2. Configure Environment Variables
Create or update your `.env` file in the `backend/` root directory:
```env
MONGODB_URL=mongodb://localhost:27017
DB_NAME=studysnap_db
```
*(For MongoDB Atlas Cloud DB, set `MONGODB_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net`)*

### 3. Initialize Database & Create Indexes
Run the initialization script from the `backend/` directory:
```bash
python -m database.init_db
```

### 4. Seed Sample Demo Data (Optional)
To insert sample users, notes, PYQ papers, and chat sessions for testing:
```bash
python -m database.seed_data
```

---

## 💻 Code Usage Examples for Backend Developers

### Import Database Helper Functions
```python
from database import get_async_db, UserCRUD, NoteCRUD, ChatbotCRUD, PYQCRUD, SourceCRUD
```

### 1. Create & Fetch Users
```python
db = get_async_db()

# Create a new user
user = await UserCRUD.create_user(db, {
    "name": "Aditi Pujari",
    "email": "aditi@dkte.ac.in",
    "password_hash": "$2b$12$hashed_password_here",
    "role": "student"
})

# Fetch user by email
user = await UserCRUD.get_user_by_email(db, "aditi@dkte.ac.in")
```

### 2. Create & Query Lecture Notes
```python
# Create a note
note = await NoteCRUD.create_note(db, {
    "user_id": user.user_id,
    "subject": "Operating Systems",
    "chapter": "Module 1: Process Control Block",
    "formatted_content": "# Process Control Block\nPCB contains PID, state, and registers.",
    "summary": "Introduction to Process Control Block.",
    "keywords": ["PCB", "Process", "PID"],
    "duration": 30.0
})

# Get all notes for a user
user_notes = await NoteCRUD.get_user_notes(db, user.user_id, subject="Operating Systems")
```

### 3. Save & Fetch Chatbot Sessions
```python
# Create a chat session
session = await ChatbotCRUD.create_session(db, {
    "user_id": user.user_id,
    "title": "OS Process Doubts"
})

# Add a user question and assistant answer
await ChatbotCRUD.add_message(db, session.session_id, {
    "role": "user",
    "content": "What is PCB?"
})
await ChatbotCRUD.add_message(db, session.session_id, {
    "role": "assistant",
    "content": "PCB stands for Process Control Block..."
})

# Retrieve session with message history
session_data = await ChatbotCRUD.get_session(db, session.session_id)
```

---

## 🛡️ Index & Search Strategy
- **Unique Constraints**: `users.email`, `users.user_id`, `notes.note_id`, `chatbot_sessions.session_id`, `sources.id`.
- **Text Search Index**: Full text search enabled on `notes` across `formatted_content`, `transcript_text`, `subject`, and `chapter`.
- **Foreign Key Indexing**: Indexed fields for fast querying: `notes.user_id`, `audio_files.user_id`, `pyq_documents.user_id`, `topic_mappings.note_id`.
