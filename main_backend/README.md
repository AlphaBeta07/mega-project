# 🎓 StudySnap AI — Main Backend (`main_backend`)

The **Main Application Backend & API Gateway** for **StudySnap AI — AI-Based Smart Lecture Notes Generator**.

Built strictly in **Node.js, Express.js, MongoDB (Mongoose), JWT, and bcryptjs**.

---

## 📌 1. Backend Purpose

`main_backend` is the central orchestration hub and primary backend server for StudySnap AI.

- **Main Backend ≠ Chatbot Backend**: The AI RAG chatbot engine is built in Python (FastAPI on Port 8000) and remains 100% untouched.
- **Role of `main_backend`**: Handles student & lecturer authentication, user academic profiles, lecture notes CRUD, community public discovery & filtering, previous year questions (PYQs) analysis & exam predictions, audio recordings & transcriptions metadata, student dashboard metrics, and proxies AI requests directly to the existing Python chatbot engine.

---

## 📁 2. Folder Structure

```text
main_backend/
│
├── config/
│   └── db.js                         # Mongoose connection to MongoDB (studysnap_db)
│
├── controllers/
│   ├── authController.js             # Register, Login, Me, Demo Login
│   ├── userController.js             # Profile, Branch, Semester, Settings
│   ├── notesController.js            # Notes CRUD, Public Discovery, Search, Likes
│   ├── pyqController.js              # PYQ Papers, Trend Analysis, Topic Predictions
│   ├── audioController.js            # Lecture Audio Files & Transcriptions
│   ├── dashboardController.js        # Aggregated Metrics & Active Notebooks
│   └── chatbotController.js          # Chatbot & RAG Proxy Controller
│
├── models/
│   ├── User.js                       # Matching existing 'users' collection
│   ├── Note.js                       # Matching existing 'notes' collection
│   ├── AudioFile.js                  # Matching existing 'audio_files' collection
│   ├── PYQDocument.js                # Matching existing 'pyq_documents' collection
│   ├── TopicMapping.js               # Matching existing 'topic_mappings' collection
│   ├── ChatbotSession.js             # Matching existing 'chatbot_sessions' collection
│   ├── Transcription.js              # Matching existing 'transcriptions' collection
│   ├── SourceDocument.js             # Matching existing 'sources' collection
│   ├── AIArtifact.js                 # Matching existing 'ai_artifacts' collection
│   └── UserSettings.js               # Matching existing 'user_settings' collection
│
├── routes/
│   ├── authRoutes.js                 # /api/auth
│   ├── userRoutes.js                 # /api/users
│   ├── notesRoutes.js                # /api/notes
│   ├── pyqRoutes.js                  # /api/pyqs
│   ├── audioRoutes.js                # /api/audio
│   ├── dashboardRoutes.js            # /api/dashboard
│   └── chatbotRoutes.js              # /api/chat, /api/sources, /api/artifacts
│
├── middleware/
│   ├── authMiddleware.js             # JWT verification & req.user injection
│   └── errorMiddleware.js            # Centralized JSON error handling
│
├── services/
│   └── chatbotService.js             # HTTP proxy to existing Python FastAPI backend
│
├── utils/
│   ├── apiResponse.js                # Consistent JSON response helper
│   └── generateToken.js              # JWT generation helper
│
├── .env.example                      # Environment variables template
├── .gitignore
├── package.json
├── server.js                         # Main Express application entry point
├── test_backend.js                   # Automated test suite
└── README.md                         # Documentation
```

---

## ⚙️ 3. Installation & Setup

### Prerequisites
- **Node.js**: v18.0.0+ (Tested on v24.x)
- **MongoDB**: Local instance running on port `27017` or MongoDB Atlas URI

### Steps
1. Navigate to the `main_backend` directory:
   ```bash
   cd main_backend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Create your `.env` configuration:
   ```bash
   cp .env.example .env
   ```

---

## 🔑 4. Environment Variables (`.env`)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `5000` | Port for the Main Backend Express server |
| `NODE_ENV` | `development` | Environment mode (`development` or `production`) |
| `MONGO_URI` | `mongodb://localhost:27017/studysnap_db` | MongoDB connection string |
| `JWT_SECRET` | `studysnap_jwt_super_secret_key_2026_change_in_production` | Secret key for signing JWT tokens |
| `JWT_EXPIRES_IN` | `7d` | Token validity duration |
| `CHATBOT_BACKEND_URL` | `http://localhost:8000` | URL of the existing Python FastAPI chatbot backend |
| `FRONTEND_URL` | `http://localhost:5173` | Allowed frontend origins for CORS |

---

## 🚀 5. How to Start the Server

```bash
# Start in production mode
npm start

# Start in development mode with nodemon auto-reload
npm run dev

# Run automated end-to-end API test suite
npm test
```

---

## 🌐 6. API Endpoints Reference

### 🔐 Authentication (`/api/auth`)
- `POST /api/auth/register` — Register a student/lecturer account with bcrypt password hashing.
- `POST /api/auth/login` — Authenticate credentials and receive signed JWT.
- `POST /api/auth/demo-login` — Quick one-click demo login for instant access.
- `GET /api/auth/me` — **(Protected)** Get logged-in user profile from token.

### 👤 User Management (`/api/users`)
- `GET /api/users/profile` — **(Protected)** Retrieve academic profile details.
- `PUT /api/users/profile` — **(Protected)** Update college, branch, semester.
- `GET /api/users/settings` — **(Protected)** Retrieve preferences (response language, research mode, theme).
- `PUT /api/users/settings` — **(Protected)** Update preferences.

### 📚 Lecture Notes (`/api/notes`)
- `GET /api/notes/public` — Public study notes with filters (`college`, `branch`, `semester`, `subject`, `note_type`, `search`, `sort_by`).
- `GET /api/notes` — **(Protected)** Get notes created by the authenticated user.
- `GET /api/notes/:id` — Get note detail with breakdown, formulas, and worked PYQs.
- `POST /api/notes` — **(Protected)** Create a new lecture note.
- `PUT /api/notes/:id` — **(Protected)** Update an existing note.
- `DELETE /api/notes/:id` — **(Protected)** Delete a note.
- `POST /api/notes/:id/like` — Toggle like on a note.

### 📝 PYQ Documents & Trend Analysis (`/api/pyqs`)
- `POST /api/pyqs` — Upload PYQ question paper metadata & extracted text.
- `GET /api/pyqs` — List PYQs by subject and year.
- `GET /api/pyqs/:id` — Get single PYQ document.
- `GET /api/pyqs/analysis/:subject` — Trend analysis, topic frequencies, and high-yield predictions.
- `GET /api/pyqs/mappings/:noteId` — Get topic mappings linked to a lecture note.

### 🎙️ Audio & Transcriptions (`/api/audio`)
- `POST /api/audio` — **(Protected)** Save uploaded lecture audio file metadata.
- `GET /api/audio` — **(Protected)** Get user's recorded audio files.
- `DELETE /api/audio/:id` — **(Protected)** Delete audio file record.
- `GET /api/audio/:id/transcription` — Get transcription for audio file.

### 📊 Dashboard & Metrics (`/api/dashboard`)
- `GET /api/dashboard/stats` — Real-time metrics: Active Notebooks, Uploaded Sources, Audio Synthesis, Solved PYQs.
- `GET /api/dashboard/notebooks` — Active study notebooks with source and note counts.

### 🤖 Chatbot & AI RAG Integration (`/api`)
*(Delegated to existing Python FastAPI backend via `chatbotService`)*
- `POST /api/chat` — Context-aware RAG query.
- `GET /api/chat/sessions` — Fetch chat history.
- `GET /api/chat/sessions/:id` — Get session messages.
- `DELETE /api/chat/sessions/:id` — Delete session.
- `POST /api/audio-overview` — Generate 2-min podcast audio overview.
- `POST /api/infographic` — Generate Markdown / Bento Grid infographic.
- `POST /api/mindmap` — Generate hierarchical Mind Map JSON.
- `POST /api/video-overview` — Generate video slides overview.
- `GET /api/sources` — List ingested RAG documents.
- `POST /api/sources/url` — Ingest web / YouTube URL.
- `DELETE /api/sources/:id` — Delete source.

---

## 📡 7. Request & Response Examples

### Register User
**Request:** `POST /api/auth/register`
```json
{
  "name": "Rohan Sharma",
  "email": "rohan.sharma@iitb.ac.in",
  "password": "password123",
  "college": "IIT Bombay",
  "branch": "Computer Science",
  "semester": "Sem 5"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "User registered successfully",
  "data": {
    "user": {
      "user_id": "8f882a93-5f09-4d64-9a1b-3ef5a4e51111",
      "name": "Rohan Sharma",
      "email": "rohan.sharma@iitb.ac.in",
      "role": "student",
      "college": "IIT Bombay",
      "branch": "Computer Science",
      "semester": "Sem 5"
    },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

### PYQ Trend Analysis
**Request:** `GET /api/pyqs/analysis/Operating%20Systems`

**Response (200 OK):**
```json
{
  "success": true,
  "message": "PYQ Analysis for Operating Systems",
  "data": {
    "subject": "Operating Systems",
    "papersAnalyzedCount": 4,
    "totalQuestionsSampled": 50,
    "predictions": [
      {
        "topic": "Process Scheduling & CPU Gantt Charts",
        "frequency": 14,
        "weightagePercentage": "28.0%",
        "priority": "High",
        "predictedOccurrence": "Almost Certain in Next Exam"
      },
      {
        "topic": "Deadlock Avoidance & Banker Algorithm",
        "frequency": 12,
        "weightagePercentage": "24.0%",
        "priority": "High",
        "predictedOccurrence": "Almost Certain in Next Exam"
      }
    ]
  }
}
```

---

## 🔒 8. Authentication & Security Flow

1. User sends credentials to `/api/auth/login`.
2. Password is verified against bcrypt hash (`password_hash`).
3. Backend issues signed JWT token containing `{ userId, email, role }`.
4. Client attaches token in header:
   ```text
   Authorization: Bearer <token>
   ```
5. `authMiddleware.js` verifies token, extracts user, and sets `req.user`.

---

## 🔄 9. Architecture & Integration Diagram

```
Frontend (React 19 / Vite - Port 5173)
   │
   ▼
main_backend (Node.js / Express - Port 5000)
   ├── /api/auth           (JWT + bcrypt authentication)
   ├── /api/users          (Academic profile & settings)
   ├── /api/notes          (Notes CRUD & public repository)
   ├── /api/pyqs           (PYQs & exam trend analysis)
   ├── /api/dashboard      (Metrics & active notebooks)
   │
   ├── MongoDB Database    (studysnap_db - 10 Collections)
   │
   └── chatbotService.js (HTTP / Axios)
          │
          ▼
   Existing Python Chatbot Backend (FastAPI - Port 8000)
          ├── ChromaDB (Vector store)
          └── LM Studio / Gemini LLM
```

---

## 🧪 10. Automated Testing

Run the automated test suite anytime with:
```bash
npm test
```
All routes, authentication flows, error handlers, and chatbot proxies are verified automatically.
