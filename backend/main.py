from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from rag import (
    ingest_document, chat_with_context, get_all_sources, delete_source,
    generate_podcast_script, generate_tts_audio, generate_infographic_data,
    generate_mind_map_data, generate_quiz_data, generate_flashcards_data
)
from comfy_client import generate_image, get_image_bytes, check_comfyui_connection
from fastapi.responses import Response, StreamingResponse
import queue
import threading
import json
import asyncio
import time
from database import (
    get_async_db, ping_db, db_manager,
    UserCRUD, NoteCRUD, AudioCRUD, PYQCRUD, ChatbotCRUD, SourceCRUD, ArtifactCRUD
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager for MongoDB Database Connection."""
    # Startup: Connect to MongoDB
    print("[StudySnap AI] Connecting to MongoDB Database...")
    db = get_async_db()
    is_connected = await ping_db()
    if is_connected:
        print("[StudySnap AI] ✓ Connected to MongoDB successfully!")
    else:
        print("[StudySnap AI] ⚠️ Warning: MongoDB connection failed or offline. Check MONGODB_URL in .env")
    yield
    # Shutdown: Close database connections
    print("[StudySnap AI] Closing MongoDB database connections...")
    db_manager.close()

app = FastAPI(title="StudySnap AI Backend", lifespan=lifespan)

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

os.makedirs("public/videos", exist_ok=True)
app.mount("/videos", StaticFiles(directory="public/videos"), name="videos")

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    history: List[dict] = []
    selected_source_ids: Optional[List[str]] = None
    response_language: Optional[str] = "English"

class AudioOverviewRequest(BaseModel):
    selected_source_ids: Optional[List[str]] = None
    response_language: Optional[str] = "English"

class InfographicRequest(BaseModel):
    selected_source_ids: Optional[List[str]] = None
    style: str = "Bento Grid"
    detail_level: str = "Standard"
    custom_prompt: str = ""
    response_language: Optional[str] = "English"

class MindMapRequest(BaseModel):
    selected_source_ids: Optional[List[str]] = None
    custom_prompt: str = ""
    response_language: Optional[str] = "English"

class QuizRequest(BaseModel):
    selected_source_ids: Optional[List[str]] = None
    response_language: Optional[str] = "English"

class FlashcardsRequest(BaseModel):
    selected_source_ids: Optional[List[str]] = None
    response_language: Optional[str] = "English"

class VideoOverviewRequest(BaseModel):
    selected_source_ids: Optional[List[str]] = None
    custom_prompt: str = ""
    response_language: Optional[str] = "English"

class UrlRequest(BaseModel):
    url: str

class NoteCreateRequest(BaseModel):
    user_id: str = "default_user"
    subject: str
    chapter: str
    formatted_content: str
    summary: Optional[str] = ""
    keywords: Optional[List[str]] = []
    duration: Optional[float] = 0.0

class NoteUpdateRequest(BaseModel):
    subject: Optional[str] = None
    chapter: Optional[str] = None
    formatted_content: Optional[str] = None
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None

# ==========================================
# 1. DATABASE HEALTH ENDPOINT
# ==========================================
@app.get("/api/health/db")
async def db_health():
    """Verify active connection status with MongoDB database."""
    is_online = await ping_db()
    if is_online:
        return {"status": "online", "database": "studysnap_db", "connected": True}
    return {"status": "offline", "database": "studysnap_db", "connected": False}

# ==========================================
# 2. SOURCES ENDPOINTS (MongoDB + ChromaDB Sync)
# ==========================================
@app.post("/api/sources/upload")
async def upload_sources(files: List[UploadFile] = File(...)):
    """Upload multiple files, process them, and store embeddings in ChromaDB and metadata in MongoDB."""
    t_start = time.perf_counter()
    
    async def process_single_file(file: UploadFile):
        try:
            file_id = str(uuid.uuid4())
            file_path = f"uploads/{file_id}_{file.filename}"
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Process and ingest document into ChromaDB & MongoDB in a separate thread
            doc_meta = await asyncio.to_thread(ingest_document, file_path, file.filename, file_id)
            return doc_meta
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            return None

    tasks = [process_single_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    
    uploaded_sources = [res for res in results if res is not None]
            
    if not uploaded_sources:
        raise HTTPException(status_code=400, detail="Failed to process any of the uploaded files.")
        
    print(f"[Audio] Total concurrent upload time: {time.perf_counter() - t_start:.2f}s")
    return {"success": True, "sources": uploaded_sources}

@app.post("/api/sources/url")
async def add_url_source(request: UrlRequest):
    """Add a URL or YouTube link as a source, streaming status updates."""
    url = request.url
    file_id = str(uuid.uuid4())
    q = queue.Queue()
    
    def sync_callback(msg: str):
        q.put(msg)
        
    def worker():
        try:
            # ingest_document uses the callback to stream {"status": ...}
            # and when done, it sends {"success": True, "source": ...}
            doc_meta = ingest_document("", url, file_id, status_callback=sync_callback)
            sync_callback(json.dumps({"success": True, "source": doc_meta}) + "\n")
        except Exception as e:
            sync_callback(json.dumps({"success": False, "detail": str(e)}) + "\n")
        finally:
            q.put(None) # Sentinel

    threading.Thread(target=worker, daemon=True).start()
    
    async def event_generator():
        while True:
            # wait in an executor so we don't block event loop
            msg = await asyncio.to_thread(q.get)
            if msg is None:
                break
            yield msg
            
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/api/sources")
async def list_sources():
    """List all ingested sources from MongoDB / ChromaDB."""
    sources = get_all_sources()
    return {"sources": sources}

@app.delete("/api/sources/{file_id}")
async def remove_source(file_id: str):
    """Delete a source from MongoDB and ChromaDB."""
    try:
        delete_source(file_id)
        for filename in os.listdir("uploads"):
            if filename.startswith(file_id):
                os.remove(os.path.join("uploads", filename))
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 3. CHATBOT SESSIONS & MESSAGES (MongoDB Persistence)
# ==========================================
@app.get("/api/sessions")
async def get_sessions(user_id: str = "default_user"):
    """Fetch saved chatbot sessions for a user from MongoDB."""
    db = get_async_db()
    sessions = await ChatbotCRUD.list_user_sessions(db, user_id=user_id)
    return {"sessions": [s.model_dump() for s in sessions]}

@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Fetch chat session details and message history from MongoDB."""
    db = get_async_db()
    session = await ChatbotCRUD.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session": session.model_dump()}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session from MongoDB."""
    db = get_async_db()
    deleted = await ChatbotCRUD.delete_session(db, session_id)
    return {"success": deleted}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with the AI and save session history into MongoDB."""
    try:
        response_text, sources_used = await chat_with_context(
            request.message, 
            request.history, 
            request.selected_source_ids,
            request.response_language
        )

        db = get_async_db()
        # Save message into MongoDB session history if session_id provided or create session
        session_id = request.session_id
        if session_id:
            session = await ChatbotCRUD.get_session(db, session_id)
            if not session:
                await ChatbotCRUD.create_session(db, {
                    "session_id": session_id,
                    "title": request.message[:30] + ("..." if len(request.message) > 30 else ""),
                    "selected_source_ids": request.selected_source_ids or []
                })
        else:
            session_id = str(uuid.uuid4())
            await ChatbotCRUD.create_session(db, {
                "session_id": session_id,
                "title": request.message[:30] + ("..." if len(request.message) > 30 else ""),
                "selected_source_ids": request.selected_source_ids or []
            })

        # Save user query and assistant response in MongoDB
        await ChatbotCRUD.add_message(db, session_id, {
            "role": "user",
            "content": request.message,
            "sources_used": []
        })
        await ChatbotCRUD.add_message(db, session_id, {
            "role": "assistant",
            "content": response_text,
            "sources_used": sources_used or []
        })

        return {"response": response_text, "sources": sources_used, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 4. LECTURE NOTES CRUD ENDPOINTS (MongoDB Persistence)
# ==========================================
@app.get("/api/notes")
async def get_notes(user_id: str = "default_user", subject: Optional[str] = None):
    """Fetch user lecture notes from MongoDB."""
    db = get_async_db()
    notes = await NoteCRUD.get_user_notes(db, user_id=user_id, subject=subject)
    return {"notes": [n.model_dump() for n in notes]}

@app.post("/api/notes")
async def create_note(request: NoteCreateRequest):
    """Create and store a lecture note in MongoDB."""
    db = get_async_db()
    note = await NoteCRUD.create_note(db, request.model_dump())
    return {"success": True, "note": note.model_dump()}

@app.put("/api/notes/{note_id}")
async def update_note(note_id: str, request: NoteUpdateRequest):
    """Update an existing lecture note in MongoDB."""
    db = get_async_db()
    updated = await NoteCRUD.update_note(db, note_id, request.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"success": True, "note": updated.model_dump()}

@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a lecture note from MongoDB."""
    db = get_async_db()
    deleted = await NoteCRUD.delete_note(db, note_id)
    return {"success": deleted}

# ==========================================
# 5. AI GENERATED ARTIFACTS ENDPOINTS
# ==========================================
@app.post("/api/audio-overview")
async def create_audio_overview(request: AudioOverviewRequest):
    """Generate a podcast-style audio overview from selected sources and save artifact in MongoDB."""
    try:
        t_total = time.perf_counter()
        
        t0 = time.perf_counter()
        script_json = await generate_podcast_script(request.selected_source_ids, request.response_language)
        print(f"[Audio] === Script generation phase: {time.perf_counter() - t0:.2f}s ===")
        
        t0 = time.perf_counter()
        audio_url = f"http://localhost:8000/uploads/{audio_filename}"
        print(f"[Audio] === TTS generation phase: {time.perf_counter() - t0:.2f}s ===")

        # Save artifact record in MongoDB
        try:
            db = get_async_db()
            await ArtifactCRUD.save_artifact(db, {
                "artifact_type": "audio_overview",
                "title": "Podcast Overview",
                "content_data": {"audio_url": audio_url, "script": script_json},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"MongoDB Artifact Save Warning: {db_err}")

        print(f"[Audio] ========== TOTAL AUDIO OVERVIEW: {time.perf_counter() - t_total:.2f}s ==========")
        return {"success": True, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/infographic")
async def create_infographic(request: InfographicRequest):
    """Generate a structured JSON infographic plan from selected sources and save artifact in MongoDB."""
    try:
        infographic_data = await generate_infographic_data(
            request.selected_source_ids,
            request.style,
            request.detail_level,
            request.custom_prompt,
            request.response_language
        )

        try:
            db = get_async_db()
            await ArtifactCRUD.save_artifact(db, {
                "artifact_type": "infographic",
                "title": f"Infographic ({request.style})",
                "content_data": {"data": infographic_data, "style": request.style},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"MongoDB Artifact Save Warning: {db_err}")

        return {"success": True, "data": infographic_data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ImageRequest(BaseModel):
    prompt: str

@app.post("/api/generate-image")
async def api_generate_image(request: ImageRequest):
    try:
        filename = await generate_image(request.prompt)
        return {"success": True, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image/{filename}")
async def api_get_image(filename: str):
    try:
        image_bytes = await get_image_bytes(filename)
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/status")
async def api_status():
    comfy_status = await check_comfyui_connection()
    return {
        "lm_studio": True, # Assuming true since the server is running if we get here (rag handles errors)
        "comfyui": comfy_status
    }

@app.post("/api/mindmap")
async def create_mind_map(request: MindMapRequest):
    """Generate structured JSON Mind Map from selected sources and save artifact in MongoDB."""
    try:
        mind_map_data = await generate_mind_map_data(
            request.selected_source_ids,
            request.custom_prompt,
            request.response_language
        )

        try:
            db = get_async_db()
            await ArtifactCRUD.save_artifact(db, {
                "artifact_type": "mind_map",
                "title": "Mind Map Overview",
                "content_data": {"mind_map": mind_map_data},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"MongoDB Artifact Save Warning: {db_err}")

        return {"success": True, "mind_map": mind_map_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video-overview")
async def create_video_overview(request: VideoOverviewRequest):
    """Generate a video overview from selected sources."""
    try:
        return {"success": False, "detail": "Video overview generation is currently not available."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quiz")
async def create_quiz(request: QuizRequest):
    """Generate a quiz from selected sources."""
    try:
        quiz_data = await generate_quiz_data(
            request.selected_source_ids,
            request.response_language
        )
        try:
            db = get_async_db()
            await ArtifactCRUD.save_artifact(db, {
                "artifact_type": "quiz",
                "title": "Quiz",
                "content_data": {"quiz": quiz_data},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"MongoDB Artifact Save Warning: {db_err}")

        return {"success": True, "quiz": quiz_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/flashcards")
async def create_flashcards(request: FlashcardsRequest):
    """Generate flashcards from selected sources."""
    try:
        flashcards_data = await generate_flashcards_data(
            request.selected_source_ids,
            request.response_language
        )
        try:
            db = get_async_db()
            await ArtifactCRUD.save_artifact(db, {
                "artifact_type": "flashcards",
                "title": "Flashcards",
                "content_data": {"flashcards": flashcards_data},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"MongoDB Artifact Save Warning: {db_err}")

        return {"success": True, "flashcards": flashcards_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
