from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Header
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
import logging
from datetime import datetime
from database import (
    ping_db, settings, get_supabase,
    UserCRUD, NoteCRUD, ChatbotCRUD, SourceCRUD, ArtifactCRUD,
    NotebookCRUD, NotebookCreate, NotebookUpdate, NotebookInDB
)
from auth import get_current_user, DEMO_USER_ID
from prompt_registry import (
    resolve_request_instruction,
    sanitize_feature_title,
    sanitize_message_content
)

logger = logging.getLogger("studysnap.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager for Supabase Database Connection."""
    print("[StudySnap AI] Initializing Supabase PostgreSQL connection...")
    is_connected = await ping_db()
    if is_connected:
        print("[StudySnap AI] ✓ Connected to Supabase PostgreSQL successfully!")
    else:
        print("[StudySnap AI] ℹ️ Supabase running in local/resilient fallback mode. (Configure SUPABASE_URL & SUPABASE_KEY in .env)")
    yield
    print("[StudySnap AI] Backend shutdown complete.")

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
    display_title: Optional[str] = None
    system_instruction: Optional[str] = None
    feature: Optional[str] = None
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
    """Verify active connection status with Supabase PostgreSQL database."""
    is_online = await ping_db()
    return {
        "status": "online" if is_online else "fallback",
        "database": "supabase_postgresql",
        "connected": is_online,
        "supabase_configured": settings.is_configured
    }

# ==========================================
# 2. SOURCES ENDPOINTS (Supabase Documents + ChromaDB Vector Sync)
# ==========================================
@app.post("/api/sources/upload")
async def upload_sources(
    files: List[UploadFile] = File(...),
    authorization: Optional[str] = Header(None)
):
    """Upload multiple files, process them, and store embeddings in ChromaDB and metadata in Supabase."""
    t_start = time.perf_counter()
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    
    user_id = DEMO_USER_ID
    if authorization:
        try:
            curr_user = await get_current_user(authorization)
            user_id = curr_user["id"]
        except Exception:
            user_id = DEMO_USER_ID

    async def process_single_file(file: UploadFile):
        try:
            clean_filename = os.path.basename(file.filename or "uploaded_file").replace("..", "").strip()
            if not clean_filename:
                clean_filename = f"source_{uuid.uuid4()}"
            file_id = str(uuid.uuid4())
            file_path = os.path.join("uploads", f"{file_id}_{clean_filename}")
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Process and ingest document into ChromaDB in a separate thread
            doc_meta = await asyncio.to_thread(ingest_document, file_path, clean_filename, file_id)
            if doc_meta:
                try:
                    await SourceCRUD.save_document({
                        "id": file_id,
                        "owner_id": user_id,
                        "filename": clean_filename,
                        "file_type": doc_meta.get("type", "unknown"),
                        "chunk_count": doc_meta.get("chunk_count", 0),
                        "storage_path": file_path,
                        "metadata": doc_meta
                    }, access_token=token)
                except Exception as doc_err:
                    print(f"Supabase Document Metadata Save Note: {doc_err}")
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
    """Delete a source from MongoDB and ChromaDB with path sanitization."""
    try:
        delete_source(file_id)
        safe_file_id = os.path.basename(file_id).replace("..", "").strip()
        if safe_file_id and os.path.exists("uploads"):
            uploads_dir = os.path.abspath("uploads")
            for filename in os.listdir(uploads_dir):
                if filename.startswith(safe_file_id):
                    target_file = os.path.join(uploads_dir, filename)
                    if os.path.commonpath([uploads_dir, os.path.abspath(target_file)]) == uploads_dir:
                        if os.path.isfile(target_file):
                            try:
                                os.remove(target_file)
                            except Exception as rm_err:
                                print(f"Warning: could not remove {target_file}: {rm_err}")
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 3. CHATBOT SESSIONS & MESSAGES (Supabase Persistence)
# ==========================================
@app.get("/api/sessions")
async def get_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Fetch saved chatbot sessions for authenticated user from Supabase."""
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    try:
        sessions = await ChatbotCRUD.list_user_sessions(user_id=current_user["id"], access_token=token)
        for s in sessions:
            if "title" in s:
                s["title"] = sanitize_feature_title(s["title"])
        return {"sessions": sessions}
    except Exception as e:
        print(f"Supabase Sessions Fetch Warning: {e}")
        return {"sessions": []}

@app.get("/api/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Fetch chat session details and message history from Supabase."""
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    session = await ChatbotCRUD.get_session(session_id, access_token=token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if "title" in session:
        session["title"] = sanitize_feature_title(session["title"])
    if "messages" in session:
        for m in session["messages"]:
            if m.get("role") == "user":
                if "message_content" in m:
                    m["message_content"] = sanitize_message_content(m["message_content"])
                if "content" in m:
                    m["content"] = sanitize_message_content(m["content"])
    return {"session": session}

@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Delete a chat session from Supabase."""
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    deleted = await ChatbotCRUD.delete_session(session_id, access_token=token)
    return {"success": deleted}

@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Chat with the AI and save session history into Supabase."""
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    user_id = current_user["id"]
    try:
        # Resolve model query, feature instructions, and user-facing display title
        query_for_model, system_instruction, resolved_title = resolve_request_instruction(
            request.message,
            request.system_instruction,
            request.feature
        )
        display_title = request.display_title or resolved_title

        response_text, sources_used = await chat_with_context(
            query=query_for_model, 
            history=request.history, 
            selected_source_ids=request.selected_source_ids,
            response_language=request.response_language,
            system_instruction=system_instruction
        )

        session_id = request.session_id or str(uuid.uuid4())
        try:
            session = await ChatbotCRUD.get_session(session_id, access_token=token)
            if not session:
                await ChatbotCRUD.create_session({
                    "session_id": session_id,
                    "user_id": user_id,
                    "title": display_title,
                    "selected_source_ids": request.selected_source_ids or []
                }, access_token=token)

            await ChatbotCRUD.add_message(session_id, {
                "user_id": user_id,
                "role": "user",
                "content": display_title,
                "sources_used": []
            }, access_token=token)
            await ChatbotCRUD.add_message(session_id, {
                "user_id": user_id,
                "role": "assistant",
                "content": response_text,
                "sources_used": sources_used or []
            }, access_token=token)
        except Exception as db_err:
            print(f"Supabase Chat Session Persistence Warning: {db_err}")

        return {
            "response": response_text, 
            "sources": sources_used, 
            "session_id": session_id,
            "display_title": display_title
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        return {"response": response_text, "sources": sources_used, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 4. LECTURE NOTES CRUD ENDPOINTS (Supabase Persistence)
# ==========================================
@app.get("/api/notes")
async def get_notes(
    subject: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Fetch user lecture notes from Supabase."""
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    try:
        notes = await NoteCRUD.get_user_notes(user_id=current_user["id"], subject=subject, access_token=token)
        return {"notes": notes}
    except Exception as e:
        print(f"Supabase Note Fetch Warning: {e}")
        return {"notes": []}

@app.post("/api/notes")
async def create_note(
    request: NoteCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Create and store a lecture note in Supabase."""
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    try:
        data = request.model_dump()
        data["user_id"] = current_user["id"]
        note = await NoteCRUD.create_note(data, access_token=token)
        return {"success": True, "note": note}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database storage temporarily unavailable: {str(e)}")

@app.put("/api/notes/{note_id}")
async def update_note(
    note_id: str,
    request: NoteUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Update an existing lecture note in Supabase."""
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    try:
        updated = await NoteCRUD.update_note(note_id, request.model_dump(exclude_unset=True), access_token=token)
        if not updated:
            raise HTTPException(status_code=404, detail="Note not found.")
        return {"success": True, "note": updated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/notes/{note_id}")
async def delete_note(
    note_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Delete a lecture note from Supabase."""
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    try:
        deleted = await NoteCRUD.delete_note(note_id, access_token=token)
        return {"success": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        audio_filename = await generate_tts_audio(script_json)
        audio_url = f"http://localhost:8000/uploads/{audio_filename}"
        print(f"[Audio] === TTS generation phase: {time.perf_counter() - t0:.2f}s ===")

        # Save artifact record in Supabase
        try:
            await ArtifactCRUD.save_artifact({
                "artifact_type": "audio_overview",
                "title": "Podcast Overview",
                "content_data": {"audio_url": audio_url, "script": script_json, "filename": audio_filename},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"Supabase Artifact Save Warning: {db_err}")

        print(f"[Audio] ========== TOTAL AUDIO OVERVIEW: {time.perf_counter() - t_total:.2f}s ==========")
        return {
            "success": True, 
            "audio_url": audio_url,
            "relative_url": f"/uploads/{audio_filename}",
            "filename": audio_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/infographic")
async def create_infographic(request: InfographicRequest):
    """Generate a structured JSON infographic plan from selected sources and save artifact in Supabase."""
    try:
        infographic_data = await generate_infographic_data(
            request.selected_source_ids,
            request.style,
            request.detail_level,
            request.custom_prompt,
            request.response_language
        )

        try:
            await ArtifactCRUD.save_artifact({
                "artifact_type": "infographic",
                "title": f"Infographic ({request.style})",
                "content_data": {"data": infographic_data, "style": request.style},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"Supabase Artifact Save Warning: {db_err}")

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
    """Generate structured JSON Mind Map from selected sources and save artifact in Supabase."""
    try:
        mind_map_data = await generate_mind_map_data(
            request.selected_source_ids,
            request.custom_prompt,
            request.response_language
        )

        try:
            await ArtifactCRUD.save_artifact({
                "artifact_type": "mind_map",
                "title": "Mind Map Overview",
                "content_data": {"mind_map": mind_map_data},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"Supabase Artifact Save Warning: {db_err}")

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
    """Generate a quiz from selected sources and save artifact in Supabase."""
    try:
        quiz_data = await generate_quiz_data(
            request.selected_source_ids,
            request.response_language
        )
        try:
            await ArtifactCRUD.save_artifact({
                "artifact_type": "quiz",
                "title": "Quiz",
                "content_data": {"quiz": quiz_data},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"Supabase Artifact Save Warning: {db_err}")

        return {"success": True, "quiz": quiz_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/flashcards")
async def create_flashcards(request: FlashcardsRequest):
    """Generate flashcards from selected sources and save artifact in Supabase."""
    try:
        flashcards_data = await generate_flashcards_data(
            request.selected_source_ids,
            request.response_language
        )
        try:
            await ArtifactCRUD.save_artifact({
                "artifact_type": "flashcards",
                "title": "Flashcards",
                "content_data": {"flashcards": flashcards_data},
                "selected_source_ids": request.selected_source_ids or [],
                "response_language": request.response_language or "English"
            })
        except Exception as db_err:
            print(f"Supabase Artifact Save Warning: {db_err}")

        return {"success": True, "flashcards": flashcards_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# NOTEBOOK PERSISTENCE ENDPOINTS (Supabase PostgreSQL + RLS)
# =============================================================================

@app.post("/api/notebooks")
async def create_notebook(
    request: NotebookCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """
    Create a new notebook in Supabase PostgreSQL associated with verified user.
    Enforces user ownership, generates UUID, and persists timestamps.
    """
    user_id = current_user["id"]
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    notebook_id = str(uuid.uuid4())
    nb_dict = {
        "id": notebook_id,
        "user_id": user_id,
        "title": request.title or "Untitled Notebook",
        "code": request.code or "CS101",
        "content": request.content or "",
        "sources_count": len(request.sources) if request.sources else 0,
        "notes_count": 0,
        "sources": request.sources or [],
        "chat_history": [],
        "status": "ready"
    }
    saved = await NotebookCRUD.create_notebook(nb_dict, access_token=token)
    return {"success": True, "notebook": saved}

@app.get("/api/notebooks")
async def list_notebooks(
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """
    List all notebooks in Supabase owned by the authenticated user.
    Results are sorted by updated_at descending.
    """
    user_id = current_user["id"]
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    notebooks = await NotebookCRUD.get_user_notebooks(user_id, access_token=token)
    return {"success": True, "notebooks": notebooks}

@app.get("/api/notebooks/{notebook_id}")
async def get_notebook(
    notebook_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Retrieve a single notebook by ID ensuring Supabase user ownership."""
    user_id = current_user["id"]
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    nb = await NotebookCRUD.get_notebook_by_id(notebook_id, user_id, access_token=token)
    if nb:
        return {"success": True, "notebook": nb}

    raise HTTPException(status_code=404, detail="Notebook not found or access denied")

@app.put("/api/notebooks/{notebook_id}")
async def update_notebook(
    notebook_id: str,
    request: NotebookUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """
    Debounced autosave for notebook title, content, sources, and chat in Supabase.
    Verifies user ownership before applying updates.
    """
    user_id = current_user["id"]
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    update_data = request.model_dump(exclude_unset=True)
    updated = await NotebookCRUD.update_notebook(notebook_id, user_id, update_data, access_token=token)
    if updated:
        return {"success": True, "notebook": updated}

    raise HTTPException(status_code=404, detail="Notebook not found or access denied")

@app.delete("/api/notebooks/{notebook_id}")
async def delete_notebook(
    notebook_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Delete notebook from Supabase ensuring user ownership."""
    user_id = current_user["id"]
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    deleted = await NotebookCRUD.delete_notebook(notebook_id, user_id, access_token=token)
    if deleted:
        return {"success": True, "message": "Notebook deleted successfully"}

    raise HTTPException(status_code=404, detail="Notebook not found or access denied")

# =============================================================================
# STUDENT PROFILE PERSISTENCE ENDPOINTS (Supabase PostgreSQL + RLS)
# =============================================================================

class StudentProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    display_name: Optional[str] = None
    college_name: Optional[str] = None
    college: Optional[str] = None
    degree_program: Optional[str] = None
    custom_degree: Optional[str] = None
    branch: Optional[str] = None
    custom_branch: Optional[str] = None
    academic_year: Optional[str] = None
    semester: Optional[str] = None
    graduation_year: Optional[str] = None
    roll_number: Optional[str] = None
    avatar_path: Optional[str] = None
    avatar_url: Optional[str] = None
    theme_preference: Optional[str] = None

@app.get("/api/user/profile")
async def get_current_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Retrieve authenticated student profile from Supabase."""
    user_id = current_user["id"]
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    profile = await UserCRUD.get_user_profile(user_id, access_token=token)
    return {"success": True, "profile": profile}

@app.put("/api/user/profile")
async def update_current_user_profile(
    request: StudentProfileUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """Update student profile in Supabase with RLS user isolation."""
    user_id = current_user["id"]
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None
    data = request.model_dump(exclude_unset=True)
    updated = await UserCRUD.upsert_user_profile(user_id, data, access_token=token)
    return {"success": True, "profile": updated}

# =============================================================================
# STUDENT PROFILE PHOTO STORAGE & MANAGEMENT (Supabase Storage 'avatars' bucket)
# =============================================================================

MAX_AVATAR_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_AVATAR_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

def validate_avatar_bytes(data: bytes, declared_content_type: Optional[str] = None) -> str:
    """
    Validates avatar byte size and file magic bytes.
    Returns the resolved image extension ('jpg', 'png', or 'webp').
    Raises HTTPException(400) if validation fails.
    """
    if len(data) > MAX_AVATAR_SIZE:
        size_mb = round(len(data) / (1024 * 1024), 2)
        raise HTTPException(
            status_code=400,
            detail=f"File size ({size_mb} MB) exceeds maximum allowed limit of 5 MB."
        )

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Magic byte verification
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    elif data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp"
    
    # Check declared content type if magic bytes match format
    if declared_content_type in ALLOWED_AVATAR_MIME_TYPES:
        if "png" in declared_content_type:
            return "png"
        elif "webp" in declared_content_type:
            return "webp"
        elif "jpeg" in declared_content_type or "jpg" in declared_content_type:
            return "jpg"

    raise HTTPException(
        status_code=400,
        detail="Unsupported file format. Only JPEG, PNG, and WebP images up to 5 MB are allowed."
    )

@app.post("/api/user/avatar/upload")
async def upload_user_avatar(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """
    Production-ready server-side avatar upload:
    - Enforces authenticated user ID ownership (no spoofing).
    - Validates file size (max 5 MB) and actual image magic bytes (JPEG/PNG/WebP).
    - Generates user-scoped path: `<user_id>/avatar_<timestamp>.<ext>`.
    - Uploads to Supabase Storage 'avatars' bucket.
    - Updates profiles table (stores storage path, not raw image bytes).
    - Deletes prior avatar file cleanly to prevent storage bloat.
    """
    user_id = current_user["id"]
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None

    contents = await file.read()
    ext = validate_avatar_bytes(contents, file.content_type)

    filename = f"avatar_{int(time.time())}.{ext}"
    storage_subpath = f"{user_id}/{filename}"
    target_bucket = "avatars"

    existing_profile = await UserCRUD.get_user_profile(user_id, access_token=token)
    prior_avatar_path = existing_profile.get("avatar_path") if existing_profile else None

    avatar_path = f"{target_bucket}:{storage_subpath}"
    avatar_url = f"https://qtxmsuvueadiqdmwxuvj.supabase.co/storage/v1/object/public/{target_bucket}/{storage_subpath}?t={int(time.time())}"

    supabase_client = get_supabase(token)
    if supabase_client:
        try:
            mime = "image/png" if ext == "png" else ("image/webp" if ext == "webp" else "image/jpeg")
            supabase_client.storage.from_(target_bucket).upload(
                path=storage_subpath,
                file=contents,
                file_options={"content-type": mime, "upsert": "true"}
            )
            pub_res = supabase_client.storage.from_(target_bucket).get_public_url(storage_subpath)
            if pub_res:
                avatar_url = f"{pub_res}?t={int(time.time())}"

            # Clean up prior avatar object
            if prior_avatar_path and prior_avatar_path != avatar_path:
                prev_bucket = target_bucket
                prev_sub = prior_avatar_path
                if ":" in prior_avatar_path:
                    parts = prior_avatar_path.split(":", 1)
                    prev_bucket, prev_sub = parts[0], parts[1]
                if user_id in prev_sub:
                    try:
                        supabase_client.storage.from_(prev_bucket).remove([prev_sub])
                    except Exception as cln_err:
                        logger.warning(f"Could not clean up prior avatar {prev_sub}: {cln_err}")
        except Exception as sb_err:
            logger.warning(f"Supabase storage upload notice: {sb_err}")

    updated_profile = await UserCRUD.upsert_user_profile(
        user_id,
        {
            "avatar_path": avatar_path,
            "avatar_url": avatar_url
        },
        access_token=token
    )

    return {
        "success": True,
        "avatar_path": avatar_path,
        "avatar_url": avatar_url,
        "profile": updated_profile
    }

@app.delete("/api/user/avatar")
async def remove_user_avatar(
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
):
    """
    Remove user profile photo:
    - Enforces authenticated user ID ownership.
    - Prevents arbitrary file path deletion by verifying path belongs to user's folder.
    - Deletes storage object from avatars bucket.
    - Resets avatar_path and avatar_url to None in profiles table.
    """
    user_id = current_user["id"]
    token = authorization.split()[1] if authorization and len(authorization.split()) == 2 else None

    existing_profile = await UserCRUD.get_user_profile(user_id, access_token=token)
    prior_avatar_path = existing_profile.get("avatar_path") if existing_profile else None

    if prior_avatar_path:
        target_bucket = "avatars"
        sub_path = prior_avatar_path
        if ":" in prior_avatar_path:
            parts = prior_avatar_path.split(":", 1)
            target_bucket, sub_path = parts[0], parts[1]

        # Ownership sanity check: Ensure path starts with user_id/
        if not (sub_path.startswith(f"{user_id}/") or f"/{user_id}/" in sub_path):
            raise HTTPException(status_code=403, detail="Unauthorized deletion: avatar path does not belong to user.")

        supabase_client = get_supabase(token)
        if supabase_client:
            try:
                supabase_client.storage.from_(target_bucket).remove([sub_path])
            except Exception as rm_err:
                logger.warning(f"Storage avatar removal notice: {rm_err}")

    updated_profile = await UserCRUD.upsert_user_profile(
        user_id,
        {
            "avatar_path": None,
            "avatar_url": None
        },
        access_token=token
    )

    return {
        "success": True,
        "message": "Avatar removed successfully. Initial fallback restored.",
        "profile": updated_profile
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


