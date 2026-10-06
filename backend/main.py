from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import shutil
import uuid
from fastapi.staticfiles import StaticFiles
from rag import ingest_document, chat_with_context, get_all_sources, delete_source, generate_podcast_script, generate_tts_audio, generate_infographic_data, generate_mind_map_data, generate_quiz_data, generate_flashcards_data
from comfy_client import generate_image, get_image_bytes, check_comfyui_connection
from fastapi.responses import Response

app = FastAPI(title="StudySnap AI Backend")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

os.makedirs("public/videos", exist_ok=True)
app.mount("/videos", StaticFiles(directory="public/videos"), name="videos")
class ChatRequest(BaseModel):
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

import asyncio
import time

@app.post("/api/sources/upload")
async def upload_sources(files: List[UploadFile] = File(...)):
    """Upload multiple files, process them, and store embeddings in ChromaDB."""
    t_start = time.perf_counter()
    
    async def process_single_file(file: UploadFile):
        try:
            file_id = str(uuid.uuid4())
            file_path = f"uploads/{file_id}_{file.filename}"
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Process and ingest document into ChromaDB in a separate thread
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

from fastapi.responses import StreamingResponse
import queue
import threading
import json
import asyncio

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
    """List all ingested sources from the database."""
    sources = get_all_sources()
    return {"sources": sources}

@app.delete("/api/sources/{file_id}")
async def remove_source(file_id: str):
    """Delete a source from ChromaDB."""
    try:
        delete_source(file_id)
        # We should also try to delete the local file if we stored it
        for filename in os.listdir("uploads"):
            if filename.startswith(file_id):
                os.remove(os.path.join("uploads", filename))
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with the AI using context retrieved from ChromaDB via LM Studio."""
    try:
        response_text, sources_used = await chat_with_context(
            request.message, 
            request.history, 
            request.selected_source_ids,
            request.response_language
        )
        return {"response": response_text, "sources": sources_used}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audio-overview")
async def create_audio_overview(request: AudioOverviewRequest):
    """Generate a podcast-style audio overview from selected sources."""
    try:
        t_total = time.perf_counter()
        
        t0 = time.perf_counter()
        script_json = await generate_podcast_script(request.selected_source_ids, request.response_language)
        print(f"[Audio] === Script generation phase: {time.perf_counter() - t0:.2f}s ===")
        
        t0 = time.perf_counter()
        audio_filename = await generate_tts_audio(script_json)
        print(f"[Audio] === TTS generation phase: {time.perf_counter() - t0:.2f}s ===")
        
        print(f"[Audio] ========== TOTAL AUDIO OVERVIEW: {time.perf_counter() - t_total:.2f}s ==========")
        return {"success": True, "audio_url": f"http://localhost:8000/uploads/{audio_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/infographic")
async def create_infographic(request: InfographicRequest):
    """Generate a structured JSON infographic plan from selected sources."""
    try:
        infographic_data = await generate_infographic_data(
            request.selected_source_ids,
            request.style,
            request.detail_level,
            request.custom_prompt,
            request.response_language
        )
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
    """Generate structured JSON Mind Map from selected sources."""
    try:
        mind_map_data = await generate_mind_map_data(
            request.selected_source_ids,
            request.custom_prompt,
            request.response_language
        )
        return {"success": True, "mind_map": mind_map_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video-overview")
async def create_video_overview(request: VideoOverviewRequest):
    """Generate a video overview from selected sources."""
    try:
        video_url = await generate_video_overview(
            request.selected_source_ids,
            request.response_language,
            request.custom_prompt
        )
        return {"success": True, "video_url": video_url}
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
        return {"success": True, "flashcards": flashcards_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
