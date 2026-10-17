import os
import json
import csv
# pyrefly: ignore [missing-import]
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
import io
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import PIL.Image
from docx import Document
from gtts import gTTS
from pptx import Presentation
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from chromadb.utils import embedding_functions
from openai import AsyncOpenAI
from pypdf import PdfReader
import re
import asyncio
import edge_tts
import uuid
from pydub import AudioSegment
import urllib.parse
import aiohttp

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_data")
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="studysnap_collection", embedding_function=emb_fn)

# Initialize LM Studio client
lm_studio_client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
MODEL_NAME = "my_custom_model_guff/Anish/StudySanpAI" # For 8B model use "my_custom_model/Anish/StudySanpAI"

def _parse_youtube_video_id(url: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    video_id = ""
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    elif "embed/" in url:
        video_id = url.split("embed/")[1].split("?")[0]
    if not video_id:
        raise ValueError("Could not find a valid video ID in the URL.")
    return video_id


def _try_caption_extraction(video_id: str) -> str:
    """Attempt to retrieve captions via youtube_transcript_api. Returns text or raises."""
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript_list = ytt_api.list(video_id)
        transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'hi'])
        data = transcript.fetch()
    except Exception:
        try:
            transcript_list = ytt_api.list(video_id)
            transcript = list(transcript_list)[0]
            data = transcript.fetch()
        except Exception:
            data = ytt_api.fetch(video_id)

    return " ".join([t.text if hasattr(t, 'text') else t.get('text', '') for t in data])


def _try_audio_fallback(url: str, status_callback=None) -> str:
    """Download audio via yt-dlp and transcribe with Whisper. Returns text or raises."""
    import subprocess
    t0 = time.perf_counter()
    temp_id = str(uuid.uuid4())[:8]
    os.makedirs("uploads", exist_ok=True)
    output_template = os.path.join("uploads", f"yt_audio_{temp_id}.%(ext)s")
    output_path = None

    try:
        if status_callback:
            status_callback(json.dumps({"status": "Captions unavailable. Extracting audio..."}) + "\n")
            
        # Download audio-only using yt-dlp
        # Try twice: first without cookies, then with browser cookies for 403 errors
        base_cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "5",       # lower quality = smaller file = faster transcription
            "--max-filesize", "50m",
            "--output", output_template,
            "--quiet",
            "--no-warnings",
            url,
        ]

        result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=120)

        # If first attempt got 403/auth error, retry with browser cookies
        if result.returncode != 0:
            stderr_first = result.stderr.strip()
            print(f"[YouTube] yt-dlp attempt 1 failed: {stderr_first[:200]}")

            if "403" in stderr_first or "sign in" in stderr_first.lower() or "forbidden" in stderr_first.lower():
                print(f"[YouTube] Retrying with browser cookies...")
                cmd_with_cookies = base_cmd[:-1] + ["--cookies-from-browser", "chrome", url]
                result = subprocess.run(cmd_with_cookies, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            stderr = result.stderr.strip()
            print(f"[YouTube] yt-dlp final failure: {stderr[:300]}")
            if "private" in stderr.lower() or "sign in" in stderr.lower():
                raise ValueError("This video is private or requires sign-in.")
            elif "unavailable" in stderr.lower() or "not available" in stderr.lower():
                raise ValueError("This video is unavailable.")
            else:
                raise ValueError(f"Could not extract audio from this video.")

        # Find the downloaded file
        for f_name in os.listdir("uploads"):
            if f_name.startswith(f"yt_audio_{temp_id}"):
                output_path = os.path.join("uploads", f_name)
                break

        if not output_path or not os.path.exists(output_path):
            raise ValueError("Audio download succeeded but output file not found.")

        print(f"[YouTube] Audio downloaded in {time.perf_counter() - t0:.2f}s → {output_path}")

        if status_callback:
            status_callback(json.dumps({"status": "Transcribing audio..."}) + "\n")

        # Transcribe using existing Whisper pipeline
        t1 = time.perf_counter()
        text = extract_audio(output_path)
        print(f"[YouTube] Whisper transcription: {time.perf_counter() - t1:.2f}s")

        if not text or text.startswith("Audio transcription failed"):
            raise ValueError("Audio transcription returned no usable text.")

        return text

    finally:
        # Cleanup temp audio file
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass


def extract_youtube_transcript(url: str, status_callback=None) -> str:
    """Extract transcript from YouTube video.
    Strategy: try captions first, then fallback to audio download + Whisper transcription.
    """
    t_total = time.perf_counter()
    video_id = _parse_youtube_video_id(url)
    print(f"[YouTube] Processing video: {video_id}")

    if status_callback:
        status_callback(json.dumps({"status": "Checking for captions..."}) + "\n")

    # ── Stage 1: Try caption extraction ──
    try:
        t0 = time.perf_counter()
        text = _try_caption_extraction(video_id)
        print(f"[YouTube] Captions retrieved in {time.perf_counter() - t0:.2f}s ({len(text)} chars)")
        print(f"[YouTube] Total: {time.perf_counter() - t_total:.2f}s")
        return text
    except Exception as caption_err:
        print(f"[YouTube] Caption extraction failed: {caption_err}")
        print(f"[YouTube] Falling back to audio transcription...")

    # ── Stage 2: Audio download + Whisper fallback ──
    try:
        text = _try_audio_fallback(url, status_callback)
        print(f"[YouTube] Audio fallback succeeded ({len(text)} chars)")
        print(f"[YouTube] Total: {time.perf_counter() - t_total:.2f}s")
        return text
    except Exception as audio_err:
        print(f"[YouTube] Audio fallback also failed: {audio_err}")
        raise ValueError(
            "We couldn't retrieve captions or extract audio from this YouTube video. "
            "The video may be private, unavailable, restricted, or inaccessible. "
            "Please try another video or upload the audio directly."
        )

def extract_webpage(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.extract()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"Webpage Extraction Error: {e}")
        return ""

import whisper

import hashlib
import time
import asyncio

# ── Centralized language instruction for AI prompts ──────────────────────
SUPPORTED_LANGUAGES = {"english", "hinglish"}

def get_language_definition(response_language: str) -> str:
    """Return the parenthetical language definition used in AI prompts."""
    lang = (response_language or "english").lower()
    if lang == "hinglish":
        return (
            " (natural Hinglish — a conversational mix of Hindi and English "
            "written entirely in Roman/English script. Mix Hindi and English naturally "
            "the way Indian college students speak. Keep technical terms, formulas, "
            "and key English terminology in English. Do NOT use Devanagari script.)"
        )
    elif lang == "manglish":
        return " (a mix of Malayalam and English written in the English alphabet)"
    return ""

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        t0 = time.perf_counter()
        import whisper
        _whisper_model = whisper.load_model("tiny")
        print(f"[Audio] Whisper 'tiny' model loaded in {time.perf_counter() - t0:.2f}s")
    return _whisper_model

def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_audio(file_path: str) -> str:
    """Extract text from audio using local Whisper instance."""
    try:
        t_start = time.perf_counter()
        
        # Check cache
        cache_dir = "uploads/cache"
        os.makedirs(cache_dir, exist_ok=True)
        file_hash = get_file_hash(file_path)
        cache_path = os.path.join(cache_dir, f"{file_hash}.txt")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"[Audio] Upload & Transcription (CACHED): {time.perf_counter() - t_start:.2f}s")
            return text
            
        t0 = time.perf_counter()
        model = get_whisper_model()
        print(f"[Audio] Model retrieval: {time.perf_counter() - t0:.2f}s")
        
        t0 = time.perf_counter()
        result = model.transcribe(file_path, fp16=False)
        text = result.get("text", "")
        print(f"[Audio] Transcription execution: {time.perf_counter() - t0:.2f}s")
        
        # Save to cache
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        print(f"[Audio] Total Audio Extraction: {time.perf_counter() - t_start:.2f}s")
        return text
    except Exception as e:
        print(f"Whisper Audio Extraction Error: {e}")
        return "Audio transcription failed. Please ensure ffmpeg is installed."

def extract_image_text(file_path: str) -> str:
    """Extract text and descriptions from images using Gemini Vision."""
    try:
        img = PIL.Image.open(file_path)
        # Using gemini-1.5-flash as it is fast and supports vision
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            "Extract all the text from this image exactly as written. If there are any diagrams, tables, or charts, describe them in detail so that a student can understand them without seeing the image.", 
            img
        ])
        return response.text
    except Exception as e:
        print(f"[Image Extraction] Error: {e}")
        raise ValueError("Could not extract text from this image.")

def extract_text(file_path: str, filename: str) -> str:
    """Extract text from all supported file types."""
    text = ""
    ext = filename.split('.')[-1].lower() if '.' in filename else ""
    
    try:
        if ext == 'pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                if page_text := page.extract_text(): text += page_text + "\n"
        elif ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
            text = extract_image_text(file_path)
        elif ext in ['txt', 'md', 'html', 'xml']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if ext in ['html', 'xml']:
                    soup = BeautifulSoup(content, 'html.parser')
                    text = soup.get_text(separator="\n", strip=True)
                else:
                    text = content
        elif ext == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
        elif ext == 'csv':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    text += " ".join(row) + "\n"
        elif ext == 'xlsx':
            df = pd.read_excel(file_path)
            text = df.to_string()
        elif ext == 'docx':
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == 'pptx':
            prs = Presentation(file_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
        elif ext in ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'opus', 'aac']:
            text = extract_audio(file_path)
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_document(file_path: str, filename: str, file_id: str, status_callback=None):
    text = ""
    # Check if it's a URL or YouTube link based on filename format or type passed from main
    if filename.startswith("http"):
        if "youtube.com" in filename or "youtu.be" in filename:
            text = extract_youtube_transcript(filename, status_callback)
            filename = f"YouTube: {filename}"
        else:
            if status_callback:
                status_callback(json.dumps({"status": "Extracting webpage..."}) + "\n")
            text = extract_webpage(filename)
            filename = f"Web: {filename}"
    else:
        text = extract_text(file_path, filename)
        
    if not text.strip():
        raise ValueError("Could not extract any text from the document.")

    if status_callback:
        status_callback(json.dumps({"status": "Processing study material..."}) + "\n")

    chunks = chunk_text(text)
    ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "file_id": file_id} for _ in chunks]
    
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    # Determine type cleanly
    if filename.startswith("YouTube: "):
        source_type = "youtube"
    elif filename.startswith("Web: "):
        source_type = "web"
    elif '.' in filename:
        source_type = filename.split('.')[-1].lower()
    else:
        source_type = "unknown"

    doc_meta = {
        "id": file_id,
        "filename": filename,
        "type": source_type,
        "file_path": file_path,
        "raw_text": text[:5000],  # First 5000 chars preview
        "chunk_count": len(chunks)
    }

    # Persist in MongoDB Sync Database
    try:
        from database.connection import get_sync_db
        from database.config import settings
        sync_db = get_sync_db()
        sync_db[settings.SOURCES_COLLECTION].replace_one(
            {"id": file_id},
            {
                "id": file_id,
                "filename": filename,
                "type": source_type,
                "file_path": file_path,
                "raw_text": text,
                "chunk_count": len(chunks),
                "created_at": datetime.utcnow()
            },
            upsert=True
        )
    except Exception as e:
        print(f"MongoDB Source Sync Warning: {e}")

    return doc_meta

def get_all_sources():
    # Try fetching from MongoDB first for full persistent data
    try:
        from database.connection import get_sync_db
        from database.config import settings
        sync_db = get_sync_db()
        cursor = sync_db[settings.SOURCES_COLLECTION].find({}, {"_id": 0, "raw_text": 0})
        sources = list(cursor)
        if sources:
            return sources
    except Exception as e:
        print(f"MongoDB Source Fetch Warning: {e}")

    # Fallback to ChromaDB
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])
    unique_sources = {}
    for meta in metadatas:
        if meta["file_id"] not in unique_sources:
            unique_sources[meta["file_id"]] = {
                "id": meta["file_id"],
                "filename": meta["source"],
                "type": meta["source"].split('.')[-1] if '.' in meta["source"] and not meta["source"].startswith("http") else "url"
            }
    return list(unique_sources.values())

def delete_source(file_id: str):
    """Delete a source and all its chunks from ChromaDB and MongoDB."""
    collection.delete(where={"file_id": file_id})
    try:
        from database.connection import get_sync_db
        from database.config import settings
        sync_db = get_sync_db()
        sync_db[settings.SOURCES_COLLECTION].delete_one({"id": file_id})
    except Exception as e:
        print(f"MongoDB Source Delete Warning: {e}")

async def chat_with_context(query: str, history: list = None, selected_source_ids: list = None, response_language: str = "English"):
    if history is None: history = []
    
    where_clause = None
    if selected_source_ids is not None and len(selected_source_ids) > 0:
        if len(selected_source_ids) == 1:
            where_clause = {"file_id": selected_source_ids[0]}
        else:
            where_clause = {"file_id": {"$in": selected_source_ids}}
            
    # If a filter is provided, pass it to where
    if where_clause:
        results = collection.query(query_texts=[query], n_results=4, where=where_clause)
    else:
        results = collection.query(query_texts=[query], n_results=4)
        
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    context = ""
    sources_used = []
    for doc, meta in zip(documents, metadatas):
        context += f"--- Source: {meta['source']} ---\n{doc}\n\n"
        if meta['source'] not in sources_used:
            sources_used.append(meta['source'])
            
    system_prompt = (
        f"You are StudySnap AI, a helpful and knowledgeable assistant. "
        f"Use the provided context to answer the user's question. "
        f"If the answer is not in the context, say that you don't know based on the provided documents.\n"
        f"IMPORTANT: You must respond in {response_language}.\n\n"
        f"Context:\n{context}"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
    final_query = query
    if response_language and response_language.lower() != "english":
        lang_def = get_language_definition(response_language)
        final_query += f"\n\n[CRITICAL DIRECTIVE: You MUST write your ENTIRE response, including all headers, bullet points, and explanations, in {response_language}{lang_def}. Do NOT output standard English!]"

    messages.append({"role": "user", "content": final_query})
    
    response = await lm_studio_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    
    answer = response.choices[0].message.content
    return answer, sources_used

async def generate_podcast_script(selected_source_ids: list = None, response_language: str = "English"):
    t_pipeline = time.perf_counter()
    
    where_clause = None
    if selected_source_ids and len(selected_source_ids) > 0:
        if len(selected_source_ids) == 1:
            where_clause = {"file_id": selected_source_ids[0]}
        else:
            where_clause = {"file_id": {"$in": selected_source_ids}}
            
    # Get chunks — limit to 8 to reduce context size for faster LLM response
    t0 = time.perf_counter()
    if where_clause:
        results = collection.get(where=where_clause, limit=8)
    else:
        results = collection.get(limit=8)
        
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    print(f"[Audio] ChromaDB fetch ({len(documents)} chunks): {time.perf_counter() - t0:.2f}s")
    
    context = ""
    for doc, meta in zip(documents, metadatas):
        context += f"--- Source: {meta['source']} ---\n{doc}\n\n"
        
    lang_def = get_language_definition(response_language)

    system_prompt = (
        "You are an expert podcast producer. Create a short, engaging podcast script between two hosts, "
        "Host A and Host B, discussing the provided documents. "
        f"IMPORTANT: The hosts MUST speak and banter in {response_language}{lang_def}. Write the actual dialogue in this language.\n"
        "Host A should introduce the topic, and they should have a natural back-and-forth banter. "
        "CRITICAL CONSTRAINT: Keep the script to exactly 6-8 dialogue turns total. Each turn should be 1-2 concise sentences (max 30 words per turn). "
        "Be punchy and informative, not verbose. "
        "Return the output STRICTLY as a JSON array of objects, with each object having 'speaker' ('Host A' or 'Host B') "
        "and 'text' (the spoken dialogue). Do not include any other text outside the JSON array.\n\n"
        "Context:\n" + context
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    t0 = time.perf_counter()
    response = await lm_studio_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    print(f"[Audio] LM Studio script generation: {time.perf_counter() - t0:.2f}s")
    
    raw_response = response.choices[0].message.content
    
    try:
        # Extract JSON if wrapped in markdown code blocks
        json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        if json_match:
            script_json = json.loads(json_match.group(0))
        else:
            script_json = json.loads(raw_response)
        print(f"[Audio] Total script generation: {time.perf_counter() - t_pipeline:.2f}s")
        return script_json
    except Exception as e:
        print(f"Error parsing script JSON: {e}")
        print(f"Raw Response: {raw_response}")
        return [{"speaker": "Host A", "text": "Welcome to our Audio Overview! We had a small issue generating the script."}, 
                {"speaker": "Host B", "text": "Yes, unfortunately we couldn't process the documents correctly."}]

def _run_gtts(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

async def _synthesize_segment(index: int, text: str, voice: str):
    """Synthesize a single TTS segment. Returns (index, filepath)."""
    t_start = time.perf_counter()
    
    # Check cache
    cache_dir = "uploads/cache"
    os.makedirs(cache_dir, exist_ok=True)
    hash_input = f"{text}_{voice}"
    segment_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    cached_file = os.path.join(cache_dir, f"tts_{segment_hash}.mp3")
    
    if os.path.exists(cached_file):
        print(f"[Audio] TTS Segment {index} (CACHED): {time.perf_counter() - t_start:.2f}s")
        # We must return a copy to the main logic so it can be merged and cleaned up properly
        output_file = f"uploads/temp_{uuid.uuid4()}_{index}.mp3"
        import shutil
        shutil.copyfile(cached_file, output_file)
        return index, output_file

    output_file = f"uploads/temp_{uuid.uuid4()}_{index}.mp3"
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        print(f"[Audio] TTS Segment {index} (edge-tts): {time.perf_counter() - t_start:.2f}s")
    except Exception as e:
        print(f"edge-tts failed for segment {index}: {e}. Falling back to gTTS.")
        t0 = time.perf_counter()
        await asyncio.to_thread(_run_gtts, text, output_file)
        print(f"[Audio] TTS Segment {index} (gTTS fallback): {time.perf_counter() - t0:.2f}s")
        
    # Save to cache for future
    if os.path.exists(output_file):
        import shutil
        shutil.copyfile(output_file, cached_file)
        
    return index, output_file


async def generate_tts_audio(script_json):
    t_tts_start = time.perf_counter()
    # Define voices
    voices = {
        "Host A": "en-US-ChristopherNeural",
        "Host B": "en-US-AriaNeural"
    }

    # Launch ALL TTS tasks in parallel
    tasks = []
    for i, line in enumerate(script_json):
        speaker = line.get("speaker", "Host A")
        text = line.get("text", "")
        voice = voices.get(speaker, voices["Host A"])
        tasks.append(_synthesize_segment(i, text, voice))

    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"[Audio] TTS parallel synthesis ({len(tasks)} segments): {time.perf_counter() - t0:.2f}s")

    # Sort by original index to preserve dialogue order
    ordered_files = [None] * len(script_json)
    for result in results:
        if isinstance(result, Exception):
            print(f"TTS segment failed: {result}")
            continue
        idx, filepath = result
        ordered_files[idx] = filepath

    # Merge audio files using pydub
    t0 = time.perf_counter()
    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=300)
    for file in ordered_files:
        if file is None:
            continue
        try:
            segment = AudioSegment.from_mp3(file)
            combined += segment + pause
        except Exception as e:
            print(f"Error merging {file}: {e}")

    final_output_name = f"podcast_{uuid.uuid4()}.mp3"
    final_output = f"uploads/{final_output_name}"
    combined.export(final_output, format="mp3")
    print(f"[Audio] Audio merge + export: {time.perf_counter() - t0:.2f}s")

    # Cleanup temp files
    for file in ordered_files:
        if file and os.path.exists(file):
            os.remove(file)

    print(f"[Audio] Total TTS pipeline: {time.perf_counter() - t_tts_start:.2f}s")
    return final_output_name

async def generate_infographic_data(selected_source_ids: list, style: str, detail_level: str, custom_prompt: str, response_language: str = "English"):
    where_clause = None
    if selected_source_ids and len(selected_source_ids) > 0:
        if len(selected_source_ids) == 1:
            where_clause = {"file_id": selected_source_ids[0]}
        else:
            where_clause = {"file_id": {"$in": selected_source_ids}}
            
    if where_clause:
        results = collection.get(where=where_clause, limit=20)
    else:
        results = collection.get(limit=20)
        
    documents = results.get("documents", [])
    context = "\n\n".join(documents)
    
    lang_def = get_language_definition(response_language)

    system_prompt = f"""You are StudySnap AI's Infographic Engine.
Analyze the uploaded sources and generate a structured JSON payload for a professional infographic. Use only the provided content.

Your JSON must STRICTLY follow this schema:
{{
  "title": "String",
  "subtitle": "String",
  "theme": "{style}",
  "sections": [
    {{
      "id": "String (unique)",
      "title": "String",
      "content": "String (concise paragraph)",
      "type": "text | process | comparison | timeline | illustration",
      "visualAssetPrompt": "String (ONLY if type is 'illustration'. A highly detailed image generation prompt for Stable Diffusion 1.5. Describe an illustration, icon, or diagram representing this section. Do NOT ask for text in the image.)"
    }}
  ]
}}

The response MUST be valid JSON. 
The text content in the JSON MUST be in {response_language}{lang_def}.
User's custom instructions: {custom_prompt}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the source content:\n\n{context}\n\nGenerate the JSON now."}
    ]
    
    response = await lm_studio_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=2500,
    )
    
    raw_content = response.choices[0].message.content.strip()
    try:
        import re
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_content, re.DOTALL)
        if json_match:
            raw_content = json_match.group(1)
        return json.loads(raw_content)
    except Exception as e:
        print(f"Failed to parse Infographic JSON: {e}")
        print(f"Raw content: {raw_content}")
        # Fallback response
        return {
            "title": "Error Generating Infographic",
            "subtitle": "Could not parse AI response",
            "theme": style,
            "sections": [
                {
                    "id": "1",
                    "title": "Error",
                    "content": "There was an issue processing the content.",
                    "type": "text"
                }
            ]
        }

async def generate_mind_map_data(selected_source_ids: list, custom_prompt: str, response_language: str = "English"):
    where_clause = None
    if selected_source_ids and len(selected_source_ids) > 0:
        if len(selected_source_ids) == 1:
            where_clause = {"file_id": selected_source_ids[0]}
        else:
            where_clause = {"file_id": {"$in": selected_source_ids}}
            
    if where_clause:
        results = collection.get(where=where_clause, limit=20)
    else:
        results = collection.get(limit=20)
        
    documents = results.get("documents", [])
    context = "\n\n".join(documents)
    
    lang_def = get_language_definition(response_language)

    system_prompt = (
        "You are an expert knowledge architect. Your task is to analyze the provided documents and create a structured Mind Map. "
        f"User's custom instructions: {custom_prompt}\n"
        f"IMPORTANT: The node labels and context descriptions MUST be in {response_language}{lang_def}.\n\n"
        "Return the output STRICTLY as a JSON object with two keys: 'nodes' and 'edges'. "
        "Each node should have: "
        "- id: string (unique identifier) "
        "- data: object containing 'label' (short title) and 'context' (1-2 sentences explaining this node based on sources) "
        "- position: object containing 'x' and 'y' coordinates (arrange them hierarchically, e.g., root at 250,0, children spread out below). "
        "Each edge should have: "
        "- id: string (e.g., 'e1-2') "
        "- source: string (id of parent node) "
        "- target: string (id of child node). "
        "Do not include any markdown formatting, only pure JSON."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the source content:\n\n{context}"}
    ]
    
    response = await lm_studio_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=2000,
    )
    
    try:
        import json
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        return json.loads(content)
    except Exception as e:
        print("Failed to parse JSON mind map:", e)
        # Return a fallback mindmap
        return {
            "nodes": [
                {"id": "1", "data": {"label": "Error parsing map", "context": "Try again."}, "position": {"x": 250, "y": 250}}
            ],
            "edges": []
        }

async def generate_quiz_data(selected_source_ids: list, response_language: str = "English"):
    where_clause = None
    if selected_source_ids and len(selected_source_ids) > 0:
        if len(selected_source_ids) == 1:
            where_clause = {"file_id": selected_source_ids[0]}
        else:
            where_clause = {"file_id": {"$in": selected_source_ids}}
            
    if where_clause:
        results = collection.get(where=where_clause, limit=20)
    else:
        results = collection.get(limit=20)
        
    documents = results.get("documents", [])
    context = "\n\n".join(documents)
    
    lang_def = get_language_definition(response_language)

    system_prompt = f"""You are an expert educator.
Analyze the uploaded sources and generate a multiple-choice quiz to test comprehension. Use ONLY the provided content.

Your JSON must STRICTLY follow this schema:
{{
  "title": "String",
  "questions": [
    {{
      "id": "String (unique)",
      "question": "String",
      "options": ["String", "String", "String", "String"],
      "correctIndex": Number (0-3),
      "explanation": "String"
    }}
  ]
}}

Generate exactly 5 questions.
The response MUST be valid JSON. 
The text content in the JSON MUST be in {response_language}{lang_def}.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the source content:\n\n{context}\n\nGenerate the JSON now."}
    ]
    
    response = await lm_studio_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=2500,
    )
    
    raw_content = response.choices[0].message.content.strip()
    try:
        import re
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_content, re.DOTALL)
        if json_match:
            raw_content = json_match.group(1)
        return json.loads(raw_content)
    except Exception as e:
        print(f"Failed to parse Quiz JSON: {e}")
        return {
            "title": "Error Generating Quiz",
            "questions": [
                {
                    "id": "1",
                    "question": "There was an error generating the quiz.",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correctIndex": 0,
                    "explanation": "Please try again later."
                }
            ]
        }

async def generate_flashcards_data(selected_source_ids: list, response_language: str = "English"):
    where_clause = None
    if selected_source_ids and len(selected_source_ids) > 0:
        if len(selected_source_ids) == 1:
            where_clause = {"file_id": selected_source_ids[0]}
        else:
            where_clause = {"file_id": {"$in": selected_source_ids}}
            
    if where_clause:
        results = collection.get(where=where_clause, limit=20)
    else:
        results = collection.get(limit=20)
        
    documents = results.get("documents", [])
    context = "\n\n".join(documents)
    
    lang_def = get_language_definition(response_language)

    system_prompt = f"""You are an expert educator.
Analyze the uploaded sources and generate a set of flashcards for key terms and concepts. Use ONLY the provided content.

Your JSON must STRICTLY follow this schema:
{{
  "title": "String",
  "flashcards": [
    {{
      "id": "String (unique)",
      "front": "String (The term or question)",
      "back": "String (The definition or answer)"
    }}
  ]
}}

Generate exactly 10 flashcards.
The response MUST be valid JSON. 
The text content in the JSON MUST be in {response_language}{lang_def}.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the source content:\n\n{context}\n\nGenerate the JSON now."}
    ]
    
    response = await lm_studio_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=2500,
    )
    
    raw_content = response.choices[0].message.content.strip()
    try:
        import re
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_content, re.DOTALL)
        if json_match:
            raw_content = json_match.group(1)
        return json.loads(raw_content)
    except Exception as e:
        print(f"Failed to parse Flashcards JSON: {e}")
        return {
            "title": "Error Generating Flashcards",
            "flashcards": [
                {
                    "id": "1",
                    "front": "Error",
                    "back": "Failed to parse AI response. Try again."
                }
            ]
        }

