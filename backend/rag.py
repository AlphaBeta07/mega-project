import os
import json
import csv
import io
import requests
from bs4 import BeautifulSoup
from docx import Document
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
MODEL_NAME = "my_custom_200m_model_gguf/AnishLandage/SnapStudyAI"

def extract_youtube_transcript(url: str) -> str:
    try:
        video_id = ""
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        elif "embed/" in url:
            video_id = url.split("embed/")[1].split("?")[0]
            
        if not video_id: 
            raise ValueError("Could not find video ID in URL.")
        
        ytt_api = YouTubeTranscriptApi()
        
        try:
            # Try getting standard English or auto-generated
            transcript_list = ytt_api.list(video_id)
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'hi'])
            data = transcript.fetch()
        except:
            try:
                # Fallback to any auto-generated transcript if manual English isn't found
                transcript_list = ytt_api.list(video_id)
                transcript = list(transcript_list)[0] # Just get the first available one
                data = transcript.fetch()
            except Exception as e:
                # If all else fails, use the basic method
                data = ytt_api.fetch(video_id)
                
        return " ".join([t.text if hasattr(t, 'text') else t.get('text', '') for t in data])
    except Exception as e:
        print(f"YouTube Extraction Error: {e}")
        raise ValueError(f"Failed to process YouTube transcript: {str(e)}. The video might not have captions enabled.")

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

def extract_audio(file_path: str) -> str:
    """Extract text from audio using local Whisper instance."""
    try:
        # Load the base model ('tiny', 'base', 'small', 'medium', 'large')
        # 'base' is a good tradeoff between speed and accuracy for local transcription
        model = whisper.load_model("base")
        
        # Transcribe the audio file
        # Whisper automatically handles chunking for long audio files
        result = model.transcribe(file_path)
        
        return result.get("text", "")
    except Exception as e:
        print(f"Whisper Audio Extraction Error: {e}")
        return "Audio transcription failed. Please ensure ffmpeg is installed."

def extract_text(file_path: str, filename: str) -> str:
    """Extract text from all supported file types."""
    text = ""
    ext = filename.split('.')[-1].lower() if '.' in filename else ""
    
    try:
        if ext == 'pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                if page_text := page.extract_text(): text += page_text + "\n"
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
        elif ext in ['wav', 'mp3', 'm4a', 'flac']:
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

def ingest_document(file_path: str, filename: str, file_id: str):
    text = ""
    # Check if it's a URL or YouTube link based on filename format or type passed from main
    if filename.startswith("http"):
        if "youtube.com" in filename or "youtu.be" in filename:
            text = extract_youtube_transcript(filename)
            filename = f"YouTube: {filename}"
        else:
            text = extract_webpage(filename)
            filename = f"Web: {filename}"
    else:
        text = extract_text(file_path, filename)
        
    if not text.strip():
        raise ValueError("Could not extract any text from the document.")

    chunks = chunk_text(text)
    ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "file_id": file_id} for _ in chunks]
    
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return {"id": file_id, "filename": filename, "type": filename.split('.')[-1] if '.' in filename and not filename.startswith("http") else "url"}

def get_all_sources():
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
    """Delete a source and all its chunks from ChromaDB."""
    collection.delete(where={"file_id": file_id})

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
        lang_def = ""
        if response_language.lower() == "hinglish":
            lang_def = " (a mix of Hindi and English written in the English alphabet)"
        elif response_language.lower() == "manglish":
            lang_def = " (a mix of Malayalam and English written in the English alphabet)"
            
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
    where_clause = None
    if selected_source_ids and len(selected_source_ids) > 0:
        if len(selected_source_ids) == 1:
            where_clause = {"file_id": selected_source_ids[0]}
        else:
            where_clause = {"file_id": {"$in": selected_source_ids}}
            
    # Get all chunks for these files, limit to 20 to avoid context limits
    if where_clause:
        results = collection.get(where=where_clause, limit=20)
    else:
        results = collection.get(limit=20)
        
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    
    context = ""
    for doc, meta in zip(documents, metadatas):
        context += f"--- Source: {meta['source']} ---\n{doc}\n\n"
        
    lang_def = ""
    if response_language and response_language.lower() == "hinglish":
        lang_def = " (a mix of Hindi and English written in the English alphabet)"
    elif response_language and response_language.lower() == "manglish":
        lang_def = " (a mix of Malayalam and English written in the English alphabet)"

    system_prompt = (
        "You are an expert podcast producer. Create a multi-turn, engaging podcast script between two hosts, "
        "Host A and Host B, discussing the provided documents. "
        f"IMPORTANT: The hosts MUST speak and banter in {response_language}{lang_def}. Write the actual dialogue in this language.\n"
        "Host A should introduce the topic, and they should have a natural back-and-forth banter. "
        "Return the output STRICTLY as a JSON array of objects, with each object having 'speaker' ('Host A' or 'Host B') "
        "and 'text' (the spoken dialogue). Do not include any other text outside the JSON array.\n\n"
        "Context:\n" + context
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    response = await lm_studio_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    
    raw_response = response.choices[0].message.content
    
    try:
        # Extract JSON if wrapped in markdown code blocks
        json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        if json_match:
            script_json = json.loads(json_match.group(0))
        else:
            script_json = json.loads(raw_response)
        return script_json
    except Exception as e:
        print(f"Error parsing script JSON: {e}")
        print(f"Raw Response: {raw_response}")
        return [{"speaker": "Host A", "text": "Welcome to our Audio Overview! We had a small issue generating the script."}, 
                {"speaker": "Host B", "text": "Yes, unfortunately we couldn't process the documents correctly."}]

async def generate_tts_audio(script_json):
    audio_files = []
    
    # Define voices
    voices = {
        "Host A": "en-US-ChristopherNeural",
        "Host B": "en-US-AriaNeural"
    }
    
    for i, line in enumerate(script_json):
        speaker = line.get("speaker", "Host A")
        text = line.get("text", "")
        voice = voices.get(speaker, voices["Host A"])
        
        output_file = f"temp_{uuid.uuid4()}_{i}.mp3"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        audio_files.append(output_file)
        
    # Merge audio files using pydub
    combined = AudioSegment.empty()
    for file in audio_files:
        try:
            segment = AudioSegment.from_mp3(file)
            combined += segment
            # Add a small pause between speakers (300ms)
            combined += AudioSegment.silent(duration=300)
        except Exception as e:
            print(f"Error merging {file}: {e}")
            
    final_output_name = f"podcast_{uuid.uuid4()}.mp3"
    final_output = f"uploads/{final_output_name}"
    combined.export(final_output, format="mp3")
    
    # Cleanup temp files
    for file in audio_files:
        if os.path.exists(file):
            os.remove(file)
            
    return final_output_name

async def generate_infographic_image(selected_source_ids: list, style: str, detail_level: str, custom_prompt: str, response_language: str = "English"):
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
    
    lang_def = ""
    if response_language and response_language.lower() == "hinglish":
        lang_def = " (a mix of Hindi and English written in the English alphabet)"
    elif response_language and response_language.lower() == "manglish":
        lang_def = " (a mix of Malayalam and English written in the English alphabet)"

    # Generate image prompt using LM Studio
    system_prompt = (
        "You are an expert prompt engineer for an AI image generator (like Midjourney/DALL-E). "
        "Your task is to analyze the provided documents and create a highly detailed, descriptive visual prompt for an infographic. "
        f"The user wants a '{style}' style infographic with a '{detail_level}' level of detail. "
        f"User's custom instructions: {custom_prompt}\n"
        f"Any explicit text described in the image prompt should be in {response_language}{lang_def}.\n\n"
        "Extract the core concepts and data from the text, and describe exactly what the infographic should look like visually. "
        "Focus on layout, colors, icons, and visual metaphors rather than specific text (since AI struggles with spelling). "
        "Return ONLY the image prompt text, no pleasantries or explanation."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the source content:\n\n{context}"}
    ]
    
    response = await lm_studio_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    
    image_prompt = response.choices[0].message.content.strip()
    
    # We will use Pollinations.ai for free, keyless image generation
    encoded_prompt = urllib.parse.quote(image_prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    
    final_output_name = f"infographic_{uuid.uuid4()}.png"
    final_output = f"uploads/{final_output_name}"
    
    # Download the image
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status == 200:
                with open(final_output, 'wb') as f:
                    f.write(await resp.read())
            else:
                raise ValueError(f"Failed to fetch image from Pollinations API. Status: {resp.status}")
                
    return final_output_name

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
    
    lang_def = ""
    if response_language and response_language.lower() == "hinglish":
        lang_def = " (a mix of Hindi and English written in the English alphabet)"
    elif response_language and response_language.lower() == "manglish":
        lang_def = " (a mix of Malayalam and English written in the English alphabet)"

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
