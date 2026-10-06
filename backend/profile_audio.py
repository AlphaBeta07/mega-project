import time
import asyncio
import os
import uuid
import json

import edge_tts
from gtts import gTTS
from pydub import AudioSegment

start_time = time.time()

print(f"[Audio] Script started")
t0 = time.time()
import whisper
print(f"[Audio] Import whisper: {time.time() - t0:.2f}s")

t0 = time.time()
model = whisper.load_model("tiny")
print(f"[Audio] Load whisper tiny model: {time.time() - t0:.2f}s")

t0 = time.time()
result = model.transcribe("test_audio.ogg", fp16=False)
transcript = result.get("text", "")
print(f"[Audio] Transcribe test_audio.ogg: {time.time() - t0:.2f}s")
print(f"Transcript length: {len(transcript)}")

async def test_edge_tts():
    try:
        t0 = time.time()
        communicate = edge_tts.Communicate("Hello world", "en-US-AriaNeural")
        await communicate.save("test_out.mp3")
        print(f"[Audio] Edge-tts save: {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"[Audio] Edge-tts failed after {time.time() - t0:.2f}s: {e}")
        t0 = time.time()
        tts = gTTS(text="Hello world", lang='en')
        tts.save("test_out.mp3")
        print(f"[Audio] gTTS fallback: {time.time() - t0:.2f}s")

asyncio.run(test_edge_tts())

print(f"[Audio] Total time: {time.time() - start_time:.2f}s")
