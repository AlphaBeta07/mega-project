import requests
import time
import json

base_url = "http://localhost:8000"

def run_test():
    print("--- Starting Pipeline Benchmark ---")
    
    # 1. Upload
    print("\n1. Uploading audio file...")
    t0 = time.time()
    with open("test_audio_speech.ogg", "rb") as f:
        files = {"files": ("test_audio_speech.ogg", f, "audio/ogg")}
        response = requests.post(f"{base_url}/api/sources/upload", files=files)
    
    upload_time = time.time() - t0
    
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return
        
    data = response.json()
    source_id = data["sources"][0]["id"]
    print(f"Upload completed in {upload_time:.2f}s. Source ID: {source_id}")
    
    # 2. Generate Audio Overview
    print("\n2. Generating Audio Overview...")
    t0 = time.time()
    payload = {
        "selected_source_ids": [source_id],
        "response_language": "English"
    }
    response = requests.post(f"{base_url}/api/audio-overview", json=payload)
    gen_time = time.time() - t0
    
    if response.status_code != 200:
        print(f"Generation failed: {response.text}")
        return
        
    data = response.json()
    print(f"Generation completed in {gen_time:.2f}s.")
    print(f"Audio URL: {data['audio_url']}")
    print(f"\nTotal End-to-End Pipeline Time: {upload_time + gen_time:.2f}s")
    
if __name__ == "__main__":
    run_test()
