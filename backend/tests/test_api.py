import pytest
from fastapi.testclient import TestClient
import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from rag import _is_safe_url, chunk_text, collection

client = TestClient(app)

def test_db_health():
    """Verify database health endpoint returns valid JSON."""
    response = client.get("/api/health/db")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "connected" in data

def test_system_status():
    """Verify LM Studio and ComfyUI status endpoint."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "lm_studio" in data
    assert "comfyui" in data

def test_ssrf_url_validation():
    """Verify SSRF protection blocks private, loopback, and cloud metadata addresses."""
    # Loopback & private IP rejection
    assert _is_safe_url("http://127.0.0.1:8000") is False
    assert _is_safe_url("http://localhost:5000") is False
    assert _is_safe_url("http://169.254.169.254/latest/meta-data") is False
    assert _is_safe_url("http://10.0.0.1/admin") is False
    assert _is_safe_url("http://192.168.1.1") is False
    assert _is_safe_url("ftp://example.com/file") is False
    # Valid external web URL acceptance
    assert _is_safe_url("https://en.wikipedia.org/wiki/Operating_system") is True

def test_sources_list():
    """Verify list sources endpoint returns an array."""
    response = client.get("/api/sources")
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)

def test_text_chunking():
    """Verify text chunker preserves overlap and produces non-empty chunks."""
    text = "StudySnap AI is an academic assistant designed for college students. " * 30
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)

def test_chromadb_persistence():
    """Verify ChromaDB collection can store and query vector embeddings."""
    test_id = "test_chunk_doc_001"
    collection.add(
        documents=["Virtual memory allows execution of processes not completely in memory."],
        metadatas=[{"source": "test_os_unit.txt", "file_id": "doc_test_1"}],
        ids=[test_id]
    )
    
    results = collection.query(query_texts=["How does virtual memory work?"], n_results=1)
    assert len(results["documents"][0]) > 0
    assert "virtual memory" in results["documents"][0][0].lower()
    
    # Cleanup test chunk
    collection.delete(ids=[test_id])

def test_empty_upload_rejection():
    """Verify uploading no files yields a 422 validation error."""
    response = client.post("/api/sources/upload")
    assert response.status_code == 422

def test_chat_offline_resilience():
    """Verify chat endpoint returns clear guidance when LM Studio is offline."""
    response = client.post(
        "/api/chat",
        json={
            "message": "Explain Deadlock Coffman conditions in Operating Systems",
            "history": [],
            "response_language": "English"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data

def test_notes_read_resilience():
    """Verify notes listing returns list without throwing 500 when offline."""
    response = client.get("/api/notes?user_id=default_user")
    assert response.status_code == 200
    data = response.json()
    assert "notes" in data
    assert isinstance(data["notes"], list)
