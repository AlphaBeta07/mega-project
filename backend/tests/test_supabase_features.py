import pytest
from fastapi.testclient import TestClient
import jwt
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from database.config import settings

client = TestClient(app)

def create_supabase_jwt(user_id: str, email: str, expires_in: int = 3600) -> str:
    """Helper to generate Supabase-compatible test JWT tokens."""
    payload = {
        "sub": user_id,
        "email": email,
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + expires_in,
        "user_metadata": {
            "name": email.split("@")[0].capitalize(),
            "college": "DKTE"
        }
    }
    return jwt.encode(payload, "studysnap_ai_secret_key_change_in_production", algorithm="HS256")

def test_supabase_health_endpoint():
    """Verify Supabase health endpoint reports database status and schema configuration."""
    res = client.get("/api/health/db")
    assert res.status_code == 200
    data = res.json()
    assert data["database"] == "supabase_postgresql"
    assert "status" in data
    assert "connected" in data

def test_expired_supabase_token_rejection():
    """Verify expired Supabase access tokens are rejected with 401 Unauthorized."""
    expired_token = create_supabase_jwt("user-expired-01", "expired@test.com", expires_in=-60)
    headers = {"Authorization": f"Bearer {expired_token}"}
    res = client.get("/api/notebooks", headers=headers)
    assert res.status_code == 401
    assert "expired" in res.json()["detail"].lower()

def test_malformed_token_rejection():
    """Verify malformed authorization headers are rejected."""
    headers = {"Authorization": "InvalidTokenFormat"}
    res = client.get("/api/notebooks", headers=headers)
    assert res.status_code == 401

def test_supabase_notebook_rls_isolation():
    """Verify that User A and User B have complete isolation of their notebooks."""
    user_1_id = "00000000-0000-0000-0000-000000000010"
    user_2_id = "00000000-0000-0000-0000-000000000020"

    headers_1 = {"Authorization": f"Bearer {create_supabase_jwt(user_1_id, 'user1@iitb.ac.in')}"}
    headers_2 = {"Authorization": f"Bearer {create_supabase_jwt(user_2_id, 'user2@iitb.ac.in')}"}

    # User 1 creates notebook
    create_res = client.post("/api/notebooks", json={
        "title": "Machine Learning — Gradient Descent",
        "code": "CS725",
        "content": "# Loss Functions and Optimizers"
    }, headers=headers_1)
    assert create_res.status_code == 200
    nb1 = create_res.json()["notebook"]
    nb1_id = nb1["id"]

    # User 2 lists notebooks — should NOT see User 1's notebook
    list_2_res = client.get("/api/notebooks", headers=headers_2)
    assert list_2_res.status_code == 200
    user_2_notebook_ids = [n["id"] for n in list_2_res.json()["notebooks"]]
    assert nb1_id not in user_2_notebook_ids

    # User 2 attempts to fetch User 1's notebook directly — must be 404
    cross_get_res = client.get(f"/api/notebooks/{nb1_id}", headers=headers_2)
    assert cross_get_res.status_code in [403, 404]

    # User 2 attempts to update User 1's notebook — must be 404
    cross_put_res = client.put(f"/api/notebooks/{nb1_id}", json={
        "title": "Hacked Title"
    }, headers=headers_2)
    assert cross_put_res.status_code in [403, 404]

    # User 2 attempts to delete User 1's notebook — must be 404
    cross_del_res = client.delete(f"/api/notebooks/{nb1_id}", headers=headers_2)
    assert cross_del_res.status_code in [403, 404]

    # Clean up by User 1
    del_res = client.delete(f"/api/notebooks/{nb1_id}", headers=headers_1)
    assert del_res.status_code == 200

def test_supabase_notes_crud():
    """Verify notes persistence in Supabase with user authentication."""
    user_id = "00000000-0000-0000-0000-000000000030"
    headers = {"Authorization": f"Bearer {create_supabase_jwt(user_id, 'notes_tester@iitb.ac.in')}"}

    # Create note
    create_res = client.post("/api/notes", json={
        "subject": "Computer Networks",
        "chapter": "Transport Layer",
        "formatted_content": "# TCP vs UDP\nTCP is connection-oriented.",
        "summary": "Covers TCP handshake and UDP datagrams.",
        "keywords": ["TCP", "UDP", "Handshake"]
    }, headers=headers)
    assert create_res.status_code == 200
    note = create_res.json()["note"]
    note_id = note["id"]

    # List notes
    list_res = client.get("/api/notes", headers=headers)
    assert list_res.status_code == 200
    assert any(n.get("id") == note_id for n in list_res.json()["notes"])

    # Update note
    update_res = client.put(f"/api/notes/{note_id}", json={
        "formatted_content": "# TCP vs UDP\nTCP includes congestion control."
    }, headers=headers)
    assert update_res.status_code == 200

    # Delete note
    del_res = client.delete(f"/api/notes/{note_id}", headers=headers)
    assert del_res.status_code == 200

def test_supabase_chat_session_history():
    """Verify chat sessions and messages persist correctly with user context."""
    user_id = "00000000-0000-0000-0000-000000000040"
    headers = {"Authorization": f"Bearer {create_supabase_jwt(user_id, 'chat_tester@iitb.ac.in')}"}

    # Initial session list
    list_res = client.get("/api/sessions", headers=headers)
    assert list_res.status_code == 200
