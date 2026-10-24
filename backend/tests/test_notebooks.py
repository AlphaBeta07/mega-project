import pytest
from fastapi.testclient import TestClient
import jwt
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app

client = TestClient(app)

# Helper to generate test Supabase JWT tokens
def make_test_token(user_id: str, email: str = "student@test.com") -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + 3600
    }
    return jwt.encode(payload, "studysnap_ai_secret_key_change_in_production", algorithm="HS256")

def test_notebook_crud_lifecycle():
    user_a_token = make_test_token("user_test_uuid_1001", "alice@college.edu")
    headers_a = {"Authorization": f"Bearer {user_a_token}"}

    # 1. Create Notebook
    create_payload = {
        "title": "Data Structures & Algorithms - Trees & Graphs",
        "code": "CS201",
        "content": "# Binary Trees\nA tree is an undirected connected acyclic graph."
    }
    res_create = client.post("/api/notebooks", json=create_payload, headers=headers_a)
    assert res_create.status_code == 200
    data_create = res_create.json()
    assert data_create["success"] is True
    notebook = data_create["notebook"]
    assert notebook["title"] == "Data Structures & Algorithms - Trees & Graphs"
    assert notebook["user_id"] == "user_test_uuid_1001"
    notebook_id = notebook["id"]

    # 2. List Notebooks for User A
    res_list = client.get("/api/notebooks", headers=headers_a)
    assert res_list.status_code == 200
    data_list = res_list.json()
    assert data_list["success"] is True
    assert any(nb["id"] == notebook_id for nb in data_list["notebooks"])

    # 3. Get Notebook Details
    res_get = client.get(f"/api/notebooks/{notebook_id}", headers=headers_a)
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["notebook"]["id"] == notebook_id
    assert data_get["notebook"]["content"].startswith("# Binary Trees")

    # 4. Autosave / Update Notebook
    update_payload = {
        "title": "DSA - Advanced Graphs & Bellman-Ford",
        "content": "# Graphs\nShortest path algorithms."
    }
    res_update = client.put(f"/api/notebooks/{notebook_id}", json=update_payload, headers=headers_a)
    assert res_update.status_code == 200
    data_update = res_update.json()
    assert data_update["notebook"]["title"] == "DSA - Advanced Graphs & Bellman-Ford"
    assert data_update["notebook"]["content"] == "# Graphs\nShortest path algorithms."

    # 5. Cross-User Authorization Test (User B should NOT be able to access User A's notebook)
    user_b_token = make_test_token("user_test_uuid_2002", "bob@college.edu")
    headers_b = {"Authorization": f"Bearer {user_b_token}"}

    res_cross_get = client.get(f"/api/notebooks/{notebook_id}", headers=headers_b)
    assert res_cross_get.status_code in [403, 404]

    res_cross_update = client.put(f"/api/notebooks/{notebook_id}", json=update_payload, headers=headers_b)
    assert res_cross_update.status_code in [403, 404]

    # 6. Delete Notebook
    res_delete = client.delete(f"/api/notebooks/{notebook_id}", headers=headers_a)
    assert res_delete.status_code == 200
    assert res_delete.json()["success"] is True

    # 7. Confirm Deleted
    res_get_deleted = client.get(f"/api/notebooks/{notebook_id}", headers=headers_a)
    assert res_get_deleted.status_code == 404
