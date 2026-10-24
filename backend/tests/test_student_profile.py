import pytest
from fastapi.testclient import TestClient
import jwt
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app

client = TestClient(app)

def create_student_jwt(user_id: str, email: str, name: str = "Aarav Patil", college: str = "DKTE", expires_in: int = 3600) -> str:
    """Helper to generate Supabase-compatible test JWT tokens with student metadata."""
    payload = {
        "sub": user_id,
        "email": email,
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + expires_in,
        "user_metadata": {
            "name": name,
            "full_name": name,
            "college": college,
            "college_name": college,
            "degree_program": "B.Tech / B.E.",
            "branch": "Computer Science Engineering (CSE)",
            "academic_year": "Third Year",
            "semester": "Semester 5",
            "graduation_year": "2026"
        }
    }
    return jwt.encode(payload, "studysnap_ai_secret_key_change_in_production", algorithm="HS256")

def test_get_student_profile():
    """Verify that an authenticated student can fetch their academic profile."""
    user_id = "11111111-1111-1111-1111-111111111111"
    token = create_student_jwt(user_id, "student1@iitb.ac.in", name="Aarav Patil")
    headers = {"Authorization": f"Bearer {token}"}

    res = client.get("/api/user/profile", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "profile" in data
    profile = data["profile"]
    assert profile["id"] == user_id

def test_update_student_profile():
    """Verify updating student academic information via PUT /api/user/profile."""
    user_id = "22222222-2222-2222-2222-222222222222"
    token = create_student_jwt(user_id, "student2@iitb.ac.in", name="Priya Desai")
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "full_name": "Priya Rajesh Desai",
        "college_name": "DKTE Society's Textile and Engineering Institute",
        "degree_program": "B.Tech / B.E.",
        "branch": "Artificial Intelligence and Machine Learning (AIML)",
        "academic_year": "Fourth Year",
        "semester": "Semester 7",
        "graduation_year": "2025",
        "roll_number": "21AIML042"
    }

    res = client.put("/api/user/profile", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    updated = data["profile"]
    assert updated["full_name"] == "Priya Rajesh Desai"
    assert updated["branch"] == "Artificial Intelligence and Machine Learning (AIML)"
    assert updated["academic_year"] == "Fourth Year"
    assert updated["semester"] == "Semester 7"
    assert updated["roll_number"] == "21AIML042"

def test_update_theme_preference():
    """Verify updating and persisting student color theme preference (VS Code inspired)."""
    user_id = "33333333-3333-3333-3333-333333333333"
    token = create_student_jwt(user_id, "student3@dkte.ac.in", name="Vikram More")
    headers = {"Authorization": f"Bearer {token}"}

    # Update theme preference to monokai
    payload = {
        "theme_preference": "monokai"
    }
    res = client.put("/api/user/profile", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["profile"].get("theme_preference") == "monokai"

    # Verify subsequent GET returns the persisted theme
    get_res = client.get("/api/user/profile", headers=headers)
    assert get_res.status_code == 200
    profile_data = get_res.json()["profile"]
    assert profile_data.get("theme_preference") == "monokai"

def test_invalid_and_expired_token_rejection():
    """Verify that requests with invalid or expired tokens are rejected with 401."""
    # Malformed token
    res_malformed = client.get("/api/user/profile", headers={"Authorization": "MalformedToken"})
    assert res_malformed.status_code == 401

    # Expired token
    expired_token = create_student_jwt("user-exp", "exp@college.edu", expires_in=-120)
    res_expired = client.get("/api/user/profile", headers={"Authorization": f"Bearer {expired_token}"})
    assert res_expired.status_code == 401

def test_student_profiles_isolation():
    """Verify that two distinct students have isolated profiles and cannot alter each other's data."""
    student_a_id = "33333333-3333-3333-3333-333333333333"
    student_b_id = "44444444-4444-4444-4444-444444444444"

    token_a = create_student_jwt(student_a_id, "student_a@college.edu.in", name="Student Alpha")
    token_b = create_student_jwt(student_b_id, "student_b@college.edu.in", name="Student Beta")

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # Student A updates their own profile
    res_a = client.put("/api/user/profile", json={
        "full_name": "Student Alpha (CSE)",
        "branch": "Computer Science Engineering (CSE)"
    }, headers=headers_a)
    assert res_a.status_code == 200

    # Student B reads their own profile — must NOT be Student A
    res_b = client.get("/api/user/profile", headers=headers_b)
    assert res_b.status_code == 200
    profile_b = res_b.json()["profile"]
    assert profile_b["id"] == student_b_id
    assert profile_b.get("full_name") != "Student Alpha (CSE)"

def test_avatar_upload_valid_png():
    """Verify uploading a valid PNG avatar within size limits succeeds and links to profile."""
    user_id = "55555555-5555-5555-5555-555555555555"
    token = create_student_jwt(user_id, "avatar_png@college.edu.in", name="Neha Sharma")
    headers = {"Authorization": f"Bearer {token}"}

    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    files = {"file": ("student_avatar.png", png_bytes, "image/png")}

    res = client.post("/api/user/avatar/upload", files=files, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["avatar_path"].startswith(f"avatars:{user_id}/avatar_")
    assert data["avatar_path"].endswith(".png")
    assert user_id in data["avatar_url"]

    # Verify profile now reflects avatar_path
    prof_res = client.get("/api/user/profile", headers=headers)
    assert prof_res.status_code == 200
    prof = prof_res.json()["profile"]
    assert prof["avatar_path"] == data["avatar_path"]

def test_avatar_upload_oversized_rejection():
    """Verify rejection when uploaded avatar exceeds the 5 MB threshold."""
    user_id = "66666666-6666-6666-6666-666666666666"
    token = create_student_jwt(user_id, "oversized@college.edu.in")
    headers = {"Authorization": f"Bearer {token}"}

    # Generate 5.2 MB payload
    large_payload = b"\xff\xd8\xff" + b"X" * (5 * 1024 * 1024 + 200 * 1024)
    files = {"file": ("huge_photo.jpg", large_payload, "image/jpeg")}

    res = client.post("/api/user/avatar/upload", files=files, headers=headers)
    assert res.status_code == 400
    assert "exceeds maximum allowed limit of 5 MB" in res.json()["detail"]

def test_avatar_upload_unsupported_format_rejection():
    """Verify rejection of executable, document, or non-image files."""
    user_id = "77777777-7777-7777-7777-777777777777"
    token = create_student_jwt(user_id, "invalid_fmt@college.edu.in")
    headers = {"Authorization": f"Bearer {token}"}

    fake_text_file = b"This is a plain text file, not an image!"
    files = {"file": ("malicious.txt", fake_text_file, "text/plain")}

    res = client.post("/api/user/avatar/upload", files=files, headers=headers)
    assert res.status_code == 400
    assert "Unsupported file format" in res.json()["detail"]

def test_avatar_upload_empty_file_rejection():
    """Verify empty 0-byte file upload rejection."""
    user_id = "88888888-8888-8888-8888-888888888888"
    token = create_student_jwt(user_id, "empty_file@college.edu.in")
    headers = {"Authorization": f"Bearer {token}"}

    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = client.post("/api/user/avatar/upload", files=files, headers=headers)
    assert res.status_code == 400
    assert "Uploaded file is empty" in res.json()["detail"]

def test_avatar_replacement_and_removal():
    """Verify photo replacement with new image and removal restoring initial fallback."""
    user_id = "99999999-9999-9999-9999-999999999999"
    token = create_student_jwt(user_id, "replace_remove@college.edu.in", name="Kunal Verma")
    headers = {"Authorization": f"Bearer {token}"}

    # Step 1: Upload initial JPEG avatar
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\xff\xd9"
    res1 = client.post("/api/user/avatar/upload", files={"file": ("photo1.jpg", jpeg_bytes, "image/jpeg")}, headers=headers)
    assert res1.status_code == 200
    first_path = res1.json()["avatar_path"]
    assert first_path.endswith(".jpg")

    # Step 2: Replace with WebP avatar
    webp_bytes = b"RIFF\x1a\x00\x00\x00WEBPVP8 \x0e\x00\x00\x00\x30\x01\x00\x9d\x01\x2a\x01\x00\x01\x00\x00"
    res2 = client.post("/api/user/avatar/upload", files={"file": ("photo2.webp", webp_bytes, "image/webp")}, headers=headers)
    assert res2.status_code == 200
    second_path = res2.json()["avatar_path"]
    assert second_path.endswith(".webp")
    assert second_path != first_path

    # Step 3: Remove avatar via DELETE
    res_del = client.delete("/api/user/avatar", headers=headers)
    assert res_del.status_code == 200
    del_data = res_del.json()
    assert del_data["success"] is True
    assert del_data["profile"]["avatar_path"] is None
    assert del_data["profile"]["avatar_url"] is None

    # Step 4: Verify profile fetch confirms avatar is cleared
    res_prof = client.get("/api/user/profile", headers=headers)
    prof = res_prof.json()["profile"]
    assert prof["avatar_path"] is None
    assert prof["avatar_url"] is None

