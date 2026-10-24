import pytest
import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompt_registry import (
    FEATURE_DEFINITIONS,
    detect_feature,
    sanitize_feature_title,
    sanitize_message_content,
    resolve_request_instruction,
)
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_detect_audio_to_notes_feature():
    raw_prompt = (
        "Convert the following audio transcript into structured educational notes using clean Markdown. "
        "You are an expert educational assistant that creates clear, structured study notes. "
        "Format all sections with bold Markdown headers: # [Topic Title], ## Key Points, ## Explanation, ## Examples, and ## Summary in detailed format."
    )
    feat = detect_feature(raw_prompt)
    assert feat is not None
    assert feat["id"] == "audionotes"
    assert feat["visible_title"] == "Audio to Notes"

def test_sanitize_feature_title():
    raw_prompt = (
        "Convert the following audio transcript into structured educational notes using clean Markdown. "
        "You are an expert educational assistant that creates clear, structured study notes."
    )
    assert sanitize_feature_title(raw_prompt) == "Audio to Notes"
    
    # PYQ analysis
    pyq_prompt = "Analyze the provided previous year question papers. Use mathematical probability and pattern recognition..."
    assert sanitize_feature_title(pyq_prompt) == "PYQ Analysis"
    
    # Normal message
    normal = "What is Newton's third law of motion?"
    assert sanitize_feature_title(normal) == normal

def test_resolve_request_instruction_for_audionotes():
    raw_prompt = (
        "Convert the following audio transcript into structured educational notes using clean Markdown. "
        "You are an expert educational assistant that creates clear, structured study notes."
    )
    query, instruction, title = resolve_request_instruction(raw_prompt)
    assert query == "Audio to Notes"
    assert "Convert the following audio transcript" in instruction
    assert title == "Audio to Notes"

def test_resolve_request_instruction_explicit_feature():
    query, instruction, title = resolve_request_instruction("Audio to Notes", feature="audionotes")
    assert query == "Audio to Notes"
    assert "Convert the following audio transcript" in instruction
    assert title == "Audio to Notes"

def test_chat_endpoint_hides_system_prompt(monkeypatch):
    # Mock chat_with_context so it doesn't need LM Studio during unit test
    async def mock_chat(query, history=None, selected_source_ids=None, response_language="English", system_instruction=None):
        assert system_instruction is not None
        assert "Convert the following audio transcript" in system_instruction
        assert query == "Audio to Notes"
        return ("# Introduction to Biology\n## Key Points\n- Cell structure", ["transcript.txt"])

    import main
    monkeypatch.setattr(main, "chat_with_context", mock_chat)

    # User sends raw prompt as message
    raw_prompt = (
        "Convert the following audio transcript into structured educational notes using clean Markdown. "
        "You are an expert educational assistant that creates clear, structured study notes. "
        "Format all sections with bold Markdown headers: # [Topic Title], ## Key Points, ## Explanation, ## Examples, and ## Summary in detailed format."
    )
    res = client.post("/api/chat", json={
        "message": raw_prompt
    })
    assert res.status_code == 200
    data = res.json()
    assert data["display_title"] == "Audio to Notes"
    assert "response" in data
