"""
StudySnap AI Database Module
============================
MongoDB Database Package supporting SRS specifications for:
- Users & Authentication
- Lecture Notes & Audio Recordings
- PYQ Papers & Topic Frequency Mapping
- Transcriptions & RAG Source Documents
- Chatbot Sessions & Message History
- AI Study Artifacts (Podcasts, Infographics, Mind Maps)
"""

from .config import settings
from .connection import db_manager, get_async_db, get_sync_db, ping_db
from .models import (
    UserInDB, UserCreate, UserResponse,
    NoteInDB, NoteCreate, NoteUpdate,
    AudioFileInDB, AudioFileCreate,
    PYQDocumentInDB, PYQDocumentCreate,
    TopicMappingInDB, TranscriptionInDB,
    ChatSessionInDB, ChatSessionCreate, ChatMessage,
    SourceDocumentInDB, AIArtifactInDB, UserSettingsInDB
)
from .crud import (
    UserCRUD, NoteCRUD, AudioCRUD, PYQCRUD,
    TopicMappingCRUD, ChatbotCRUD, SourceCRUD, ArtifactCRUD
)

__all__ = [
    "settings",
    "db_manager",
    "get_async_db",
    "get_sync_db",
    "ping_db",
    "UserInDB", "UserCreate", "UserResponse",
    "NoteInDB", "NoteCreate", "NoteUpdate",
    "AudioFileInDB", "AudioFileCreate",
    "PYQDocumentInDB", "PYQDocumentCreate",
    "TopicMappingInDB", "TranscriptionInDB",
    "ChatSessionInDB", "ChatSessionCreate", "ChatMessage",
    "SourceDocumentInDB", "AIArtifactInDB", "UserSettingsInDB",
    "UserCRUD", "NoteCRUD", "AudioCRUD", "PYQCRUD",
    "TopicMappingCRUD", "ChatbotCRUD", "SourceCRUD", "ArtifactCRUD"
]
