"""
StudySnap AI Database Module
============================
Supabase PostgreSQL & Storage Package supporting:
- Profiles & Authentication (Supabase Auth)
- Notebooks (User Study Workspaces)
- Lecture Notes & Audio Recordings
- Documents & RAG Source Embeddings
- Chat Sessions & Message History
- AI Generated Materials (Podcasts, Infographics, Mind Maps, Quizzes, Flashcards)
"""

from .config import settings
from .connection import supabase_manager, get_supabase, ping_db
from .models import (
    UserInDB, UserCreate, UserResponse,
    NoteInDB, NoteCreate, NoteUpdate,
    AudioFileInDB, AudioFileCreate,
    PYQDocumentInDB, PYQDocumentCreate,
    TopicMappingInDB, TranscriptionInDB,
    ChatSessionInDB, ChatSessionCreate, ChatMessage,
    SourceDocumentInDB, AIArtifactInDB, UserSettingsInDB,
    NotebookInDB, NotebookCreate, NotebookUpdate, NotebookResponse
)
from .crud import (
    UserCRUD, NoteCRUD, ChatbotCRUD, SourceCRUD, ArtifactCRUD,
    NotebookCRUD
)

# Provide backward-compatibility aliases if needed
db_manager = supabase_manager
get_async_db = lambda: supabase_manager.get_client()
get_sync_db = lambda: supabase_manager.get_client()

__all__ = [
    "settings",
    "supabase_manager",
    "get_supabase",
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
    "NotebookInDB", "NotebookCreate", "NotebookUpdate", "NotebookResponse",
    "UserCRUD", "NoteCRUD", "ChatbotCRUD", "SourceCRUD", "ArtifactCRUD",
    "NotebookCRUD"
]
