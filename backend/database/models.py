from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# Helper to generate string UUIDs
def generate_uuid() -> str:
    return str(uuid.uuid4())

# ==========================================
# 1. USER MODEL (SRS Entity: Users)
# ==========================================
class UserBase(BaseModel):
    name: str = Field(..., description="Student or Lecturer Full Name")
    email: EmailStr = Field(..., description="Unique User Email")
    role: str = Field(default="student", description="Role: student, lecturer, or admin")

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="Raw Password")

class UserInDB(UserBase):
    user_id: str = Field(default_factory=generate_uuid, description="Unique User ID")
    password_hash: str = Field(..., description="Bcrypt Hashed Password")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    auth_token: Optional[str] = None

class UserResponse(UserBase):
    user_id: str
    created_at: datetime

# ==========================================
# 2. LECTURE NOTE MODEL (SRS Entity: Notes)
# ==========================================
class NoteBase(BaseModel):
    subject: str = Field(..., description="Subject Name (e.g. Operating Systems)")
    chapter: str = Field(..., description="Chapter Title or Module Label")
    transcript_text: Optional[str] = Field(default="", description="Raw Speech-to-Text transcript")
    formatted_content: str = Field(..., description="AI-Structured academic notes (Markdown)")
    summary: Optional[str] = Field(default="", description="High-level summary")
    keywords: List[str] = Field(default_factory=list, description="Key concepts and definitions")
    duration: Optional[float] = Field(default=0.0, description="Lecture duration in minutes")

class NoteCreate(NoteBase):
    user_id: str = Field(..., description="Owner User ID")
    audio_file_id: Optional[str] = Field(default=None, description="Linked Audio File ID")

class NoteInDB(NoteBase):
    note_id: str = Field(default_factory=generate_uuid)
    user_id: str
    audio_file_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class NoteUpdate(BaseModel):
    subject: Optional[str] = None
    chapter: Optional[str] = None
    formatted_content: Optional[str] = None
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None

# ==========================================
# 3. AUDIO FILE MODEL (SRS Entity: AudioFiles)
# ==========================================
class AudioFileBase(BaseModel):
    file_name: str = Field(..., description="Original Audio Filename")
    mime_type: str = Field(default="audio/mp3", description="MIME type")
    file_size: int = Field(..., description="File size in bytes (max 200MB)")

class AudioFileCreate(AudioFileBase):
    user_id: str
    storage_url: str

class AudioFileInDB(AudioFileBase):
    file_id: str = Field(default_factory=generate_uuid)
    user_id: str
    storage_url: str
    upload_time: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# 4. PYQ DOCUMENT MODEL (SRS Entity: PYQDocuments)
# ==========================================
class PYQDocumentBase(BaseModel):
    subject: str = Field(..., description="Subject Name")
    year: int = Field(..., description="Examination Year")
    extracted_text: str = Field(..., description="Extracted Question Text")

class PYQDocumentCreate(PYQDocumentBase):
    user_id: str
    topic_frequencies: Optional[Dict[str, int]] = Field(default_factory=dict)

class PYQDocumentInDB(PYQDocumentBase):
    pyq_id: str = Field(default_factory=generate_uuid)
    user_id: str
    topic_frequencies: Dict[str, int] = Field(default_factory=dict)
    upload_date: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# 5. TOPIC MAPPING MODEL (SRS Entity: TopicMapping)
# ==========================================
class TopicMappingBase(BaseModel):
    note_id: str
    pyq_id: str
    topic_name: str
    frequency: int = Field(default=1)
    priority_flag: str = Field(default="High", description="Importance Priority: High, Medium, Low")

class TopicMappingInDB(TopicMappingBase):
    mapping_id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# 6. TRANSCRIPTION MODEL (SRS Entity: Transcription)
# ==========================================
class TranscriptionInDB(BaseModel):
    transcription_id: str = Field(default_factory=generate_uuid)
    file_id: str
    raw_text: str
    language: str = Field(default="English")
    wer_score: Optional[float] = Field(default=None, description="Word Error Rate estimate")
    processed_at: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# 7. CHATBOT SESSION & MESSAGES MODEL (SRS Entity: ChatbotSessions)
# ==========================================
class ChatMessage(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    sources_used: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatSessionCreate(BaseModel):
    user_id: str = Field(default="default_user")
    title: str = Field(default="New Lecture Doubt Chat")
    selected_source_ids: Optional[List[str]] = Field(default_factory=list)

class ChatSessionInDB(BaseModel):
    session_id: str = Field(default_factory=generate_uuid)
    user_id: str = Field(default="default_user")
    title: str
    messages: List[ChatMessage] = Field(default_factory=list)
    selected_source_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# 8. SOURCE DOCUMENT MODEL (StudySnap RAG Ingestion)
# ==========================================
class SourceDocumentInDB(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    filename: str
    type: str = Field(description="File extension or URL type: pdf, docx, txt, audio, url, youtube")
    file_path: Optional[str] = None
    url: Optional[str] = None
    raw_text: Optional[str] = None
    chunk_count: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# 9. AI ARTIFACT MODEL (Audio Podcast, Infographic, Mind Map)
# ==========================================
class AIArtifactInDB(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    user_id: str = Field(default="default_user")
    artifact_type: str = Field(..., description="audio_overview, infographic, mind_map")
    title: str
    content_data: Dict[str, Any] = Field(description="URL or raw JSON structure")
    selected_source_ids: List[str] = Field(default_factory=list)
    response_language: str = Field(default="English")
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ==========================================
# 10. USER SETTINGS MODEL
# ==========================================
class UserSettingsInDB(BaseModel):
    user_id: str = Field(default="default_user")
    response_language: str = Field(default="English")
    research_mode: str = Field(default="Fast Research")
    theme: str = Field(default="dark")
    updated_at: datetime = Field(default_factory=datetime.utcnow)
