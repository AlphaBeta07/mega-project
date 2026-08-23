from typing import List, Optional, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from .connection import get_async_db
from .config import settings
from .models import (
    UserInDB, NoteInDB, AudioFileInDB, PYQDocumentInDB,
    TopicMappingInDB, TranscriptionInDB, ChatSessionInDB,
    ChatMessage, SourceDocumentInDB, AIArtifactInDB, UserSettingsInDB
)

# ==========================================
# 1. USER CRUD OPERATIONS
# ==========================================
class UserCRUD:
    @staticmethod
    async def create_user(db: AsyncIOMotorDatabase, user_data: Dict[str, Any]) -> UserInDB:
        user = UserInDB(**user_data)
        doc = user.model_dump()
        await db[settings.USERS_COLLECTION].insert_one(doc)
        return user

    @staticmethod
    async def get_user_by_email(db: AsyncIOMotorDatabase, email: str) -> Optional[UserInDB]:
        doc = await db[settings.USERS_COLLECTION].find_one({"email": email})
        return UserInDB(**doc) if doc else None

    @staticmethod
    async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: str) -> Optional[UserInDB]:
        doc = await db[settings.USERS_COLLECTION].find_one({"user_id": user_id})
        return UserInDB(**doc) if doc else None

# ==========================================
# 2. LECTURE NOTES CRUD OPERATIONS
# ==========================================
class NoteCRUD:
    @staticmethod
    async def create_note(db: AsyncIOMotorDatabase, note_data: Dict[str, Any]) -> NoteInDB:
        note = NoteInDB(**note_data)
        doc = note.model_dump()
        await db[settings.NOTES_COLLECTION].insert_one(doc)
        return note

    @staticmethod
    async def get_note_by_id(db: AsyncIOMotorDatabase, note_id: str) -> Optional[NoteInDB]:
        doc = await db[settings.NOTES_COLLECTION].find_one({"note_id": note_id})
        return NoteInDB(**doc) if doc else None

    @staticmethod
    async def get_user_notes(db: AsyncIOMotorDatabase, user_id: str, subject: Optional[str] = None) -> List[NoteInDB]:
        query: Dict[str, Any] = {"user_id": user_id}
        if subject:
            query["subject"] = subject
        cursor = db[settings.NOTES_COLLECTION].find(query).sort("created_at", -1)
        notes = []
        async for doc in cursor:
            notes.append(NoteInDB(**doc))
        return notes

    @staticmethod
    async def update_note(db: AsyncIOMotorDatabase, note_id: str, update_data: Dict[str, Any]) -> Optional[NoteInDB]:
        update_data["updated_at"] = datetime.utcnow()
        result = await db[settings.NOTES_COLLECTION].find_one_and_update(
            {"note_id": note_id},
            {"$set": update_data},
            return_document=True
        )
        return NoteInDB(**result) if result else None

    @staticmethod
    async def delete_note(db: AsyncIOMotorDatabase, note_id: str) -> bool:
        res = await db[settings.NOTES_COLLECTION].delete_one({"note_id": note_id})
        return res.deleted_count > 0

# ==========================================
# 3. AUDIO FILES CRUD OPERATIONS
# ==========================================
class AudioCRUD:
    @staticmethod
    async def save_audio_file(db: AsyncIOMotorDatabase, audio_data: Dict[str, Any]) -> AudioFileInDB:
        audio = AudioFileInDB(**audio_data)
        doc = audio.model_dump()
        await db[settings.AUDIO_FILES_COLLECTION].insert_one(doc)
        return audio

    @staticmethod
    async def get_user_audio_files(db: AsyncIOMotorDatabase, user_id: str) -> List[AudioFileInDB]:
        cursor = db[settings.AUDIO_FILES_COLLECTION].find({"user_id": user_id}).sort("upload_time", -1)
        files = []
        async for doc in cursor:
            files.append(AudioFileInDB(**doc))
        return files

    @staticmethod
    async def delete_audio_file(db: AsyncIOMotorDatabase, file_id: str) -> bool:
        res = await db[settings.AUDIO_FILES_COLLECTION].delete_one({"file_id": file_id})
        return res.deleted_count > 0

# ==========================================
# 4. PYQ DOCUMENTS CRUD OPERATIONS
# ==========================================
class PYQCRUD:
    @staticmethod
    async def save_pyq_document(db: AsyncIOMotorDatabase, pyq_data: Dict[str, Any]) -> PYQDocumentInDB:
        pyq = PYQDocumentInDB(**pyq_data)
        doc = pyq.model_dump()
        await db[settings.PYQ_DOCUMENTS_COLLECTION].insert_one(doc)
        return pyq

    @staticmethod
    async def get_pyqs_by_subject(db: AsyncIOMotorDatabase, subject: str) -> List[PYQDocumentInDB]:
        cursor = db[settings.PYQ_DOCUMENTS_COLLECTION].find({"subject": subject}).sort("year", -1)
        pyqs = []
        async for doc in cursor:
            pyqs.append(PYQDocumentInDB(**doc))
        return pyqs

# ==========================================
# 5. TOPIC MAPPING CRUD OPERATIONS
# ==========================================
class TopicMappingCRUD:
    @staticmethod
    async def save_mapping(db: AsyncIOMotorDatabase, mapping_data: Dict[str, Any]) -> TopicMappingInDB:
        mapping = TopicMappingInDB(**mapping_data)
        doc = mapping.model_dump()
        await db[settings.TOPIC_MAPPINGS_COLLECTION].insert_one(doc)
        return mapping

    @staticmethod
    async def get_mappings_for_note(db: AsyncIOMotorDatabase, note_id: str) -> List[TopicMappingInDB]:
        cursor = db[settings.TOPIC_MAPPINGS_COLLECTION].find({"note_id": note_id})
        mappings = []
        async for doc in cursor:
            mappings.append(TopicMappingInDB(**doc))
        return mappings

# ==========================================
# 6. CHATBOT SESSIONS & MESSAGES CRUD
# ==========================================
class ChatbotCRUD:
    @staticmethod
    async def create_session(db: AsyncIOMotorDatabase, session_data: Dict[str, Any]) -> ChatSessionInDB:
        session = ChatSessionInDB(**session_data)
        doc = session.model_dump()
        await db[settings.CHATBOT_SESSIONS_COLLECTION].insert_one(doc)
        return session

    @staticmethod
    async def get_session(db: AsyncIOMotorDatabase, session_id: str) -> Optional[ChatSessionInDB]:
        doc = await db[settings.CHATBOT_SESSIONS_COLLECTION].find_one({"session_id": session_id})
        return ChatSessionInDB(**doc) if doc else None

    @staticmethod
    async def list_user_sessions(db: AsyncIOMotorDatabase, user_id: str = "default_user") -> List[ChatSessionInDB]:
        cursor = db[settings.CHATBOT_SESSIONS_COLLECTION].find({"user_id": user_id}).sort("updated_at", -1)
        sessions = []
        async for doc in cursor:
            sessions.append(ChatSessionInDB(**doc))
        return sessions

    @staticmethod
    async def add_message(db: AsyncIOMotorDatabase, session_id: str, message_data: Dict[str, Any]) -> Optional[ChatSessionInDB]:
        message = ChatMessage(**message_data)
        msg_doc = message.model_dump()
        now = datetime.utcnow()
        result = await db[settings.CHATBOT_SESSIONS_COLLECTION].find_one_and_update(
            {"session_id": session_id},
            {
                "$push": {"messages": msg_doc},
                "$set": {"updated_at": now}
            },
            return_document=True
        )
        return ChatSessionInDB(**result) if result else None

    @staticmethod
    async def delete_session(db: AsyncIOMotorDatabase, session_id: str) -> bool:
        res = await db[settings.CHATBOT_SESSIONS_COLLECTION].delete_one({"session_id": session_id})
        return res.deleted_count > 0

# ==========================================
# 7. SOURCE DOCUMENTS CRUD (RAG Ingestion)
# ==========================================
class SourceCRUD:
    @staticmethod
    async def save_source(db: AsyncIOMotorDatabase, source_data: Dict[str, Any]) -> SourceDocumentInDB:
        source = SourceDocumentInDB(**source_data)
        doc = source.model_dump()
        await db[settings.SOURCES_COLLECTION].replace_one({"id": source.id}, doc, upsert=True)
        return source

    @staticmethod
    async def get_all_sources(db: AsyncIOMotorDatabase) -> List[SourceDocumentInDB]:
        cursor = db[settings.SOURCES_COLLECTION].find().sort("created_at", -1)
        sources = []
        async for doc in cursor:
            sources.append(SourceDocumentInDB(**doc))
        return sources

    @staticmethod
    async def delete_source(db: AsyncIOMotorDatabase, source_id: str) -> bool:
        res = await db[settings.SOURCES_COLLECTION].delete_one({"id": source_id})
        return res.deleted_count > 0

# ==========================================
# 8. AI ARTIFACTS CRUD
# ==========================================
class ArtifactCRUD:
    @staticmethod
    async def save_artifact(db: AsyncIOMotorDatabase, artifact_data: Dict[str, Any]) -> AIArtifactInDB:
        artifact = AIArtifactInDB(**artifact_data)
        doc = artifact.model_dump()
        await db[settings.ARTIFACTS_COLLECTION].insert_one(doc)
        return artifact

    @staticmethod
    async def get_artifacts_by_type(db: AsyncIOMotorDatabase, artifact_type: str, user_id: str = "default_user") -> List[AIArtifactInDB]:
        cursor = db[settings.ARTIFACTS_COLLECTION].find({"artifact_type": artifact_type, "user_id": user_id}).sort("created_at", -1)
        artifacts = []
        async for doc in cursor:
            artifacts.append(AIArtifactInDB(**doc))
        return artifacts
