import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from .connection import get_supabase, supabase_manager
from .config import settings
from .models import (
    UserInDB, NoteInDB, AudioFileInDB, PYQDocumentInDB,
    ChatSessionInDB, ChatMessage, SourceDocumentInDB,
    AIArtifactInDB, UserSettingsInDB, NotebookInDB
)

logger = logging.getLogger("studysnap.crud")

# In-memory local stores for offline/test fallback
_local_notebooks: Dict[str, Dict[str, Any]] = {}
_local_notes: Dict[str, Dict[str, Any]] = {}
_local_sessions: Dict[str, Dict[str, Any]] = {}
_local_messages: List[Dict[str, Any]] = []
_local_documents: Dict[str, Dict[str, Any]] = {}
_local_artifacts: List[Dict[str, Any]] = []
_local_profiles: Dict[str, Dict[str, Any]] = {}

DEMO_USER_ID = "00000000-0000-0000-0000-000000000001"

def _ensure_valid_uuid(val: Any) -> str:
    if not val:
        return DEMO_USER_ID
    val_str = str(val).strip()
    try:
        uuid.UUID(val_str)
        return val_str
    except (ValueError, AttributeError):
        return DEMO_USER_ID


def _format_notebook_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Supabase table record to API / frontend schema."""
    d = dict(item)
    if "owner_id" in d and "user_id" not in d:
        d["user_id"] = d["owner_id"]
    if "user_id" in d and "owner_id" not in d:
        d["owner_id"] = d["user_id"]
    if isinstance(d.get("created_at"), datetime):
        d["created_at"] = d["created_at"].isoformat()
    if isinstance(d.get("updated_at"), datetime):
        d["updated_at"] = d["updated_at"].isoformat()
    if "sources" not in d:
        d["sources"] = []
    if "chat_history" not in d:
        d["chat_history"] = []
    return d


# ==========================================
# 1. USER PROFILES CRUD
# ==========================================
class UserCRUD:
    @staticmethod
    async def get_user_profile(user_id: str, access_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("profiles").select("*").eq("id", user_id).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                logger.warning(f"Supabase profile get error: {e}")
        
        if user_id in _local_profiles:
            return _local_profiles[user_id]

        return {
            "id": user_id,
            "display_name": "Student",
            "full_name": "Student",
            "college": "DKTE Society's Textile and Engineering Institute",
            "college_name": "DKTE Society's Textile and Engineering Institute",
            "degree_program": "B.Tech / B.E.",
            "branch": "Computer Science Engineering (CSE)",
            "academic_year": "Third Year",
            "semester": "Semester 5",
            "graduation_year": "2026",
            "avatar_path": None,
            "avatar_url": None,
            "theme_preference": "dark-modern"
        }

    @staticmethod
    async def upsert_user_profile(user_id: str, profile_data: Dict[str, Any], access_token: Optional[str] = None) -> Dict[str, Any]:
        existing = _local_profiles.get(user_id, {})
        data = {
            "id": user_id,
            "full_name": profile_data.get("full_name") or profile_data.get("name") or profile_data.get("display_name") or existing.get("full_name", "Student"),
            "display_name": profile_data.get("display_name") or profile_data.get("name") or profile_data.get("full_name") or existing.get("display_name", "Student"),
            "college_name": profile_data.get("college_name") or profile_data.get("college") or existing.get("college_name", "DKTE"),
            "college": profile_data.get("college") or profile_data.get("college_name") or existing.get("college", "DKTE"),
            "degree_program": profile_data.get("degree_program") if profile_data.get("degree_program") is not None else existing.get("degree_program", "B.Tech / B.E."),
            "custom_degree": profile_data.get("custom_degree") if "custom_degree" in profile_data else existing.get("custom_degree"),
            "branch": profile_data.get("branch") if profile_data.get("branch") is not None else existing.get("branch", "Computer Science Engineering (CSE)"),
            "custom_branch": profile_data.get("custom_branch") if "custom_branch" in profile_data else existing.get("custom_branch"),
            "academic_year": profile_data.get("academic_year") if profile_data.get("academic_year") is not None else existing.get("academic_year", "Third Year"),
            "semester": profile_data.get("semester") if profile_data.get("semester") is not None else existing.get("semester", "Semester 5"),
            "graduation_year": profile_data.get("graduation_year") if profile_data.get("graduation_year") is not None else existing.get("graduation_year", "2026"),
            "roll_number": profile_data.get("roll_number") if "roll_number" in profile_data else existing.get("roll_number"),
            "avatar_path": profile_data.get("avatar_path") if "avatar_path" in profile_data else existing.get("avatar_path"),
            "avatar_url": profile_data.get("avatar_url") if "avatar_url" in profile_data else existing.get("avatar_url"),
            "theme_preference": profile_data.get("theme_preference") if "theme_preference" in profile_data else existing.get("theme_preference", "dark-modern"),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        _local_profiles[user_id] = data

        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("profiles").upsert(data).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                logger.warning(f"Supabase profile upsert error: {e}")
        return _local_profiles[user_id]


# ==========================================
# 2. NOTEBOOK CRUD OPERATIONS (Supabase PostgreSQL + RLS)
# ==========================================
class NotebookCRUD:
    @staticmethod
    async def create_notebook(notebook_data: Dict[str, Any], access_token: Optional[str] = None) -> Dict[str, Any]:
        """Create a new notebook in Supabase PostgreSQL."""
        nb_id = notebook_data.get("id") or str(uuid.uuid4())
        owner_id = notebook_data.get("user_id") or notebook_data.get("owner_id")
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "id": nb_id,
            "owner_id": owner_id,
            "title": notebook_data.get("title") or "Untitled Notebook",
            "code": notebook_data.get("code") or "CS101",
            "content": notebook_data.get("content") or "",
            "sources_count": notebook_data.get("sources_count", 0),
            "notes_count": notebook_data.get("notes_count", 0),
            "status": notebook_data.get("status", "ready"),
            "created_at": now,
            "updated_at": now
        }

        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("notebooks").insert(record).execute()
                if res.data:
                    saved = _format_notebook_record(res.data[0])
                    _local_notebooks[nb_id] = saved
                    return saved
            except Exception as e:
                logger.warning(f"Supabase notebook insert note: {e}")

        # Local fallback persistence
        saved_fallback = _format_notebook_record({
            **record,
            "user_id": owner_id,
            "sources": notebook_data.get("sources", []),
            "chat_history": notebook_data.get("chat_history", [])
        })
        _local_notebooks[nb_id] = saved_fallback
        return saved_fallback

    @staticmethod
    async def get_user_notebooks(user_id: str, access_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all notebooks for authenticated user, sorted by updated_at descending."""
        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("notebooks").select("*").eq("owner_id", user_id).order("updated_at", desc=True).execute()
                if res.data is not None:
                    return [_format_notebook_record(item) for item in res.data]
            except Exception as e:
                logger.warning(f"Supabase notebook list error: {e}")

        # Local fallback
        matched = [nb for nb in _local_notebooks.values() if nb.get("user_id") == user_id or nb.get("owner_id") == user_id]
        matched.sort(key=lambda x: str(x.get("updated_at", "")), reverse=True)
        return matched

    @staticmethod
    async def get_notebook_by_id(notebook_id: str, user_id: str, access_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve single notebook verifying user ownership."""
        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("notebooks").select("*").eq("id", notebook_id).eq("owner_id", user_id).execute()
                if res.data:
                    return _format_notebook_record(res.data[0])
            except Exception as e:
                logger.warning(f"Supabase notebook get error: {e}")

        # Local fallback
        cached = _local_notebooks.get(notebook_id)
        if cached and (cached.get("user_id") == user_id or cached.get("owner_id") == user_id):
            return cached
        return None

    @staticmethod
    async def update_notebook(notebook_id: str, user_id: str, update_data: Dict[str, Any], access_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Debounced autosave update for notebook verifying ownership."""
        now = datetime.now(timezone.utc).isoformat()
        clean_update: Dict[str, Any] = {}
        for k in ["title", "code", "content", "sources_count", "notes_count", "status"]:
            if k in update_data and update_data[k] is not None:
                clean_update[k] = update_data[k]
        clean_update["updated_at"] = now

        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("notebooks").update(clean_update).eq("id", notebook_id).eq("owner_id", user_id).execute()
                if res.data:
                    updated = _format_notebook_record(res.data[0])
                    _local_notebooks[notebook_id] = updated
                    return updated
            except Exception as e:
                logger.warning(f"Supabase notebook update error: {e}")

        cached = _local_notebooks.get(notebook_id)
        if cached and (cached.get("user_id") == user_id or cached.get("owner_id") == user_id):
            cached.update({k: v for k, v in update_data.items() if v is not None})
            cached["updated_at"] = now
            return cached
        return None

    @staticmethod
    async def delete_notebook(notebook_id: str, user_id: str, access_token: Optional[str] = None) -> bool:
        """Delete notebook verifying ownership."""
        client = get_supabase(access_token)
        deleted = False
        if client:
            try:
                res = client.table("notebooks").delete().eq("id", notebook_id).eq("owner_id", user_id).execute()
                if res.data:
                    deleted = True
            except Exception as e:
                logger.warning(f"Supabase notebook delete error: {e}")

        cached = _local_notebooks.get(notebook_id)
        if cached and (cached.get("user_id") == user_id or cached.get("owner_id") == user_id):
            _local_notebooks.pop(notebook_id, None)
            return True

        return deleted


# ==========================================
# 3. LECTURE NOTES CRUD OPERATIONS
# ==========================================
class NoteCRUD:
    @staticmethod
    async def create_note(note_data: Dict[str, Any], access_token: Optional[str] = None) -> Dict[str, Any]:
        note_id = note_data.get("id") or note_data.get("note_id") or str(uuid.uuid4())
        owner_id = note_data.get("user_id") or note_data.get("owner_id")
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "id": note_id,
            "owner_id": owner_id,
            "notebook_id": note_data.get("notebook_id"),
            "title": note_data.get("title") or note_data.get("subject") or "Lecture Note",
            "subject": note_data.get("subject"),
            "chapter": note_data.get("chapter"),
            "content": note_data.get("formatted_content") or note_data.get("content", ""),
            "summary": note_data.get("summary", ""),
            "keywords": note_data.get("keywords", []),
            "created_at": now,
            "updated_at": now
        }

        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("notes").insert(record).execute()
                if res.data:
                    _local_notes[note_id] = res.data[0]
                    return res.data[0]
            except Exception as e:
                logger.warning(f"Supabase note insert note: {e}")

        _local_notes[note_id] = record
        return record

    @staticmethod
    async def get_user_notes(user_id: str, subject: Optional[str] = None, access_token: Optional[str] = None) -> List[Dict[str, Any]]:
        client = get_supabase(access_token)
        if client:
            try:
                query = client.table("notes").select("*").eq("owner_id", user_id)
                if subject:
                    query = query.eq("subject", subject)
                res = query.order("created_at", desc=True).execute()
                if res.data is not None:
                    return res.data
            except Exception as e:
                logger.warning(f"Supabase note list error: {e}")

        notes = [n for n in _local_notes.values() if n.get("owner_id") == user_id or n.get("user_id") == user_id]
        if subject:
            notes = [n for n in notes if n.get("subject") == subject]
        return notes

    @staticmethod
    async def update_note(note_id: str, update_data: Dict[str, Any], access_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        clean = {k: v for k, v in update_data.items() if v is not None}
        clean["updated_at"] = now

        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("notes").update(clean).eq("id", note_id).execute()
                if res.data:
                    _local_notes[note_id] = res.data[0]
                    return res.data[0]
            except Exception as e:
                logger.warning(f"Supabase note update error: {e}")

        if note_id in _local_notes:
            _local_notes[note_id].update(clean)
            return _local_notes[note_id]
        return None

    @staticmethod
    async def delete_note(note_id: str, access_token: Optional[str] = None) -> bool:
        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("notes").delete().eq("id", note_id).execute()
                if res.data:
                    _local_notes.pop(note_id, None)
                    return True
            except Exception as e:
                logger.warning(f"Supabase note delete error: {e}")

        if note_id in _local_notes:
            _local_notes.pop(note_id, None)
            return True
        return False


# ==========================================
# 4. CHAT SESSIONS & MESSAGES CRUD
# ==========================================
class ChatbotCRUD:
    @staticmethod
    async def create_session(session_data: Dict[str, Any], access_token: Optional[str] = None) -> Dict[str, Any]:
        s_id = session_data.get("session_id") or session_data.get("id") or str(uuid.uuid4())
        owner_id = _ensure_valid_uuid(session_data.get("user_id") or session_data.get("owner_id"))
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "id": s_id,
            "owner_id": owner_id,
            "notebook_id": session_data.get("notebook_id"),
            "title": session_data.get("title", "New Conversation"),
            "selected_source_ids": session_data.get("selected_source_ids", []),
            "created_at": now,
            "updated_at": now
        }

        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("chat_sessions").insert(record).execute()
                if res.data:
                    _local_sessions[s_id] = res.data[0]
                    return res.data[0]
            except Exception as e:
                logger.warning(f"Supabase session insert note: {e}")

        _local_sessions[s_id] = record
        return record

    @staticmethod
    async def list_user_sessions(user_id: str, access_token: Optional[str] = None) -> List[Dict[str, Any]]:
        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("chat_sessions").select("*").eq("owner_id", user_id).order("updated_at", desc=True).execute()
                if res.data is not None:
                    return res.data
            except Exception as e:
                logger.warning(f"Supabase session list error: {e}")

        return [s for s in _local_sessions.values() if s.get("owner_id") == user_id or s.get("user_id") == user_id]

    @staticmethod
    async def get_session(session_id: str, access_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("chat_sessions").select("*").eq("id", session_id).execute()
                if res.data:
                    session = res.data[0]
                    # Also fetch messages
                    msg_res = client.table("chat_messages").select("*").eq("session_id", session_id).order("creation_time", desc=False).execute()
                    session["messages"] = msg_res.data or []
                    return session
            except Exception as e:
                logger.warning(f"Supabase session get error: {e}")

        if session_id in _local_sessions:
            session = dict(_local_sessions[session_id])
            session["messages"] = [m for m in _local_messages if m.get("session_id") == session_id]
            return session
        return None

    @staticmethod
    async def add_message(session_id: str, message_data: Dict[str, Any], access_token: Optional[str] = None) -> Dict[str, Any]:
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": msg_id,
            "session_id": session_id,
            "owner_id": _ensure_valid_uuid(message_data.get("owner_id") or message_data.get("user_id")),
            "role": message_data.get("role", "user"),
            "message_content": message_data.get("content") or message_data.get("message_content", ""),
            "source_references": message_data.get("sources_used") or message_data.get("source_references", []),
            "creation_time": now
        }

        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("chat_messages").insert(record).execute()
                if res.data:
                    _local_messages.append(res.data[0])
                    return res.data[0]
            except Exception as e:
                logger.warning(f"Supabase message insert error: {e}")

        _local_messages.append(record)
        return record

    @staticmethod
    async def delete_session(session_id: str, access_token: Optional[str] = None) -> bool:
        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("chat_sessions").delete().eq("id", session_id).execute()
                if res.data:
                    _local_sessions.pop(session_id, None)
                    return True
            except Exception as e:
                logger.warning(f"Supabase session delete error: {e}")

        if session_id in _local_sessions:
            _local_sessions.pop(session_id, None)
            return True
        return False


# ==========================================
# 5. GENERATED ARTIFACTS / MATERIALS CRUD
# ==========================================
class ArtifactCRUD:
    @staticmethod
    async def save_artifact(artifact_data: Dict[str, Any], access_token: Optional[str] = None) -> Dict[str, Any]:
        art_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": art_id,
            "owner_id": _ensure_valid_uuid(artifact_data.get("owner_id") or artifact_data.get("user_id")),
            "notebook_id": artifact_data.get("notebook_id"),
            "material_type": artifact_data.get("artifact_type") or artifact_data.get("material_type", "general"),
            "title": artifact_data.get("title", "Study Material"),
            "storage_path": artifact_data.get("storage_path"),
            "metadata": artifact_data.get("content_data") or artifact_data.get("metadata", {}),
            "created_at": now,
            "updated_at": now
        }

        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("generated_materials").insert(record).execute()
                if res.data:
                    _local_artifacts.append(res.data[0])
                    return res.data[0]
            except Exception as e:
                logger.warning(f"Supabase artifact insert error: {e}")

        _local_artifacts.append(record)
        return record


# ==========================================
# 6. DOCUMENTS & SOURCE METADATA CRUD
# ==========================================
class SourceCRUD:
    @staticmethod
    async def save_document(doc_data: Dict[str, Any], access_token: Optional[str] = None) -> Dict[str, Any]:
        doc_id = doc_data.get("id") or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        record = {
            "id": doc_id,
            "owner_id": _ensure_valid_uuid(doc_data.get("owner_id") or doc_data.get("user_id")),
            "notebook_id": doc_data.get("notebook_id"),
            "filename": doc_data.get("filename", "document"),
            "file_type": doc_data.get("file_type") or doc_data.get("type", "unknown"),
            "file_size": doc_data.get("file_size", 0),
            "storage_path": doc_data.get("storage_path") or doc_data.get("file_path", ""),
            "processing_status": doc_data.get("processing_status", "completed"),
            "chunk_count": doc_data.get("chunk_count", 0),
            "metadata": doc_data.get("metadata", {}),
            "created_at": now,
            "updated_at": now
        }

        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("documents").insert(record).execute()
                if res.data:
                    _local_documents[doc_id] = res.data[0]
                    return res.data[0]
            except Exception as e:
                logger.warning(f"Supabase document insert error: {e}")

        _local_documents[doc_id] = record
        return record

    @staticmethod
    async def delete_document(doc_id: str, access_token: Optional[str] = None) -> bool:
        client = get_supabase(access_token)
        if client:
            try:
                res = client.table("documents").delete().eq("id", doc_id).execute()
                if res.data:
                    _local_documents.pop(doc_id, None)
                    return True
            except Exception as e:
                logger.warning(f"Supabase document delete error: {e}")

        if doc_id in _local_documents:
            _local_documents.pop(doc_id, None)
            return True
        return False
