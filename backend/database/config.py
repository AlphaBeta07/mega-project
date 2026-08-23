import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

class DatabaseSettings:
    """MongoDB Database Settings for StudySnap AI."""
    
    # MongoDB Connection String
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    
    # Primary Database Name
    DB_NAME: str = os.getenv("DB_NAME", "studysnap_db")
    
    # Collection Names based on System SRS Specification
    USERS_COLLECTION: str = "users"
    NOTES_COLLECTION: str = "notes"
    AUDIO_FILES_COLLECTION: str = "audio_files"
    PYQ_DOCUMENTS_COLLECTION: str = "pyq_documents"
    TOPIC_MAPPINGS_COLLECTION: str = "topic_mappings"
    CHATBOT_SESSIONS_COLLECTION: str = "chatbot_sessions"
    CHATBOT_MESSAGES_COLLECTION: str = "chatbot_messages"
    TRANSCRIPTIONS_COLLECTION: str = "transcriptions"
    SOURCES_COLLECTION: str = "sources"
    ARTIFACTS_COLLECTION: str = "ai_artifacts"
    USER_SETTINGS_COLLECTION: str = "user_settings"

settings = DatabaseSettings()
