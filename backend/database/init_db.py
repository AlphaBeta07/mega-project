import asyncio
import sys
import os
import logging
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT

# Ensure database package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.config import settings
from database.connection import get_async_db, ping_db, db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("studysnap.init_db")

async def init_database():
    """Initialize MongoDB Database, Collections, and Performance Indexes."""
    logger.info("Initializing StudySnap AI MongoDB Database...")
    
    # 1. Test Connectivity
    connected = await ping_db()
    if not connected:
        logger.error(f"Failed to connect to MongoDB at {settings.MONGODB_URL}. Please ensure MongoDB server is running.")
        sys.exit(1)
        
    db = get_async_db()
    logger.info(f"Connected to Database: '{settings.DB_NAME}'")

    # 2. Setup Indexes for Collections
    try:
        # Users Collection Indexes
        await db[settings.USERS_COLLECTION].create_index([("email", ASCENDING)], unique=True)
        await db[settings.USERS_COLLECTION].create_index([("user_id", ASCENDING)], unique=True)
        logger.info("✓ Created indexes for 'users' collection.")

        # Notes Collection Indexes
        await db[settings.NOTES_COLLECTION].create_index([("note_id", ASCENDING)], unique=True)
        await db[settings.NOTES_COLLECTION].create_index([("user_id", ASCENDING)])
        await db[settings.NOTES_COLLECTION].create_index([("subject", ASCENDING)])
        await db[settings.NOTES_COLLECTION].create_index([
            ("formatted_content", TEXT),
            ("transcript_text", TEXT),
            ("subject", TEXT),
            ("chapter", TEXT)
        ], name="notes_text_search_index")
        logger.info("✓ Created indexes & full-text search for 'notes' collection.")

        # Audio Files Collection Indexes
        await db[settings.AUDIO_FILES_COLLECTION].create_index([("file_id", ASCENDING)], unique=True)
        await db[settings.AUDIO_FILES_COLLECTION].create_index([("user_id", ASCENDING)])
        logger.info("✓ Created indexes for 'audio_files' collection.")

        # PYQ Documents Collection Indexes
        await db[settings.PYQ_DOCUMENTS_COLLECTION].create_index([("pyq_id", ASCENDING)], unique=True)
        await db[settings.PYQ_DOCUMENTS_COLLECTION].create_index([("user_id", ASCENDING), ("subject", ASCENDING)])
        logger.info("✓ Created indexes for 'pyq_documents' collection.")

        # Topic Mappings Collection Indexes
        await db[settings.TOPIC_MAPPINGS_COLLECTION].create_index([("mapping_id", ASCENDING)], unique=True)
        await db[settings.TOPIC_MAPPINGS_COLLECTION].create_index([("note_id", ASCENDING)])
        logger.info("✓ Created indexes for 'topic_mappings' collection.")

        # Chatbot Sessions Collection Indexes
        await db[settings.CHATBOT_SESSIONS_COLLECTION].create_index([("session_id", ASCENDING)], unique=True)
        await db[settings.CHATBOT_SESSIONS_COLLECTION].create_index([("user_id", ASCENDING)])
        logger.info("✓ Created indexes for 'chatbot_sessions' collection.")

        # Sources Collection Indexes
        await db[settings.SOURCES_COLLECTION].create_index([("id", ASCENDING)], unique=True)
        logger.info("✓ Created indexes for 'sources' collection.")

        # Artifacts Collection Indexes
        await db[settings.ARTIFACTS_COLLECTION].create_index([("id", ASCENDING)], unique=True)
        await db[settings.ARTIFACTS_COLLECTION].create_index([("user_id", ASCENDING), ("artifact_type", ASCENDING)])
        logger.info("✓ Created indexes for 'ai_artifacts' collection.")

        logger.info("🎉 StudySnap AI Database initialization complete! All collections and indexes are ready.")

    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    asyncio.run(init_database())
