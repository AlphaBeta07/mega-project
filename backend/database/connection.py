from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.database import Database as SyncDatabase
import logging
from typing import Optional

from .config import settings

logger = logging.getLogger("studysnap.database")

class DatabaseManager:
    """Async Motor and Sync PyMongo Connection Manager for StudySnap AI."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.sync_client: Optional[MongoClient] = None
        self.sync_db: Optional[SyncDatabase] = None

    def connect(self) -> AsyncIOMotorDatabase:
        """Initialize Async Motor MongoDB client."""
        if not self.client:
            logger.info(f"Connecting to MongoDB at: {settings.MONGODB_URL}")
            self.client = AsyncIOMotorClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)
            self.db = self.client[settings.DB_NAME]
        return self.db

    def connect_sync(self) -> SyncDatabase:
        """Initialize Synchronous PyMongo client."""
        if not self.sync_client:
            self.sync_client = MongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)
            self.sync_db = self.sync_client[settings.DB_NAME]
        return self.sync_db

    def close(self):
        """Close all active MongoDB connections."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("Closed Motor MongoDB Async Connection.")
        if self.sync_client:
            self.sync_client.close()
            self.sync_client = None
            self.sync_db = None
            logger.info("Closed PyMongo Sync Connection.")

db_manager = DatabaseManager()

def get_async_db() -> AsyncIOMotorDatabase:
    """Dependency helper to get active Motor async database instance."""
    return db_manager.connect()

def get_sync_db() -> SyncDatabase:
    """Dependency helper to get active PyMongo sync database instance."""
    return db_manager.connect_sync()

async def ping_db() -> bool:
    """Ping MongoDB server to verify active connection status."""
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URL, serverSelectionTimeoutMS=3000)
        await client.admin.command('ping')
        client.close()
        return True
    except Exception as e:
        logger.error(f"MongoDB Ping Error: {e}")
        return False
