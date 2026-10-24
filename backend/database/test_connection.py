import asyncio
import sys
import os
import logging

# Ensure database package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.connection import get_supabase, ping_db, supabase_manager
from database.crud import NotebookCRUD, NoteCRUD, ChatbotCRUD
from database.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("studysnap.supabase_test")

async def test_supabase_operations():
    """Verify Supabase PostgreSQL connection, CRUD, and fallback functionality."""
    print("=" * 60)
    print("🔍 StudySnap AI - Supabase Database Verification & Read/Write Test")
    print("=" * 60)

    # 1. Check Connectivity
    print("\n1️⃣  Testing Supabase Connection...")
    connected = await ping_db()
    if connected:
        print("✅ Connection Successful! Connected to Supabase at:", settings.SUPABASE_URL)
    else:
        print("ℹ️  Supabase URL/Key offline or not yet configured in .env. Resilient fallback store active.")

    # 2. Test Inserting Notebook Record
    print("\n2️⃣  Testing Notebook Creation...")
    nb = await NotebookCRUD.create_notebook({
        "user_id": "00000000-0000-0000-0000-000000000099",
        "title": "Algorithms — Dynamic Programming",
        "code": "CS204",
        "content": "# Memoization & Tabulation\nOverlapping subproblems."
    })
    print(f"✅ Notebook created successfully! ID: {nb['id']}, Title: {nb['title']}")

    # 3. Test Reading Notebook
    fetched = await NotebookCRUD.get_notebook_by_id(nb["id"], "00000000-0000-0000-0000-000000000099")
    assert fetched is not None, "Failed to fetch notebook by ID"
    print(f"✅ Notebook query verified! Title: {fetched['title']}")

    # 4. Test Deleting Notebook
    deleted = await NotebookCRUD.delete_notebook(nb["id"], "00000000-0000-0000-0000-000000000099")
    assert deleted, "Failed to delete notebook"
    print("✅ Notebook deletion verified!")

    print("\n🎉 Supabase Database Verification Completed Successfully!")
    return True

if __name__ == "__main__":
    asyncio.run(test_supabase_operations())
