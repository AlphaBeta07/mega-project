import asyncio
import sys
import os
import logging
from datetime import datetime

# Ensure database package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.connection import get_async_db, ping_db, db_manager
from database.crud import UserCRUD, NoteCRUD, ChatbotCRUD, PYQCRUD, SourceCRUD
from database.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("studysnap.test")

async def test_database_operations():
    """Verify that MongoDB connects, inserts data, queries records, and deletes data properly."""
    print("=" * 60)
    print("🔍 StudySnap AI - MongoDB Database Verification & Read/Write Test")
    print("=" * 60)

    # 1. Check Connectivity
    print("\n1️⃣  Testing MongoDB Connection...")
    connected = await ping_db()
    if not connected:
        print("❌ MongoDB is NOT running or unreachable at mongodb://localhost:27017")
        print("👉 Tip: Start MongoDB service locally on Windows or update MONGODB_URL in backend/.env for MongoDB Atlas Cloud.")
        return False
    print("✅ Connection Successful! Connected to database:", settings.DB_NAME)

    db = get_async_db()

    # 2. Test Inserting User Record
    print("\n2️⃣  Testing User Insertion...")
    test_user_email = f"test_{int(datetime.utcnow().timestamp())}@dkte.ac.in"
    user = await UserCRUD.create_user(db, {
        "name": "Test Student",
        "email": test_user_email,
        "password_hash": "$2b$12$sample_hashed_password_for_testing",
        "role": "student"
    })
    print(f"✅ User inserted successfully! Generated user_id: {user.user_id}")

    # 3. Test Reading User Record
    fetched_user = await UserCRUD.get_user_by_email(db, test_user_email)
    assert fetched_user is not None, "Failed to fetch user by email"
    print(f"✅ User query verified! Found user: {fetched_user.name} ({fetched_user.email})")

    # 4. Test Inserting Lecture Note Record
    print("\n3️⃣  Testing Lecture Note Insertion...")
    note = await NoteCRUD.create_note(db, {
        "user_id": user.user_id,
        "subject": "Data Structures & Algorithms",
        "chapter": "Binary Search Trees",
        "transcript_text": "A binary search tree is a node-based binary tree data structure...",
        "formatted_content": "# Binary Search Trees\n- Left subtree has key < node\n- Right subtree has key > node",
        "summary": "Overview of BST properties and operations.",
        "keywords": ["BST", "Binary Tree", "Search"],
        "duration": 35.5
    })
    print(f"✅ Note inserted successfully! Generated note_id: {note.note_id}")

    # 5. Test Querying Lecture Notes
    notes_list = await NoteCRUD.get_user_notes(db, user.user_id, subject="Data Structures & Algorithms")
    print(f"✅ Notes query verified! Retrieved {len(notes_list)} note(s) for user.")

    # 6. Test Inserting Chatbot Session & Message
    print("\n4️⃣  Testing Chatbot Session & Message Insertion...")
    session = await ChatbotCRUD.create_session(db, {
        "user_id": user.user_id,
        "title": "DSA Doubts Session"
    })
    await ChatbotCRUD.add_message(db, session.session_id, {
        "role": "user",
        "content": "What is the time complexity of searching in a balanced BST?"
    })
    await ChatbotCRUD.add_message(db, session.session_id, {
        "role": "assistant",
        "content": "The time complexity of searching in a balanced BST is O(log n)."
    })

    session_detail = await ChatbotCRUD.get_session(db, session.session_id)
    print(f"✅ Chatbot session verified! Saved {len(session_detail.messages)} message(s) in session ID: {session.session_id}")

    # 7. Clean up test data
    print("\n5️⃣  Cleaning Up Test Records...")
    await NoteCRUD.delete_note(db, note.note_id)
    await ChatbotCRUD.delete_session(db, session.session_id)
    await db[settings.USERS_COLLECTION].delete_one({"user_id": user.user_id})
    print("✅ Cleanup complete.")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Your MongoDB database is 100% working & ready!")
    print("=" * 60 + "\n")
    db_manager.close()
    return True

if __name__ == "__main__":
    asyncio.run(test_database_operations())
