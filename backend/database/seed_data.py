import asyncio
import sys
import os
import logging
from datetime import datetime

# Ensure database package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.connection import get_async_db, db_manager
from database.crud import UserCRUD, NoteCRUD, PYQCRUD, ChatbotCRUD, SourceCRUD
from database.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("studysnap.seed")

async def seed_database():
    """Seed sample test data into StudySnap AI MongoDB database."""
    logger.info("Seeding test dataset into StudySnap AI Database...")
    db = get_async_db()

    try:
        # 1. Seed Demo User
        existing_user = await UserCRUD.get_user_by_email(db, "demo.student@dkte.ac.in")
        if not existing_user:
            user = await UserCRUD.create_user(db, {
                "name": "Anish Landge (Demo Student)",
                "email": "demo.student@dkte.ac.in",
                "password_hash": "$2b$12$eImiTXuWVxfM37uY4JANjO5E.y5bA6gP6aG0S9u4Gk0u0m0m0m0m0", # Dummy hash
                "role": "student"
            })
            user_id = user.user_id
            logger.info(f"✓ Seeded User: {user.name} ({user.email})")
        else:
            user_id = existing_user.user_id
            logger.info(f"Using existing user: {user_id}")

        # 2. Seed Sample Lecture Note
        note = await NoteCRUD.create_note(db, {
            "user_id": user_id,
            "subject": "Operating Systems",
            "chapter": "Module 1: Process & Thread Management",
            "transcript_text": "Processes are executing programs. A process context includes program counter, registers, and variables...",
            "formatted_content": "# Module 1: Process Management\n\n## 1. Process Concept\n- A **process** is an instance of a program in execution.\n- Contains text section, data section, heap, and stack.\n\n## 2. Process Control Block (PCB)\n- Stores PID, process state, program counter, and CPU registers.",
            "summary": "Covers basic process definitions, memory layout, and Process Control Block structure.",
            "keywords": ["Process", "PCB", "Context Switch", "Thread"],
            "duration": 45.0
        })
        logger.info(f"✓ Seeded Sample Note ID: {note.note_id} for subject '{note.subject}'")

        # 3. Seed Sample PYQ Document
        pyq = await PYQCRUD.save_pyq_document(db, {
            "user_id": user_id,
            "subject": "Operating Systems",
            "year": 2024,
            "extracted_text": "Q1. Explain Process Control Block (PCB) with a neat diagram. (8 Marks)\nQ2. Differentiate between Process and Thread. (6 Marks)",
            "topic_frequencies": {
                "Process Control Block": 4,
                "Process vs Thread": 3,
                "Context Switching": 2
            }
        })
        logger.info(f"✓ Seeded Sample PYQ ID: {pyq.pyq_id} for subject '{pyq.subject}'")

        # 4. Seed Chatbot Session
        session = await ChatbotCRUD.create_session(db, {
            "user_id": user_id,
            "title": "Operating Systems Doubts Session",
            "selected_source_ids": []
        })
        await ChatbotCRUD.add_message(db, session.session_id, {
            "role": "user",
            "content": "What is PCB in Operating Systems?"
        })
        await ChatbotCRUD.add_message(db, session.session_id, {
            "role": "assistant",
            "content": "A Process Control Block (PCB) is a data structure maintained by the Operating System for every process. It contains PID, process state, CPU registers, and scheduling information."
        })
        logger.info(f"✓ Seeded Sample Chatbot Session ID: {session.session_id}")

        logger.info("🎉 Database seeding completed successfully!")

    except Exception as e:
        logger.error(f"Error during database seeding: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    asyncio.run(seed_database())
