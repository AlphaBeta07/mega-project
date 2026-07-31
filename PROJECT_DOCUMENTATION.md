# StudySnap AI — Complete Project Documentation

This document provides a comprehensive, end-to-end overview of the **StudySnap AI** project. It covers the full technical stack, the architecture linking the frontend and backend, the core AI integrations, and the division of team responsibilities.

---

## 1. Project Overview

**StudySnap AI** is an intelligent, full-stack application designed to enhance learning and productivity. It allows users to ingest various sources (documents, URLs), and leverages a custom-trained **7-Billion Parameter AI Model** to perform Retrieval-Augmented Generation (RAG). 

Beyond basic text-based chat, the system can synthesize information to generate **Podcast-style Audio Overviews**, **Visual Infographics**, and structured **Mind Maps**.

---

## 2. Core Features

- **Document & Source Ingestion:** Users can upload documents or provide URLs/YouTube links. These are processed, vectorized, and stored in a local ChromaDB database.
- **Context-Aware Chat (RAG):** Users can chat directly with the AI. The system pulls relevant context from the ingested sources to ground the AI's responses.
- **Audio Overviews:** Converts selected sources into engaging, podcast-style audio scripts and synthesizes them into playable audio using TTS (Text-to-Speech).
- **Infographic Generation:** Transforms complex text from sources into visually structured infographics (e.g., Bento Grid style) customized by detail level.
- **Mind Map Generation:** Extracts key concepts and their relationships from sources, converting them into structured JSON which is then rendered interactively on the frontend.
- **Multi-language Support:** Outputs can be dynamically generated in various languages.

---

## 3. Frontend Architecture (User Interface)

The frontend is a modern, reactive Single Page Application (SPA) focusing on smooth UX/UI and dynamic data visualization.

### Tech Stack
- **Framework:** React 19 with TypeScript
- **Build Tool:** Vite (for fast HMR and optimized builds)
- **Styling/Animations:** Framer Motion (for fluid animations)
- **Icons:** Lucide React
- **Diagrams/Mind Maps:** `@xyflow/react` (React Flow for rendering interactive node-based mind maps)
- **Markdown Rendering:** `react-markdown` and `remark-gfm`

### Key Components
- **`App.tsx`**: The main chat interface where users interact with the AI and view messages.
- **`AddSourceModal.tsx`**: Handles drag-and-drop file uploads and URL inputs.
- **`AudioOverviewModal.tsx`**: Interface for generating and playing synthesized podcast overviews of selected sources.
- **`InfographicModal.tsx`**: Allows users to select visual styles (e.g., Bento Grid) and detail levels to generate informational graphics.
- **`MindMapModal.tsx` & `MindMapRenderer.tsx`**: Handles the prompt creation for mind maps and uses React Flow to render the JSON output into a visual, interactive graph.

---

## 4. Backend Architecture (API & Logic)

The backend acts as the orchestration layer, connecting the user interface with the databases and the AI inference engine.

### Tech Stack
- **Framework:** FastAPI (Python)
- **Server:** Uvicorn
- **Vector Database:** ChromaDB (local persistence for RAG)
- **Data Handling:** Pydantic (for strict API payload validation)

### Core Modules
- **`main.py`**: The API Gateway. It exposes endpoints for:
  - `POST /api/sources/upload` and `/api/sources/url`: Ingesting data.
  - `GET /api/sources` and `DELETE /api/sources/{id}`: Managing sources.
  - `POST /api/chat`: Handling the main conversational RAG pipeline.
  - `POST /api/audio-overview`, `/api/infographic`, `/api/mindmap`: Triggering advanced AI generation pipelines.
- **`rag.py`**: The "brain" of the backend operations. It handles chunking documents, generating embeddings, storing them in ChromaDB, and querying the local AI server (LM Studio) to perform context-aware text generation.

---

## 5. AI / Machine Learning Layer

The core intelligence of the application is powered by a custom-trained local LLM, ensuring privacy and offline capability.

### Custom 7B Model Details
- **Base Model:** `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- **Training Method:** Fine-tuned using **LoRA (Low-Rank Adaptation)** on the `yahma/alpaca-cleaned` dataset (52,000 instruction-following examples).
- **Hardware Optimization:** Trained locally on an RTX 4050 (6GB VRAM) using a hybrid GPU + CPU RAM strategy and 4-bit NF4 quantization.
- **Inference Engine:** The merged model is exported as a **GGUF (Q4_K_M)** file and served locally using **LM Studio**. The FastAPI backend communicates with LM Studio using an OpenAI-compatible local API.
