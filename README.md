# Notes Pathv
**Lecture to Notes AI**

Welcome to **Notes Pathv**, an AI-powered educational application built with Streamlit! This application allows you to upload recorded lectures (or record them live) and instantly generates highly structured, premium lecture notes, saving you hours of manual transcription and formatting. It also acts as an educational chatbot, allowing you to ask questions with or without a lecture's context.

The UI is meticulously designed with a premium, Gemini-inspired aesthetic for a highly responsive and modern feel.

---

## Key Features

- **Multi-Format Uploads**: Support for `mp3`, `m4a`, `wav`, `mp4`, and `webm` files.
- **Live Recording**: Record lectures directly from your browser's microphone.
- **Lightning Fast Transcription**: Powered by Groq's high-speed Whisper (`whisper-large-v3`) API for accurate audio-to-text conversion.
- **Structured Note Generation**: Automatically summarizes your transcript into detailed structured notes using Groq's `llama-3.1-8b-instant`. Notes include:
  - Lecture Titles
  - Key Topics
  - Important Definitions
  - Detailed Bullet Points
  - Summary
  - Possible Exam Questions
- **Context-Aware Chat**: Ask specific questions regarding a past lecture, or interact with a general AI Study Assistant. Powered by `llama-3.3-70b-versatile`.
- **Lecture History**: Seamlessly access all previously saved lectures and their generated notes through the sidebar.

---

## Getting Started

Follow these steps to run the application locally.

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your machine. You will also need a Groq API key to power the AI models.

### 2. Installation Setup
Clone or download the repository, then navigate to your project directory:

```bash
cd /path/to/mega_project
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root of your project directory (or copy the provided `.env.example` file) and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```
*(You can obtain an API key by signing up on the [Groq Console](https://console.groq.com/keys))*

### 4. Run the App
Start the Streamlit development server:

```bash
streamlit run app.py
```
Your browser should automatically open the app at `http://localhost:8501`.

---

## 🛠️ Technology Stack

- **Frontend/Backend**: [Streamlit](https://streamlit.io/) (requires `>=1.38.0` for local mic feature)
- **AI Backend / Inference**: [Groq API](https://groq.com/)
- **Transcription Model**: `whisper-large-v3`
- **Text & Chat Models**: `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`

---

## Repository Structure

```
├── app.py                  # Main application flow, UI styling, and AI integration
├── history.json            # Automatically generated file saving your lecture histories
├── temp_audio/             # Auto-generated directory for audio processing 
├── requirements.txt        # Required python packages
├── .env.example            # Environment variables example template
└── README.md               # Project documentation (You are here!)
```

---

## Usage Tips
- **Navigating History**: Your transcribed lectures dynamically populate the Sidebar for quick retrieval.
- **Clearing Context**: To start a general-purpose chat not bound to recent lecture notes, simply click the "⊕ New Lecture Notes" button in the Sidebar.
