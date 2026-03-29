import streamlit as st
import os
import time
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

# Load env variables (looks in current dir, and parent dirs if needed)
load_dotenv()
load_dotenv(dotenv_path="../.env") # Support finding .env from d:\mega_project\.env too

# Setup Groq API
api_key = os.getenv("GROQ_API_KEY")
client = None
if api_key:
    client = Groq(api_key=api_key)

# Filesystem config
HISTORY_FILE = "history.json"
TEMP_DIR = "temp_audio"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Page Config
st.set_page_config(
    page_title="Notes Pathv - Lecture -> Notes AI",
    # page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

/* Font & Base Overrides */
html, body, [class*="css"] {
    font-family: 'Google Sans', 'Inter', sans-serif;
}
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
}

/* Gemini Flat Backgrounds */
[data-testid="stAppViewContainer"] {
    background-color: #131314;
    color: #e3e3e3;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 1.5rem !important;
}
.block-container {
    padding-top: 1.5rem !important;
}
[data-testid="stSidebar"] {
    background-color: #1e1f20 !important;
    border-right: none;
}

/* Override Streamlit's default collapse sidebar icon color to blend in */
[data-testid="stSidebarCollapseButton"] button {
    color: #c4c7c5 !important;
}

/* Sidebar Title */
.sidebar-title {
    font-size: 22px;
    font-weight: 500;
    color: #e3e3e3;
    margin-bottom: 30px;
    letter-spacing: -0.2px;
}
.sidebar-title span {
    /* Gemini classic gradient */
    background: linear-gradient(74deg, #4285f4 0, #9b72cb 9%, #d96570 20%, #d96570 24%, #9b72cb 35%, #4285f4 44%);
    background-size: 400% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Primary Button (New Lecture) */
[data-testid="stSidebar"] button[kind="primary"] {
    width: 100%;
    border-radius: 24px;
    background-color: #282a2c;
    color: #c4c7c5;
    border: none;
    padding: 12px 15px;
    font-weight: 500;
    font-size: 14px;
    box-shadow: none;
    transition: background-color 0.2s;
}
[data-testid="stSidebar"] button[kind="primary"]:hover {
    background-color: #333537;
    color: #e3e3e3;
    transform: none;
    box-shadow: none;
}

/* Secondary Button (History items) */
[data-testid="stSidebar"] button[kind="secondary"] {
    width: 100%;
    border-radius: 24px;
    background-color: transparent;
    color: #e3e3e3;
    border: none;
    text-align: left;
    padding: 10px 14px;
    justify-content: flex-start;
    font-weight: 400;
    font-size: 14px;
    margin-bottom: 2px;
    transition: background-color 0.2s;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background-color: #282a2c;
    color: #e3e3e3;
    transform: none;
}

/* Saved Lectures text */
.saved-lectures-label {
    font-size: 11px;
    color: #c4c7c5;
    text-transform: none;
    margin-top: 30px;
    margin-bottom: 10px;
    font-weight: 500;
    padding-left: 10px;
}

/* Main Content area Header */
.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0 40px 0;
}
.header-title {
    font-size: 22px;
    font-weight: 500;
    color: #e3e3e3;
}

/* Hero Section */
.hero-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    margin-top: 60px;
}

.bot-icon {
    width: 60px;
    height: 60px;
    background-color: #1e1f20;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    color: #e3e3e3;
    margin-bottom: 24px;
    border: none;
    box-shadow: none;
    animation: none;
}
.hero-title {
    font-size: 38px;
    font-weight: 500;
    margin-bottom: 16px;
    color: #e3e3e3;
}
.hero-title span {
    background: linear-gradient(74deg, #4285f4 0, #9b72cb 9%, #d96570 20%, #d96570 24%, #9b72cb 35%, #4285f4 44%);
    background-size: 400% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 16px;
    color: #c4c7c5;
    max-width: 600px;
    line-height: 1.5;
    margin-bottom: 50px;
    font-weight: 400;
}

/* Notes container */
.notes-container {
    background: #1e1f20;
    padding: 30px 40px;
    border-radius: 20px;
    border: none;
    margin-bottom: 30px;
    box-shadow: none;
    color: #e3e3e3;
}
.notes-container h1, .notes-container h2, .notes-container h3 {
    color: #4285f4;
    border-bottom: none;
    padding-bottom: 6px;
    margin-bottom: 12px;
    font-weight: 500;
}

/* File Uploader override to look like Gemini inputs */
[data-testid="stFileUploadDropzone"] {
    background-color: #1e1f20 !important;
    border: 1px solid #444746 !important;
    border-radius: 24px !important;
    transition: all 0.2s ease !important;
    color: #e3e3e3 !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: #282a2c !important;
    border-color: #c4c7c5 !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    background-color: #1e1f20 !important;
    border-radius: 24px !important;
    border: 1px solid #444746 !important;
    padding: 0 10px !important;
}
[data-testid="stChatInput"] > div, 
[data-testid="stChatInput"] > div > div, 
[data-testid="stChatInputTextArea"] {
    background-color: transparent !important;
    border: none !important;
}
[data-testid="stChatInputTextArea"] {
    color: #e3e3e3 !important;
}

/* Optional styling for the Assistant avatar specifically if needed */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ----------------- APP LOGIC & STATE -----------------
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'home'
if 'active_note' not in st.session_state:
    st.session_state['active_note'] = None
if 'chat_messages' not in st.session_state:
    st.session_state['chat_messages'] = []

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

history = load_history()

def process_audio(uploaded_file):
    if not client:
        st.error("Please configure GROQ_API_KEY in `.env` or `../.env` file to use Groq API.")
        return None
        
    try:
        with st.spinner("Step 1/2: Transcribing audio directly with Groq Whisper..."):
            audio_bytes = uploaded_file.read()
            
            # 1. Transcribe audio using Whisper
            transcription = client.audio.transcriptions.create(
                file=(uploaded_file.name, audio_bytes),
                model="whisper-large-v3",
                response_format="json"
            )
            transcript_text = transcription.text
                
        with st.spinner("Step 2/2: Generating Structured Notes with LLaMA 3..."):
            
            prompt = f"""
            You are a smart AI Note Generator. I will provide you with a raw lecture transcript.
            Please create beautifully structured notes in the following specific format exactly. Use Markdown.

            Lecture Title: [Generate an appropriate title based on context]
            
            Key Topics:
            - [Topic 1]
            - [Topic 2]
            
            Important Definitions:
            - **Term**: Definition
            
            Detailed Notes (Bullet Points):
            - Include important details from the lecture
            
            Examples Discussed:
            - If any
            
            Summary:
            - A short paragraph summarizing the lecture
            
            Possible Exam Questions:
            1. [Question 1]
            2. [Question 2]
            
            Here is the transcript:
            {transcript_text}
            """

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful study assistant that takes excellent notes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2048
            )
            
            notes = completion.choices[0].message.content
            
            # Combine the two
            combined = f"# TRANSCRIPT\n{transcript_text}\n\n***\n# NOTES\n{notes}"
            return {"transcript": transcript_text, "notes": notes, "content": combined}
            
    except Exception as e:
        st.error(f"Error during audio AI processing: {str(e)}")
        return None

def handle_chat_completion_stream(prompt_text, context_text):
    if not client:
        yield "GROQ API is not configured!"
        return
        
    system_prompt = "You are a helpful study assistant. Answer the user's questions based on their provided context/notes if any."
    if context_text:
        system_prompt += f"\n\nContext/Lecture Transcript to use:\n{context_text}"
        
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state['chat_messages']:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    api_messages.append({"role": "user", "content": prompt_text})
    
    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                # Groq is incredibly fast, so stream yields instantly.
                # Artificial sleep gives the 'real typewriter' aesthetic.
                time.sleep(0.012)
                yield content
    except Exception as e:
        yield f"Error connecting to chat LLM: {str(e)}"

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Notes <span>Pathv</span></div>', unsafe_allow_html=True)
    
    if st.button("⊕ New Lecture Notes", type="primary", use_container_width=True):
        st.session_state['current_view'] = 'home'
        st.session_state['active_note'] = None
        st.session_state['chat_messages'] = []
        st.rerun()
    
    st.markdown('<div class="saved-lectures-label">SAVED LECTURES</div>', unsafe_allow_html=True)
    
    if len(history) == 0:
        st.caption("No history yet. Start a lecture!")
    else:
        for item in reversed(history):
            date_str = item.get('date', '')
            title_str = item.get('title', 'Unknown')
            if len(title_str) > 23:
                title_str = title_str[:20] + "..."
                
            if st.button(f"📄 {title_str}\n{date_str}", key=f"hist_{item['id']}", use_container_width=True):
                st.session_state['active_note'] = item
                st.session_state['current_view'] = 'notes'
                st.session_state['chat_messages'] = [] # clear chat on new lecture load
                st.rerun()
    
    st.markdown("<br>"*5, unsafe_allow_html=True) # Spacer
    
    # User Profile
    st.markdown('''
        <div style="border-top: 1px solid #2d313c; padding-top: 15px; display: flex; align-items: center; gap: 12px; margin-top: 50px;">
            <div style="width: 36px; height: 36px; background-color: #3b82f6; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 16px;">👤</div>
            <div>
                <div style="font-size: 14px; font-weight: 600; color: #f8fafc;">Student</div>
                <div style="font-size: 11px; color: #64748b;">Pro features enabled</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)


# ----------------- MAIN CONTENT -----------------

# col_bg1, col_bg2 = st.columns([1, 1])
# with col_bg1:
#     if st.session_state['current_view'] == 'home':
#         st.markdown('<div class="header-title">New Lecture</div>', unsafe_allow_html=True)
#     else:
#         st.markdown('<div class="header-title">Notes Viewer</div>', unsafe_allow_html=True)
        
# with col_bg2:
#     st.markdown('''
#         <div style="display: flex; justify-content: flex-end;">
#             <div class="powered-by">
#                 ✨ Powered by Groq Whisper & LLaMA 3
#             </div>
#         </div>
#     ''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.session_state['current_view'] == 'home':
    # Hero Section
    st.markdown('''
        <div class="hero-section">
            <div class="bot-icon">🤖</div>
            <div class="hero-title">Lecture → <span>Notes AI</span></div>
            <div class="hero-subtitle">Upload a recorded lecture or record directly from your microphone. I'll automatically generate an English transcript and highly structured notes based on the context.</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Action cards area
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_space, col_c, col_d = st.columns([0.5, 2, 0.1, 2, 0.5])
    
    with col_b:
        uploaded_file = st.file_uploader("📤 **Upload Audio/Video**", type=["mp3", "m4a", "wav", "mp4", "webm"])
        if uploaded_file is not None:
            result = process_audio(uploaded_file)
            if result:
                note_id = str(int(time.time()))
                new_item = {
                    "id": note_id,
                    "title": uploaded_file.name,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "content": result["content"],
                    "raw_transcript": result["transcript"]
                }
                history.append(new_item)
                save_history(history)
                st.session_state['active_note'] = new_item
                st.session_state['current_view'] = 'notes'
                st.rerun()

    with col_c:
        try:
            audio_value = st.audio_input("🎙️ **Record Live Lecture**")
            if audio_value is not None:
                audio_value.name = f"Live_Recording_{datetime.now().strftime('%H%M%S')}.wav"
                result = process_audio(audio_value)
                if result:
                    note_id = str(int(time.time()))
                    new_item = {
                        "id": note_id,
                        "title": audio_value.name,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "content": result["content"],
                        "raw_transcript": result["transcript"]
                    }
                    history.append(new_item)
                    save_history(history)
                    st.session_state['active_note'] = new_item
                    st.session_state['current_view'] = 'notes'
                    st.rerun()
        except AttributeError:
            st.warning("Please upgrade Streamlit to >= 1.38.0 to use the local microphone feature.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render Chat UI on home screen below cards
    col_chat1, col_chat2, col_chat3 = st.columns([1.5, 7, 1.5])
    with col_chat2:
        home_chat_placeholder = st.container()
        with home_chat_placeholder:
            if len(st.session_state['chat_messages']) > 0:
                st.markdown("### 🎓 General Education Assistant")
            for msg in st.session_state['chat_messages']:
                if msg["role"] == "user":
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
                            <div style="background-color: #282a2c; color: #e3e3e3; border-radius: 24px; padding: 12px 20px; max-width: 80%; line-height: 1.5; font-size: 15px;">
                                {msg["content"]}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.chat_message("assistant", avatar="✨").write(msg["content"])
        
        home_stream_placeholder = st.container()

elif st.session_state['current_view'] == 'notes':
    if st.session_state['active_note']:
        with st.container():
            col1, col2, col3 = st.columns([1, 10, 1])
            with col2:
                if st.button("← Back to Upload"):
                    st.session_state['current_view'] = 'home'
                    st.session_state['active_note'] = None
                    st.rerun()
                st.markdown('<div class="notes-container">', unsafe_allow_html=True)
                st.markdown(f"**Archive Date:** {st.session_state['active_note']['date']}")
                st.markdown(st.session_state['active_note']['content'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Render Chat UI above the bottom bar
                chat_placeholder = st.container()
                with chat_placeholder:
                    if len(st.session_state['chat_messages']) > 0:
                        st.markdown("### Context Chat")
                    for msg in st.session_state['chat_messages']:
                        if msg["role"] == "user":
                            st.markdown(f"""
                                <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
                                    <div style="background-color: #282a2c; color: #e3e3e3; border-radius: 24px; padding: 12px 20px; max-width: 80%; line-height: 1.5; font-size: 15px;">
                                        {msg["content"]}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.chat_message("assistant", avatar="✨").write(msg["content"])
                
                stream_placeholder = st.container()
    else:
        stream_placeholder = st.empty()

# ----------------- BOTTOM CHAT BAR -----------------
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
prompt = st.chat_input("Ask about a lecture or ask any educational question...")
if prompt:
    st.session_state['chat_messages'].append({"role": "user", "content": prompt})
    
    ctx = None
    if st.session_state['current_view'] == 'notes' and st.session_state.get('active_note'):
        ctx = st.session_state['active_note'].get("raw_transcript", st.session_state['active_note']['content'])
        
    target_placeholder = stream_placeholder if st.session_state['current_view'] == 'notes' else home_stream_placeholder
    
    # Use the appropriate placeholder for typing effect to render in correct layout
    with target_placeholder:
        st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
                <div style="background-color: #282a2c; color: #e3e3e3; border-radius: 24px; padding: 12px 20px; max-width: 80%; line-height: 1.5; font-size: 15px;">
                    {prompt}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="✨"):
            response_stream = handle_chat_completion_stream(prompt, ctx)
            bot_reply = st.write_stream(response_stream)
    
    st.session_state['chat_messages'].append({"role": "assistant", "content": bot_reply})
    st.rerun()
