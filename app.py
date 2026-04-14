import streamlit as st
import tempfile
import os
import re
import torch
from faster_whisper import WhisperModel
from openai import OpenAI
from audiorecorder import audiorecorder
import yt_dlp
from pydub import AudioSegment

# ==========================================
# FFMPEG PATH — set explicitly so pydub & yt-dlp always find it
# ==========================================
FFMPEG_BIN = r"C:\Users\Anish\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
FFMPEG_EXE = os.path.join(FFMPEG_BIN, "ffmpeg.exe")

# Tell pydub where ffmpeg lives
AudioSegment.converter  = FFMPEG_EXE
AudioSegment.ffmpeg     = FFMPEG_EXE
AudioSegment.ffprobe    = os.path.join(FFMPEG_BIN, "ffprobe.exe")

# Add ffmpeg to PATH so yt-dlp subprocess can find it too
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Educational Assistant",
    # page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS — ChatGPT-like Dark UI
# ==========================================
st.markdown("""
<style>
    body, .stApp { background-color: #111111; }
    /* Style Streamlit's native chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.6rem;
    }
    /* User message — blue left border */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #2b313e;
        border-left: 4px solid #4fa3e0;
    }
    /* Assistant message — purple left border */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #1e2430;
        border-left: 4px solid #6c63ff;
    }
    .status-ok  { color: #4caf50; font-weight: bold; }
    .status-err { color: #f44336; font-weight: bold; }
    /* Proper list spacing inside chat */
    [data-testid="stChatMessage"] ul  { padding-left: 1.4rem; margin-top: 0.3rem; }
    [data-testid="stChatMessage"] ol  { padding-left: 1.4rem; margin-top: 0.3rem; }
    [data-testid="stChatMessage"] li  { margin-bottom: 0.25rem; }
    [data-testid="stChatMessage"] h1  { font-size: 1.4rem; margin-top: 0.5rem; }
    [data-testid="stChatMessage"] h2  { font-size: 1.15rem; margin-top: 0.8rem; color: #a78bfa; }
    [data-testid="stChatMessage"] h3  { font-size: 1rem; margin-top: 0.6rem; }
    [data-testid="stChatMessage"] p   { margin-bottom: 0.4rem; line-height: 1.6; }
    [data-testid="stChatMessage"] code { background: #2d2d2d; padding: 2px 6px; border-radius: 4px; }
    [data-testid="stChatMessage"] pre  { background: #1a1a2e; padding: 0.8rem; border-radius: 8px; overflow-x: auto; }
    [data-testid="stChatMessage"] blockquote { border-left: 3px solid #6c63ff; padding-left: 0.8rem; color: #aaa; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS
# ==========================================
STT_MODEL_SIZE  = "small"
LM_STUDIO_URL   = "http://localhost:1234/v1"
LM_STUDIO_KEY   = "lm-studio"          # LM Studio ignores this but the library needs it
LM_MODEL_NAME   = "local-model"        # LM Studio routes this to whatever model is loaded

# ==========================================
# LM STUDIO CLIENT (lightweight — no VRAM)
# ==========================================
@st.cache_resource(show_spinner=False)
def get_lm_client():
    return OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_KEY)

def lm_studio_online() -> bool:
    """Ping LM Studio to check if the server is live."""
    try:
        client = get_lm_client()
        client.models.list()
        return True
    except Exception:
        return False

# ==========================================
# FASTER-WHISPER STT MODEL (GPU, fp16)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_stt_model():
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(STT_MODEL_SIZE, device=device, compute_type=compute_type)

# ==========================================
# CORE FUNCTIONS
# ==========================================
def is_youtube_url(text: str) -> bool:
    """Return True if text looks like a YouTube URL."""
    pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"
    return bool(re.search(pattern, text.strip()))

def download_youtube_audio(url: str) -> str:
    """Download audio from a YouTube URL using yt-dlp and return the file path."""
    tmp_dir = tempfile.mkdtemp()
    output_template = os.path.join(tmp_dir, "yt_audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "ffmpeg_location": FFMPEG_BIN,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "YouTube Lecture")
    wav_path = os.path.join(tmp_dir, "yt_audio.wav")
    return wav_path, title

def transcribe_audio(file_path: str, stt_model) -> str:
    segments, _ = stt_model.transcribe(file_path, beam_size=5)
    return " ".join(seg.text for seg in segments).strip()

def generate_notes(transcript: str):
    """Stream structured notes token-by-token from LM Studio."""
    client = get_lm_client()
    prompt = f"""Convert the following lecture transcript into structured educational notes using Markdown.
You MUST include all five sections with these exact headings:

# Title
## Key Points
## Explanation
## Examples
## Summary

Transcript:
{transcript[:3000]}
"""
    stream = client.chat.completions.create(
        model=LM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert educational assistant that creates clear, structured study notes."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024,
        stream=True,   # ← Enable streaming
    )
    # Yield each token as it arrives
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

def chat_with_model(messages: list):
    """Stream chat reply token-by-token from LM Studio."""
    client = get_lm_client()
    system_msg = {"role": "system", "content": "You are a helpful educational AI assistant. Answer questions clearly and concisely using proper markdown formatting with headings, bullet points, and bold text where appropriate."}
    stream = client.chat.completions.create(
        model=LM_MODEL_NAME,
        messages=[system_msg] + messages[-10:],
        temperature=0.7,
        max_tokens=512,
        stream=True,   # ← Enable streaming
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

# ==========================================
# SESSION STATE
# ==========================================
def init_session_state():
    if "messages"        not in st.session_state: st.session_state.messages        = []
    if "latest_notes"    not in st.session_state: st.session_state.latest_notes    = ""
    if "yt_audio_path"   not in st.session_state: st.session_state.yt_audio_path   = None
    if "yt_title"        not in st.session_state: st.session_state.yt_title        = ""

# ==========================================
# MAIN APP
# ==========================================
def main():
    init_session_state()

    st.title("🎙️ AI Educational Assistant")
    st.markdown("Convert lectures into structured notes — powered by **your own custom model** running in LM Studio.")

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("⚙️ System Status")

        # STT Model
        with st.spinner("Loading Whisper STT..."):
            stt_model = load_stt_model()
        st.markdown('<span class="status-ok">✅ Whisper STT loaded (GPU)</span>', unsafe_allow_html=True)

        # LM Studio
        lm_ok = lm_studio_online()
        if lm_ok:
            st.markdown('<span class="status-ok">✅ LM Studio server connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">❌ LM Studio server offline</span>', unsafe_allow_html=True)
            st.warning("👉 Open LM Studio → Local Server tab → Load your model → Click **Start Server**")

        st.markdown("---")
        st.markdown("### 🧠 Model Info")
        st.markdown(f"- **STT**: Whisper `{STT_MODEL_SIZE}` (float16)")
        st.markdown(f"- **LLM**: Your custom `my_custom_model.Q4_K_M.gguf`")
        st.markdown(f"- **Server**: `{LM_STUDIO_URL}`")

    # Stop here if LM Studio is offline
    if not lm_studio_online():
        st.error("⚠️ LM Studio server is not running. Please start it to use this app.")
        st.stop()

    # ---- YOUTUBE INPUT ----
    st.markdown("---")
    st.subheader("📺 YouTube Lecture URL")
    yt_col, yt_btn_col = st.columns([5, 1])
    with yt_col:
        yt_url = st.text_input(
            "Paste a YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="yt_url_input"
        )
    with yt_btn_col:
        fetch_yt = st.button("⬇️ Fetch", type="primary", key="fetch_yt_btn")

    if fetch_yt and yt_url:
        if not is_youtube_url(yt_url):
            st.error("❌ That doesn't look like a valid YouTube URL. Please try again.")
        else:
            with st.spinner("⏳ Downloading audio from YouTube... (this may take a minute)"):
                try:
                    wav_path, vid_title = download_youtube_audio(yt_url)
                    st.session_state.yt_audio_path = wav_path
                    st.session_state.yt_title = vid_title
                    st.success(f"✅ Downloaded: **{vid_title}**")
                except Exception as e:
                    st.error(f"❌ Failed to download video: {e}")
                    st.session_state.yt_audio_path = None

    if st.session_state.yt_audio_path and os.path.exists(st.session_state.yt_audio_path):
        yt_process = st.button("🚀 Transcribe & Generate Notes from YouTube", type="primary", key="yt_process_btn")
        if yt_process:
            with st.spinner("Transcribing YouTube audio with Whisper..."):
                transcript = transcribe_audio(st.session_state.yt_audio_path, stt_model)
            try:
                os.remove(st.session_state.yt_audio_path)
                st.session_state.yt_audio_path = None
            except Exception:
                pass

            with st.expander(f"📝 Raw Transcript — {st.session_state.yt_title}", expanded=False):
                st.write(transcript)

            if transcript.strip():
                with st.chat_message("assistant"):
                    with st.spinner("Generating notes..."):
                        notes = st.write_stream(generate_notes(transcript))
                st.session_state.latest_notes = notes
                st.session_state.messages.append({"role": "user",     "content": f"[YouTube: {st.session_state.yt_title}] {transcript[:200]}..."})
                st.session_state.messages.append({"role": "assistant", "content": notes})

    # ---- AUDIO INPUT ----
    st.markdown("---")
    col1, col2 = st.columns(2)
    audio_path = None

    with col1:
        st.subheader("🎵 Upload Audio")
        uploaded_file = st.file_uploader("MP3 / WAV / M4A", type=["mp3", "wav", "m4a"])
        if uploaded_file:
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                audio_path = tmp.name
            st.audio(uploaded_file)

    with col2:
        st.subheader("🎙️ Record Audio")
        recording = audiorecorder("Click to Record", "Click to Stop")
        if len(recording) > 0:
            st.audio(recording.export().read())
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                recording.export(tmp.name, format="wav")
                audio_path = tmp.name

    # ---- PROCESS BUTTON ----
    if audio_path and st.button("🚀 Transcribe & Generate Notes", type="primary"):
        with st.spinner("Transcribing audio with Whisper..."):
            transcript = transcribe_audio(audio_path, stt_model)
        try:
            os.remove(audio_path)
        except Exception:
            pass

        with st.expander("📝 Raw Transcript", expanded=False):
            st.write(transcript)

        if transcript.strip():
            with st.chat_message("assistant"):
                with st.spinner("Generating notes..."):
                    # write_stream() feeds tokens one-by-one → live typing effect
                    notes = st.write_stream(generate_notes(transcript))
            st.session_state.latest_notes = notes
            st.session_state.messages.append({"role": "user",      "content": f"[Audio Transcript]: {transcript[:200]}..."})
            st.session_state.messages.append({"role": "assistant",  "content": notes})

    # ---- NOTES OUTPUT ----
    if st.session_state.latest_notes:
        st.markdown("---")
        st.header("✨ Generated Notes")
        st.markdown(st.session_state.latest_notes)
        st.download_button(
            "⬇️ Download Notes (.md)",
            data=st.session_state.latest_notes,
            file_name="structured_notes.md",
            mime="text/markdown"
        )

    # ---- CHAT ----
    st.markdown("---")
    st.subheader("💬 Chat with your Model")
    st.caption("Ask questions about your notes or any educational topic.")

    # Render each message using Streamlit's native chat_message component
    # This correctly renders markdown: headings, bullet lists, bold, code, etc.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question...")
    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Stream assistant reply token-by-token → typing effect
        with st.chat_message("assistant"):
            reply = st.write_stream(chat_with_model(st.session_state.messages))
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

if __name__ == "__main__":
    main()
