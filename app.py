import json
import time
from datetime import datetime

import requests
import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Jenne AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CONFIG
# ==========================================
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"
KEEP_ALIVE = "20m"

SYSTEM_PROMPT = """
You are Jenne AI, a premium local AI model.

Always answer in this exact format:

Relevant content:
- Give a clear and direct explanation in 3 to 6 lines.

Realtime example:
- Give one practical real-world example in 2 to 4 lines.

Suggested content:
- Give 3 short helpful bullet points for next learning or action.

Rules:
- Keep answers concise, clean, and practical.
- Avoid long essays.
- Avoid repeating the question.
- Do not say you are an assistant.
- Refer to yourself as an AI model when needed.
- Make output user-friendly and well structured.
"""

MODEL_OPTIONS = {
    "temperature": 0.4,
    "top_p": 0.9,
    "num_predict": 120,
    "num_ctx": 1024,
}

MAX_HISTORY_MESSAGES = 6

# ==========================================
# SESSION INIT
# ==========================================
def init_session():
    if "chats" not in st.session_state:
        first_chat_id = f"chat_{int(time.time() * 1000)}"
        st.session_state.chats = {
            first_chat_id: {
                "title": "New Chat",
                "messages": [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_response": "",
                "last_response_time": None,
            }
        }
        st.session_state.current_chat = first_chat_id

    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "rename_mode" not in st.session_state:
        st.session_state.rename_mode = False


init_session()

# ==========================================
# HELPERS
# ==========================================
def create_new_chat():
    chat_id = f"chat_{int(time.time() * 1000)}"
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_response": "",
        "last_response_time": None,
    }
    st.session_state.current_chat = chat_id
    st.session_state.stop_generation = False
    st.session_state.rename_mode = False
    st.session_state.pending_prompt = None


def delete_chat(chat_id: str):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]

    if not st.session_state.chats:
        create_new_chat()
    else:
        st.session_state.current_chat = list(st.session_state.chats.keys())[-1]

    st.session_state.stop_generation = False
    st.session_state.rename_mode = False
    st.session_state.pending_prompt = None


def clear_current_chat():
    current_id = st.session_state.current_chat
    st.session_state.chats[current_id]["messages"] = []
    st.session_state.chats[current_id]["title"] = "New Chat"
    st.session_state.chats[current_id]["last_response"] = ""
    st.session_state.chats[current_id]["last_response_time"] = None
    st.session_state.stop_generation = False
    st.session_state.pending_prompt = None


def rename_current_chat(new_title: str):
    current_id = st.session_state.current_chat
    cleaned = " ".join(new_title.strip().split())
    if cleaned:
        st.session_state.chats[current_id]["title"] = cleaned[:40]
    st.session_state.rename_mode = False


def generate_chat_title(user_message: str) -> str:
    cleaned = " ".join(user_message.strip().split())
    if len(cleaned) <= 32:
        return cleaned
    return cleaned[:32].rstrip() + "..."


def export_current_chat() -> str:
    current_chat = st.session_state.chats[st.session_state.current_chat]
    messages = current_chat["messages"]

    if not messages:
        return "No chat to export."

    lines = []
    lines.append("Jenne AI Chat Export")
    lines.append("=" * 60)
    lines.append(f"Title: {current_chat['title']}")
    lines.append(f"Created: {current_chat.get('created_at', '-')}")
    lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for msg in messages:
        role = "You" if msg["role"] == "user" else "Jenne AI"
        lines.append(f"{role}:")
        lines.append(msg["content"])
        lines.append("-" * 60)

    return "\n".join(lines)


def build_ollama_messages(chat_messages: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent = chat_messages[-MAX_HISTORY_MESSAGES:]

    for msg in recent:
        messages.append(
            {
                "role": "assistant" if msg["role"] == "assistant" else "user",
                "content": msg["content"],
            }
        )

    return messages


def stream_ollama_response(chat_messages: list[dict]):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": build_ollama_messages(chat_messages),
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": MODEL_OPTIONS,
    }

    try:
        with requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            stream=True,
            timeout=(10, 180),
        ) as response:
            if response.status_code != 200:
                yield f"Error: Ollama returned status code {response.status_code}"
                return

            full_text = ""

            for line in response.iter_lines():
                if st.session_state.stop_generation:
                    break

                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    full_text += chunk
                    yield full_text

                if data.get("done", False):
                    break

    except requests.exceptions.ConnectionError:
        yield "Error: Could not connect to Ollama. Make sure Ollama is running."
    except requests.exceptions.Timeout:
        yield "Error: Response timed out. Try a smaller prompt or restart Ollama."
    except Exception as exc:
        yield f"Error: {str(exc)}"


# ==========================================
# STYLING
# ==========================================
st.markdown(
    """
    <style>
    :root {
        --text: #f8fafc;
        --muted: #94a3b8;
        --border: rgba(255,255,255,0.08);
        --gold: #f8d66d;
        --gold-soft: rgba(248, 214, 109, 0.16);
        --blue-glow: rgba(96, 165, 250, 0.18);
    }

    .stApp {
        background:
            radial-gradient(circle at top, #1a2747 0%, #0b1327 35%, #04070f 100%);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #0a1020 100%);
        border-right: 1px solid var(--border);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 7rem;
        max-width: 1100px;
    }

    .center-wrap {
        text-align: center;
        margin-bottom: 1.2rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.48rem 1rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        color: #dbeafe;
        font-size: 0.88rem;
        letter-spacing: 0.02em;
        margin-bottom: 0.9rem;
        box-shadow: 0 0 24px rgba(255,255,255,0.03);
    }

    .brand-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.05;
        margin-bottom: 0.35rem;
        background: linear-gradient(90deg, #ffffff 0%, #f8d66d 45%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(255,255,255,0.05);
    }

    .brand-subtitle {
        color: #a5b4fc;
        font-size: 1.02rem;
        font-weight: 500;
        margin-top: 0.15rem;
    }

    .empty-title {
        text-align: center;
        color: white;
        font-size: 2.1rem;
        font-weight: 750;
        margin-top: 5rem;
        margin-bottom: 0.55rem;
        letter-spacing: -0.02em;
    }

    .empty-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        max-width: 760px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }

    .sidebar-head {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 1rem 0 0.5rem 0;
    }

    .footer-note {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 0.45rem;
    }

    .tiny-note {
        color: #cbd5e1;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    .stButton > button,
    div[data-testid="stDownloadButton"] > button {
        width: 100%;
        border-radius: 16px;
        padding: 0.74rem 0.95rem;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.04);
        color: white;
        font-weight: 600;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(255,255,255,0.16);
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.08);
    }

    div[data-testid="stChatInput"] {
        position: fixed;
        left: max(21rem, calc((100vw - 1100px) / 2 + 1rem));
        right: 2rem;
        bottom: 1.25rem;
        z-index: 999;
        background: rgba(2, 6, 23, 0.74);
        backdrop-filter: blur(14px);
        border-radius: 18px;
        padding-top: 0.25rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.28);
    }

    @media (max-width: 1200px) {
        div[data-testid="stChatInput"] {
            left: 21rem;
            right: 1rem;
        }
    }

    @media (max-width: 900px) {
        div[data-testid="stChatInput"] {
            position: static;
            left: auto;
            right: auto;
            bottom: auto;
            background: transparent;
            backdrop-filter: none;
            padding-top: 0;
            box-shadow: none;
        }

        .brand-title {
            font-size: 2.3rem;
        }

        .empty-title {
            font-size: 1.7rem;
            margin-top: 3rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## ✨ Jenne AI")
    st.markdown(
        '<div class="footer-note">Premium local AI model by Vijay Kumar Jenne</div>',
        unsafe_allow_html=True,
    )

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    current_id = st.session_state.current_chat
    current_chat = st.session_state.chats[current_id]

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✏ Rename", use_container_width=True):
            st.session_state.rename_mode = not st.session_state.rename_mode
    with c2:
        if st.button("🗑 Clear", use_container_width=True):
            clear_current_chat()
            st.rerun()

    if st.session_state.rename_mode:
        new_title = st.text_input(
            "Rename chat",
            value=current_chat["title"],
            key="rename_input",
            label_visibility="collapsed",
            placeholder="Enter chat title",
        )
        if st.button("✅ Save Title", use_container_width=True):
            rename_current_chat(new_title)
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-head">RECENT CHATS</div>', unsafe_allow_html=True)

    for chat_id in list(st.session_state.chats.keys())[::-1]:
        chat = st.session_state.chats[chat_id]
        col1, col2 = st.columns([4, 1])

        with col1:
            if st.button(chat["title"], key=f"select_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id
                st.session_state.rename_mode = False
                st.rerun()

        with col2:
            if st.button("✕", key=f"delete_{chat_id}", use_container_width=True):
                delete_chat(chat_id)
                st.rerun()

    st.markdown("---")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇ Export",
            data=export_current_chat(),
            file_name="jenne_ai_chat.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d2:
        copy_text = current_chat.get("last_response", "")
        st.download_button(
            label="📋 Copy Last",
            data=copy_text if copy_text else "No response yet.",
            file_name="last_response.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown(f"**Model:** `{OLLAMA_MODEL}`")
    st.markdown(f"**Keep alive:** `{KEEP_ALIVE}`")

    if current_chat.get("last_response_time") is not None:
        st.markdown(
            f'<div class="tiny-note">Last response time: {current_chat["last_response_time"]:.2f}s</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer-note">Powered locally with Ollama • Fast private inference</div>',
        unsafe_allow_html=True,
    )

# ==========================================
# HEADER
# ==========================================
st.markdown(
    """
    <div class="center-wrap">
        <div class="hero-badge">Premium Local AI Experience</div>
        <div class="brand-title">Jenne AI</div>
        <div class="brand-subtitle">
            Intelligent. Clean. Structured. Practical.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

current_chat = st.session_state.chats[st.session_state.current_chat]
messages = current_chat["messages"]

# ==========================================
# EMPTY STATE
# ==========================================
if not messages:
    st.markdown('<div class="empty-title">How can Jenne AI help today?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="empty-subtitle">A premium local AI model built for clean, structured, and practical answers.</div>',
        unsafe_allow_html=True,
    )

# ==========================================
# DISPLAY CHAT
# ==========================================
for msg in messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        label = "You" if msg["role"] == "user" else "Jenne AI"
        st.markdown(f"**{label}**")
        st.markdown(msg["content"])

# ==========================================
# INPUT
# ==========================================
typed_prompt = st.chat_input("Message Jenne AI...")
final_prompt = typed_prompt or st.session_state.pending_prompt

# ==========================================
# HANDLE PROMPT
# ==========================================
if final_prompt:
    st.session_state.pending_prompt = None

    if current_chat["title"] == "New Chat":
        current_chat["title"] = generate_chat_title(final_prompt)

    current_chat["messages"].append({"role": "user", "content": final_prompt})

    with st.chat_message("user"):
        st.markdown("**You**")
        st.markdown(final_prompt)

    with st.chat_message("assistant"):
        st.markdown("**Jenne AI**")

        stop_col1, stop_col2 = st.columns([1, 6])
        with stop_col1:
            if st.button("⏹ Stop", key="stop_generation_live"):
                st.session_state.stop_generation = True

        response_placeholder = st.empty()
        timer_placeholder = st.empty()

        full_response = ""
        start_time = time.perf_counter()

        for partial in stream_ollama_response(current_chat["messages"]):
            full_response = partial
            elapsed = time.perf_counter() - start_time
            response_placeholder.markdown(full_response + "▌")
            timer_placeholder.caption(f"Generating… {elapsed:.2f}s")

        total_time = time.perf_counter() - start_time
        final_text = full_response.strip()

        if st.session_state.stop_generation:
            final_text = final_text + "\n\n[Generation stopped]" if final_text else "[Generation stopped]"

        response_placeholder.markdown(final_text)
        timer_placeholder.caption(f"Completed in {total_time:.2f}s")

        current_chat["messages"].append({"role": "assistant", "content": final_text})
        current_chat["last_response"] = final_text
        current_chat["last_response_time"] = total_time

        st.session_state.stop_generation = False