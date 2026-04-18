import json
import time
from datetime import datetime
from typing import List, Dict

import requests
import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Vijay Jenne AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CONFIG
# ==========================================
HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
HF_API_KEY = st.secrets.get("HF_API_KEY", "")

# Tighter params for deterministic, structured output
HF_PARAMS = {
    "max_new_tokens": 180,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "return_full_text": False,
}

MAX_HISTORY_MESSAGES = 4
REQUEST_TIMEOUT = 45
MAX_RETRIES = 2

SYSTEM_PROMPT = """
You are Vijay Jenne AI, a premium AI model.

STRICT OUTPUT FORMAT (MANDATORY):

Relevant content:
- Explain clearly in 3 to 5 lines.

Realtime example:
- Give exactly one practical real-world example in 2 to 3 lines.

Suggested content:
- Give exactly 3 bullet points.

RULES:
- Do NOT skip any section
- Do NOT add extra sections
- Do NOT change headings
- Keep output concise
- No introductions or conclusions
- No markdown headings (#, ##)
"""

# ==========================================
# SESSION INIT
# ==========================================
def init_session():
    if "chats" not in st.session_state:
        first_chat_id = f"chat_{int(time.time()*1000)}"
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

    for key, default in [
        ("stop_generation", False),
        ("pending_prompt", None),
        ("rename_mode", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


init_session()

# ==========================================
# HELPERS
# ==========================================
def create_new_chat():
    chat_id = f"chat_{int(time.time()*1000)}"
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
    return cleaned if len(cleaned) <= 32 else cleaned[:32].rstrip() + "..."


def export_current_chat() -> str:
    current_chat = st.session_state.chats[st.session_state.current_chat]
    messages = current_chat["messages"]
    if not messages:
        return "No chat to export."
    lines = [
        "Vijay Jenne AI Chat Export",
        "=" * 60,
        f"Title: {current_chat['title']}",
        f"Created: {current_chat.get('created_at', '-')}",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Vijay Jenne AI"
        lines += [f"{role}:", msg["content"], "-" * 60]
    return "\n".join(lines)


def build_prompt(chat_messages: List[Dict]) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nConversation:\n\n"
    for msg in chat_messages[-MAX_HISTORY_MESSAGES:]:
        role = "User" if msg["role"] == "user" else "AI"
        prompt += f"{role}: {msg['content']}\n"
    prompt += "\nAI:"
    return prompt


def hf_generate(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": prompt, "parameters": HF_PARAMS}

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                HF_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data[0].get("generated_text", "").strip()
                return str(data)
            else:
                last_err = f"API error {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    return f"Error: {last_err or 'Unknown error'}"


def enforce_structure(text: str) -> str:
    """Clean + enforce the 3-section format deterministically."""
    t = (text or "").strip()
    t = t.replace("AI:", "").strip()

    # Quick happy path
    if all(h in t for h in ["Relevant content:", "Realtime example:", "Suggested content:"]):
        return t

    # Fallback construction
    body = t.split("\n")
    body = [b.strip() for b in body if b.strip()]
    short = " ".join(body)[:220] if body else "Answer generated."

    return f"""Relevant content:
- {short}

Realtime example:
- Example: Applying this in a real app shows the concept working in practice.

Suggested content:
- Explore related concepts
- Try another query
- Refine your prompt
"""


def safe_check_prompt(p: str) -> bool:
    """Basic guardrail for obviously unsafe requests (extend as needed)."""
    blocked = ["malware", "exploit", "illegal", "harm"]
    return not any(b in p.lower() for b in blocked)


# ==========================================
# STYLING (Premium)
# ==========================================
st.markdown(
    """
<style>
:root {
    --text: #f8fafc;
    --muted: #94a3b8;
    --border: rgba(255,255,255,0.08);
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
.center-wrap { text-align: center; margin-bottom: 1.2rem; }
.hero-badge {
    display: inline-block; padding: 0.48rem 1rem; border-radius: 999px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
    color: #dbeafe; font-size: 0.88rem; margin-bottom: 0.9rem;
}
.brand-title {
    font-size: 3rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.05;
    background: linear-gradient(90deg, #ffffff 0%, #f8d66d 45%, #93c5fd 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.brand-subtitle { color: #a5b4fc; font-size: 1.02rem; }
.empty-title {
    text-align:center; color:white; font-size:2.1rem; font-weight:750;
    margin-top:5rem; margin-bottom:0.55rem;
}
.empty-subtitle {
    text-align:center; color:#94a3b8; font-size:1rem;
    max-width:760px; margin:0 auto 2rem auto;
}
.sidebar-head {
    color:#94a3b8; font-size:0.82rem; font-weight:700;
    letter-spacing:0.08em; margin:1rem 0 0.5rem 0;
}
.footer-note { color:#94a3b8; font-size:0.92rem; margin-top:0.45rem; }
.tiny-note { color:#cbd5e1; font-size:0.85rem; margin-top:0.2rem; }
.stButton > button,
div[data-testid="stDownloadButton"] > button {
    width:100%; border-radius:16px; padding:0.74rem 0.95rem;
    border:1px solid rgba(255,255,255,0.09);
    background: rgba(255,255,255,0.04); color:white; font-weight:600;
}
.stButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.16);
}
div[data-testid="stChatInput"] {
    position: fixed;
    left: max(21rem, calc((100vw - 1100px) / 2 + 1rem));
    right: 2rem; bottom: 1.25rem; z-index: 999;
    background: rgba(2, 6, 23, 0.74);
    backdrop-filter: blur(14px);
    border-radius: 18px; padding-top: 0.25rem;
}
@media (max-width: 900px) {
    div[data-testid="stChatInput"] {
        position: static; background: transparent; backdrop-filter: none;
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
    st.markdown("## ✨ Vijay Jenne AI")
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
            file_name="vijay_jenne_ai_chat.txt",
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
    st.markdown(f"**Model:** `{HF_API_URL.split('/')[-1]}`")
    if current_chat.get("last_response_time") is not None:
        st.markdown(
            f'<div class="tiny-note">Last response time: {current_chat["last_response_time"]:.2f}s</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="footer-note">Powered by Hugging Face • Cloud inference</div>',
        unsafe_allow_html=True,
    )

# ==========================================
# HEADER
# ==========================================
st.markdown(
    """
<div class="center-wrap">
    <div class="hero-badge">Premium AI Experience</div>
    <div class="brand-title">Vijay Jenne AI</div>
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
    st.markdown('<div class="empty-title">How can Vijay Jenne AI help today?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="empty-subtitle">Cloud-ready AI with strict structured outputs.</div>',
        unsafe_allow_html=True,
    )

# ==========================================
# DISPLAY CHAT
# ==========================================
for msg in messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        label = "You" if msg["role"] == "user" else "Vijay Jenne AI"
        st.markdown(f"**{label}**")
        st.markdown(msg["content"])

# ==========================================
# INPUT
# ==========================================
typed_prompt = st.chat_input("Message Vijay Jenne AI...")
final_prompt = typed_prompt or st.session_state.pending_prompt

# ==========================================
# HANDLE PROMPT
# ==========================================
if final_prompt:
    st.session_state.pending_prompt = None

    if not safe_check_prompt(final_prompt):
        with st.chat_message("assistant"):
            st.markdown("**Vijay Jenne AI**")
            st.error("Request blocked by safety filter.")
    else:
        if current_chat["title"] == "New Chat":
            current_chat["title"] = generate_chat_title(final_prompt)

        current_chat["messages"].append({"role": "user", "content": final_prompt})

        with st.chat_message("user"):
            st.markdown("**You**")
            st.markdown(final_prompt)

        with st.chat_message("assistant"):
            st.markdown("**Vijay Jenne AI**")

            stop_col1, _ = st.columns([1, 6])
            with stop_col1:
                if st.button("⏹ Stop", key="stop_generation_live"):
                    st.session_state.stop_generation = True

            response_placeholder = st.empty()
            timer_placeholder = st.empty()

            start = time.perf_counter()

            # Build + call HF
            prompt = build_prompt(current_chat["messages"])
            raw = hf_generate(prompt)

            if st.session_state.stop_generation:
                final_text = "[Generation stopped]"
            else:
                final_text = enforce_structure(raw)

            elapsed = time.perf_counter() - start

            response_placeholder.markdown(final_text)
            timer_placeholder.caption(f"Completed in {elapsed:.2f}s")

            current_chat["messages"].append({"role": "assistant", "content": final_text})
            current_chat["last_response"] = final_text
            current_chat["last_response_time"] = elapsed

            st.session_state.stop_generation = False