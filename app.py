"""Simple Streamlit RAG chatbot using Pinecone and OpenAI."""

from __future__ import annotations

import os
from typing import List

import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

# ---- Configuration ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-chat-demo")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing")
if not PINECONE_HOST:
    raise ValueError("PINECONE_HOST missing")


@st.cache_resource(show_spinner=False)
def get_clients() -> tuple[OpenAI, Pinecone]:
    """Instantiate and cache API clients."""
    return OpenAI(api_key=OPENAI_API_KEY), Pinecone(api_key=PINECONE_API_KEY)


@st.cache_resource(show_spinner=False)
def get_index():
    """Connect to the Pinecone index once per session."""
    _, pc = get_clients()
    return pc.Index(name=INDEX_NAME, host=PINECONE_HOST)


def embed_text(text: str) -> List[float]:
    """Create an embedding vector for the given text."""
    client, _ = get_clients()
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding


def retrieve_context(query: str, top_k: int = 5) -> str:
    """Return concatenated context passages for the query."""
    q_vec = embed_text(query)
    index = get_index()
    res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)

    contexts: List[str] = []
    for match in res.matches:
        text = match.metadata.get("text", "") if match.metadata else ""
        if text:
            contexts.append(text)
    return "\n\n".join(contexts)


def generate_answer(question: str, context: str) -> str:
    """Generate an answer from the retrieved context using OpenAI."""
    client, _ = get_clients()
    system_prompt = (
        "Чи Монгол хэлтэй туслах. Доорх CONTEXT бол бидний мэдлэгийн сан (CSV + PDF). "
        "Зөвхөн CONTEXT дээр үндэслэж товч, ойлгомжтой хариул. "
        "Хэрвээ CONTEXT-д байхгүй мэдээлэл асуувал 'Мэдээлэл манай мэдлэгийн санд алга байна' гэж хэл."
    )
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    return res.choices[0].message.content


# ---- UI ----
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 RAG Chatbot (Pinecone + OpenAI)")
st.caption("CSV + PDF мэдлэг дээр үндэслэн хариулна.")

bubble_css = """
<style>
.chat-container { width: 100%; margin-top: 8px; }
.user-bubble {
    background-color: #2b2b2b; color: white; padding: 12px 16px;
    border-radius: 16px; border-bottom-right-radius: 4px;
    max-width: 70%; float: right; margin: 6px 0; font-size: 16px;
}
.bot-bubble {
    background-color: #ffe082; color: black; padding: 12px 16px;
    border-radius: 16px; border-bottom-left-radius: 4px;
    max-width: 70%; float: left; margin: 6px 0; font-size: 16px;
}
.clearfix { clear: both; }
</style>
"""
st.markdown(bubble_css, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='chat-container'><div class='user-bubble'>{msg['content']}</div><div class='clearfix'></div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='chat-container'><div class='bot-bubble'>{msg['content']}</div><div class='clearfix'></div></div>",
            unsafe_allow_html=True,
        )

user_input = st.chat_input("Асуух зүйлээ бичээрэй...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.markdown(
        f"<div class='chat-container'><div class='user-bubble'>{user_input}</div><div class='clearfix'></div></div>",
        unsafe_allow_html=True,
    )

    context = retrieve_context(user_input)
    answer = generate_answer(user_input, context)

    st.markdown(
        f"<div class='chat-container'><div class='bot-bubble'>{answer}</div><div class='clearfix'></div></div>",
        unsafe_allow_html=True,
    )
    st.session_state["messages"].append({"role": "assistant", "content": answer})
