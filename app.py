import os
import streamlit as st
from pinecone import Pinecone
from openai import OpenAI

# --- ENV ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-chat-demo").strip()

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)   # ✅ host-гүй

# ---------- Embedding ----------
def embed_text(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

# ---------- Query Rewriter ----------
def rewrite_query(question: str) -> str:
    """
    Хэрэглэгчийн асуултыг vector search-д илүү тохиромжтой болгож өөрчилнө.
    """
    prompt = f"""
Чи Ecommerce + Kids clothing catalog дээр хайлт хийх туслах.

Хэрэглэгчийн асуултыг Pinecone vector search-д хамгийн тохиромжтой хайлтын query болгон өөрчил.
- Англи хэлээр бич.
- Боломжтой бол gender (boys/girls/unisex), product type, size/age range, material, season зэргийг таамаглан нэм.
- Хэт урт болгохгүй, 1 өгүүлбэр байхад хангалттай.

Жишээ:
"11 настай хүү" -> "boys clothing size 140 age 10-12"
"охидын даашинз" -> "girls dress cotton size 120-150"
"өвлийн куртик" -> "kids winter jacket warm size 130-150"

Original: {question}
Rewritten:
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        # rewrite алдаа гарвал original-оо ашиглана
        return question

# ---------- Context Cleaner ----------
def clean_context(text: str) -> str:
    """
    Давхардсан мөр/хоосон зай арилгаж, GPT-д цэвэр context өгнө.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    # давхардлыг арилгана
    uniq = []
    seen = set()
    for ln in lines:
        if ln not in seen:
            uniq.append(ln)
            seen.add(ln)
    return "\n".join(uniq)

# ---------- Retrieval (Hybrid) ----------
def retrieve_context(query: str, top_k: int = 5):
    """
    Original query + rewritten query 2-оор хайж context-оо баяжуулна.
    """
    rewritten = rewrite_query(query)

    vectors = []
    for q in [query, rewritten]:
        try:
            q_vec = embed_text(q)
            res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
            for m in res.matches:
                t = m.metadata.get("text", "")
                if t:
                    vectors.append(t)
        except Exception as e:
            st.error(f"Pinecone query error: {e}")
            return ""

    return clean_context("\n\n".join(vectors))

# ---------- Answer Generation ----------
def generate_answer(question: str, context: str) -> str:
    system_prompt = (
        "Чи Монгол хэлтэй ухаалаг туслах. "
        "Доорх CONTEXT бол манай мэдлэгийн сан (CSV + PDF). "
        "Зөвхөн CONTEXT дээр үндэслэж хариул. "
        "CONTEXT хангалтгүй бол 'Мэдээлэл манай мэдлэгийн санд алга байна' гэж хэл. "
        "Хариултаа товч, ойлгомжтой, хэрэгтэй зөвлөгөөтэй өг."
    )

    user_content = f"""
CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1) Эхлээд CONTEXT-с хамаарах хэсгийг ол.
2) Дараа нь хэрэглэгчид ойлгомжтой монголоор хариул.
3) CONTEXT-д байхгүй зүйлийг бүү зохиож хэл.
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    return res.choices[0].message.content

# ---------- UI ----------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 FIVEBABY дэлгүүрийн Chatbot (Smarter RAG")
st.caption("CSV + PDF мэдлэг дээр үндэслэн Bolortsoojin ухаалаг хайлт хийж хариулдаг чат.")

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

    context = retrieve_context(user_input, top_k=6)
    answer = generate_answer(user_input, context)

    st.markdown(
        f"<div class='chat-container'><div class='bot-bubble'>{answer}</div><div class='clearfix'></div></div>",
        unsafe_allow_html=True,
    )
    st.session_state["messages"].append({"role": "assistant", "content": answer})
