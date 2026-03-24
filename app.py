from dotenv import load_dotenv
import os
import tempfile
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory

from db import build_vectorstore

# Load environment variables
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chat", page_icon="📄", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}
.stApp { background-color: #0a0a0f; }

h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtitle {
    font-size: 0.88rem;
    color: #666;
    margin-top: 4px;
    margin-bottom: 24px;
    letter-spacing: 0.05em;
}
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}
.bubble-user {
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    color: #0a0a0f;
    padding: 12px 18px;
    border-radius: 20px 20px 4px 20px;
    max-width: 75%;
    font-size: 0.93rem;
    font-weight: 500;
    line-height: 1.55;
}
.bubble-bot {
    background: #16161f;
    color: #e8e8f0;
    padding: 12px 18px;
    border-radius: 20px 20px 20px 4px;
    max-width: 75%;
    font-size: 0.93rem;
    line-height: 1.55;
    border: 1px solid #2a2a3a;
}

section[data-testid="stFileUploader"] {
    background: #16161f;
    border: 1px dashed #2a2a3a;
    border-radius: 12px;
    padding: 8px 12px;
}
div[data-testid="stFileUploadDropzone"] label {
    color: #666 !important;
    font-size: 0.85rem;
}
.stButton > button {
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    color: #0a0a0f;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 8px 20px;
    cursor: pointer;
}
.stButton > button:hover {
    opacity: 0.88;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>📄 RAG Chat</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Mistral AI · Upload a PDF · Document-grounded answers with memory</p>', unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "ready" not in st.session_state:
    st.session_state.ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None


# ── Cached chain — defined BEFORE the upload block so .clear() can be called ──
@st.cache_resource(show_spinner=False)
def load_chain():
    embeddings = MistralAIEmbeddings(api_key=os.getenv("MISTRAL_API_KEY"))

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
        collection_name="rag_db",
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
        search_type="mmr",
    )

    llm = ChatMistralAI(
        model="mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: \"I could not find the answer in the document.\""""),
        ("human",
         """Chat History:
{history}

Context:
{context}

Question:
{question}
""")
    ])

    return retriever, llm, prompt


# ── PDF upload & indexing ─────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF to chat with", type=["pdf"])

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.indexed_filename:
        with st.spinner(f"Indexing **{uploaded_file.name}** ..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            build_vectorstore(tmp_path)
            os.unlink(tmp_path)

        # Clear cache so load_chain() reloads with the new vectorstore
        load_chain.clear()

        st.session_state.indexed_filename = uploaded_file.name
        st.session_state.ready = True
        st.session_state.chat_history = ChatMessageHistory()
        st.session_state.display_messages = []
        st.success(f"✅ **{uploaded_file.name}** indexed! Start asking questions below.")


# ── Chat UI ───────────────────────────────────────────────────────────────────
if st.session_state.ready:
    retriever, llm, prompt = load_chain()

    # Render existing messages
    for role, content in st.session_state.display_messages:
        if role == "user":
            st.markdown(
                f'<div class="msg-user"><div class="bubble-user">{content}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="msg-bot"><div class="bubble-bot">{content}</div></div>',
                unsafe_allow_html=True,
            )

    query = st.chat_input("Ask something about your document...")

    if query:
        st.markdown(
            f'<div class="msg-user"><div class="bubble-user">{query}</div></div>',
            unsafe_allow_html=True,
        )
        st.session_state.display_messages.append(("user", query))

        with st.spinner("Searching document..."):
            history_text = "\n".join(
                [f"{msg.type}: {msg.content}" for msg in st.session_state.chat_history.messages]
            )

            search_query = query + " " + history_text
            docs = retriever.invoke(search_query)
            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.invoke({
                "history": history_text,
                "context": context,
                "question": query,
            })

            response = llm.invoke(final_prompt)

        st.markdown(
            f'<div class="msg-bot"><div class="bubble-bot">{response.content}</div></div>',
            unsafe_allow_html=True,
        )
        st.session_state.display_messages.append(("ai", response.content))

        st.session_state.chat_history.add_user_message(query)
        st.session_state.chat_history.add_ai_message(response.content)

else:
    st.info("⬆️ Upload a PDF above to get started.")