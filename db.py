from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import shutil

load_dotenv()


def build_vectorstore(pdf_path: str, persist_directory: str = "./chroma_langchain_db"):

    # ── Wipe previous DB so stale docs don't bleed into new session ──
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # ── Load ──────────────────────────────────────────────────────────
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # ── Split ─────────────────────────────────────────────────────────
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = text_splitter.split_documents(docs)

    # ── Embed & store ─────────────────────────────────────────────────
    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_db",          # FIX: named collection
        persist_directory=persist_directory,
    )

    return vector_store