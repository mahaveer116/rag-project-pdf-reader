from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory  # FIX: correct import

# Load environment variables
load_dotenv()

# ------------------ Embeddings ------------------
embeddings = MistralAIEmbeddings(
    api_key=os.getenv("MISTRAL_API_KEY"),
)

# ------------------ Vector Store ------------------
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
    collection_name="rag_db",              # FIX: must match build_vectorstore()
)

# ------------------ Retriever ------------------
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
    search_type="mmr",
)

# ------------------ LLM ------------------
llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
)

# ------------------ Prompt ------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
    ),
    (
        "human",
        """Chat History:
{history}

Context:
{context}

Question:
{question}
"""
    )
])

# ------------------ Chat History ------------------
chat_history = ChatMessageHistory()

print("RAG system with memory ready.\n")
print("Press 0 to exit\n")

# ------------------ Chat Loop ------------------
while True:
    query = input("You: ")

    if query == "0":
        break

    # Format history
    history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in chat_history.messages]
    )

    # Improve retrieval using history
    search_query = query + " " + history_text

    # Retrieve documents
    docs = retriever.invoke(search_query)

    # Build context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create final prompt
    final_prompt = prompt.invoke({
        "history": history_text,
        "context": context,
        "question": query
    })

    # Get response
    response = llm.invoke(final_prompt)

    # Print response
    print(f"\nAI: {response.content}\n")

    # Save conversation
    chat_history.add_user_message(query)
    chat_history.add_ai_message(response.content)