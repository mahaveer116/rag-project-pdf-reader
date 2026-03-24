# RAG Chat – PDF-Based Conversational AI

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions based on their content. The system uses Mistral for response generation and ChromaDB for efficient semantic retrieval.

---

## Overview

This project implements a document-grounded AI assistant that retrieves relevant information from uploaded PDFs and generates accurate, context-aware responses.

Unlike traditional chatbots, this system combines retrieval and generation, improving answer accuracy and reducing hallucinations. ([GitHub][1])

---

## Features

* Upload and chat with PDF documents
* Context-aware question answering
* Retrieval-Augmented Generation (RAG) pipeline
* Semantic search using ChromaDB
* Fast responses using Mistral
* Memory-enabled conversations
* Clean and interactive Streamlit interface

---

## Tech Stack

| Component    | Technology              |
| ------------ | ----------------------- |
| LLM          | Mistral                 |
| Embeddings   | Hugging Face Embeddings |
| Vector Store | ChromaDB                |
| Framework    | LangChain               |
| Backend      | Python                  |
| Frontend     | Streamlit               |

---

## Screenshots

### Chat Interface

Interactive UI for asking questions on uploaded PDFs:

![Interface](https://github.com/mahaveer116/rag-project-pdf-reader/blob/0fe0d9d39557c2d7bfc1ecf13a6f92c5be81d5e4/interface.png)

---

### When Answer is Found

The system retrieves relevant context and generates an answer:

![Answer Found](https://github.com/mahaveer116/rag-project-pdf-reader/blob/0fe0d9d39557c2d7bfc1ecf13a6f92c5be81d5e4/iftextthere.png)

---

### When Answer is Not Found

Graceful fallback when the document lacks relevant information:

![No Answer](https://github.com/mahaveer116/rag-project-pdf-reader/blob/0fe0d9d39557c2d7bfc1ecf13a6f92c5be81d5e4/textnotthere.png)

---

## How It Works



### Pipeline Steps

1. Upload PDF document
2. Extract and split text into chunks
3. Convert text into embeddings
4. Store embeddings in Chroma vector database
5. User asks a query
6. Relevant chunks are retrieved using similarity search
7. Mistral generates a response using retrieved context
8. Answer is displayed in real time

---

## Project Structure

```bash
.
├── app.py
├── requirements.txt
├── vectorstore/
├── utils/
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mahaveer116/rag-project-pdf-reader.git
cd rag-project-pdf-reader
```

---

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Set environment variables

```bash
export HUGGINGFACEHUB_API_TOKEN=your_api_key
```

---

### 5. Run the application

```bash
streamlit run app.py
```

---

## Example Usage

* Upload a PDF (e.g., technical documentation)
* Ask questions like:

  * "What is SQL used for?"
  * "Explain normalization"
* Get answers grounded strictly in the document

---

## Key Highlights

* Reduces hallucinations using document grounding
* Efficient retrieval with ChromaDB
* Scalable for large documents
* Modular and extendable architecture

---

## Limitations

* Depends on document quality
* Cannot answer outside document scope
* Requires API key for model usage

---

## Future Improvements

* Multi-document support
* Source highlighting in answers
* Chat history persistence
* Streaming responses
* Deployment with Docker
* Advanced memory and personalization

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

* Mistral AI
* Hugging Face
* LangChain
* ChromaDB
* Streamlit

---

