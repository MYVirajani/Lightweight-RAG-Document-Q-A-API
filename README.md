# Lightweight RAG Document Q&A API

A minimal Retrieval-Augmented Generation (RAG) microservice built with **FastAPI**, **ChromaDB**, **sentence-transformers**, and **Groq**. Ingest a PDF document, ask questions about it via a REST endpoint, and get answers grounded strictly in the document — with an explicit refusal when the answer isn't in scope.

## Stack

| Component      | Choice                                 |
|----------------|-----------------------------------------|
| API framework  | FastAPI                                 |
| Vector store   | ChromaDB (persistent, local)            |
| Embeddings     | sentence-transformers (`all-MiniLM-L6-v2`, runs locally, free) |
| LLM            | Groq                                     |
| PDF parsing    | pypdf                                    |

## Project structure

```
rag-doc-qa-api/
├── app/
│   ├── config.py       # env-driven settings
│   ├── embeddings.py   # sentence-transformers loader
│   ├── ingest.py        # PDF parsing + chunking + embedding + storage
│   ├── retriever.py    # semantic search over ChromaDB
│   ├── generator.py    # grounded prompt + Groq call
│   └── main.py          # FastAPI app / POST /ask
├── data/
│   └── sample_document.pdf
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Clone and create a virtual environment

```bash

python -m venv venv
venv/bin/activate   
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

### 4. Ingest a PDF document

Use the included sample, or replace it with any `.pdf` file:

```bash
python -m app.ingest data/sample_document.pdf
```

You should see:
```
Ingested N chunks from 'data/sample_document.pdf' into collection 'documents'.
```

Re-running ingestion on the same file will add duplicate chunks — delete the `chroma_db/` folder first if you want to start fresh.

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Interactive docs: `http://127.0.0.1:8000/docs`. Testing was done against `http://127.0.0.1:8000/health` (basic liveness check) and `http://127.0.0.1:8000/ask` (the main Q&A endpoint).
