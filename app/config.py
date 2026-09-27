"""
Central configuration for the RAG Document Q&A API.
All values are loaded from environment variables (see .env.example),
with sensible defaults so the app still runs if a var is missing.
"""
import os
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")


TOP_K = int(os.getenv("TOP_K", 3))

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
