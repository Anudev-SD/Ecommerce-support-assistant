"""
Central configuration for the E-Commerce Support Assistant.

Reads settings from environment variables (loaded from a .env file if present).
Copy .env.example to .env and fill in your OPENAI_API_KEY before running anything.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if it exists.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- OpenAI settings -------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Chat model used for the assistant's reasoning + tool calling.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Embedding model used to build the RAG vector index.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- RAG settings ------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_DIR.mkdir(exist_ok=True)

CHUNK_SIZE_CHARS = 300       # approx characters per chunk
CHUNK_OVERLAP_CHARS = 120    # overlap between consecutive chunks
TOP_K = 3                    # how many chunks to retrieve per query
MIN_SIMILARITY = 0.20        # below this, we treat retrieval as "no good match"

# --- Agent settings ------------------------------------------------------
MAX_TOOL_HOPS = 5            # safety limit on tool-calling loop iterations
ESCALATION_PHRASE = "I'm connecting you with a human support agent"

# --- Runtime data (mock "database") ---------------------------------------
ORDERS_FILE = DATA_DIR / "mock_orders.json"
PRODUCTS_FILE = DATA_DIR / "products.json"
RETURNS_STORE_FILE = DATA_DIR / "returns_store.json"
