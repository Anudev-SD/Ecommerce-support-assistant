"""
Lightweight RAG (Retrieval-Augmented Generation) module.

Design choice for this capstone: rather than pulling in a full vector-database
dependency (Chroma/FAISS/Pinecone), we implement the same *concept* -
chunk -> embed -> cosine-similarity search - with plain numpy. This keeps the
project easy to install and run on a college lab machine, while still
demonstrating the core RAG technique. Swapping in a real vector DB later
would only require changing this file.
"""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from openai import OpenAI

from . import config


@dataclass
class Chunk:
    chunk_id: str
    source: str
    section: str
    text: str


def _split_into_chunks(text: str, source: str) -> list:
    """
    Markdown-aware chunking: split on '##' headings first (keeps each FAQ/policy
    section intact), then further split any section that's still too long.
    """
    chunks = []
    # Split on level-2 headings, keeping the heading with its content.
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading_match = re.match(r"^##\s*(.+)", section)
        heading = heading_match.group(1).strip() if heading_match else "General"

        if len(section) <= config.CHUNK_SIZE_CHARS:
            chunks.append((heading, section))
        else:
            # Further split long sections on Q&A boundaries or paragraph breaks.
            pieces = re.split(r"(?=\*\*Q)", section)
            buf = ""
            for piece in pieces:
                if len(buf) + len(piece) <= config.CHUNK_SIZE_CHARS:
                    buf += piece
                else:
                    if buf.strip():
                        chunks.append((heading, buf.strip()))
                    buf = piece
            if buf.strip():
                chunks.append((heading, buf.strip()))

    result = []
    for i, (heading, content) in enumerate(chunks):
        result.append(
            Chunk(
                chunk_id=f"{source}::{i}",
                source=source,
                section=heading,
                text=content,
            )
        )
    return result


def build_chunks() -> list:
    """Read all knowledge-base markdown files and split them into chunks."""
    all_chunks = []
    for filename in ["faq.md", "policies.md"]:
        path = config.DATA_DIR / filename
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(_split_into_chunks(text, source=filename))
    return all_chunks


class Retriever:
    """Embeds the knowledge base once, caches it to disk, and serves top-k queries."""

    def __init__(self, client: OpenAI = None):
        self.client = client or OpenAI(api_key=config.OPENAI_API_KEY)
        self.chunks: list = []
        self.embeddings: np.ndarray = None
        self._load_or_build_index()

    # -- index persistence ---------------------------------------------
    def _index_paths(self):
        return (
            config.INDEX_DIR / "chunks.json",
            config.INDEX_DIR / "embeddings.npy",
        )

    def _load_or_build_index(self):
        chunks_path, emb_path = self._index_paths()
        if chunks_path.exists() and emb_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.chunks = [Chunk(**c) for c in raw]
            self.embeddings = np.load(emb_path)
        else:
            self.build_index()

    def build_index(self):
        """Embed all knowledge-base chunks and persist the index to disk."""
        self.chunks = build_chunks()
        texts = [c.text for c in self.chunks]
        self.embeddings = self._embed_batch(texts)

        chunks_path, emb_path = self._index_paths()
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, indent=2)
        np.save(emb_path, self.embeddings)

    def _embed_batch(self, texts: list) -> np.ndarray:
        response = self.client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
        vectors = [d.embedding for d in response.data]
        arr = np.array(vectors, dtype=np.float32)
        # Normalize so dot product == cosine similarity.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return arr / norms

    def retrieve(self, query: str, top_k: int = None) -> list:
        """Return the top-k most relevant chunks as dicts with a similarity score."""
        top_k = top_k or config.TOP_K
        query_vec = self._embed_batch([query])[0]
        scores = self.embeddings @ query_vec  # cosine similarity (already normalized)
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < config.MIN_SIMILARITY:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "source": chunk.source,
                    "section": chunk.section,
                    "text": chunk.text,
                    "score": round(score, 4),
                }
            )
        return results


if __name__ == "__main__":
    # Quick manual test: `python -m src.rag`
    r = Retriever()
    for q in ["What is the return window?", "Do you deliver to my pin code?", "refund timeline"]:
        print(f"\nQuery: {q}")
        for res in r.retrieve(q):
            print(f"  [{res['score']}] {res['source']} / {res['section']}")
