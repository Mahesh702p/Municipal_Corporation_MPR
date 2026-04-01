"""
retriever.py
============
FAISS-based retrieval for the Municipal Corporation RAG pipeline.

Uses TF-IDF + keyword scoring for fast, lightweight retrieval
(no heavy sentence-transformers needed on CPU).

For production: swap TF-IDF with sentence-transformers/mUSE embeddings.

Usage:
  from rag.retriever import MunicipalRetriever
  ret = MunicipalRetriever()
  ret.build_index("data/rag/faq_chunks.jsonl")
  ret.save("artifacts/rag_index")

  # At inference time:
  ret = MunicipalRetriever.load("artifacts/rag_index")
  results = ret.retrieve("birth certificate ke liye kya chahiye", top_k=3)
"""

import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

# TF-IDF from sklearn — no heavy deps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MunicipalRetriever:
    """
    Lightweight FAQ retriever using TF-IDF cosine similarity + keyword boost.

    Why TF-IDF instead of dense embeddings for now:
    - No GPU needed, runs on any machine
    - Fast index build (< 1 sec for 100 chunks)
    - Accurate enough for domain-specific FAQ retrieval
    - Easy to swap for sentence-transformers later
    """

    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None

    # ──────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────
    def _preprocess(self, text: str) -> str:
        """Basic normalization for retrieval."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _text_for_indexing(self, chunk: dict) -> str:
        """Combine question + answer + keywords for indexing."""
        parts = [
            chunk.get("question", ""),
            chunk.get("answer", ""),
        ]
        keywords = chunk.get("keywords", [])
        if keywords:
            parts.append(" ".join(keywords))
        return self._preprocess(" ".join(parts))

    # ──────────────────────────────────────
    # Build index
    # ──────────────────────────────────────
    def build_index(self, chunks_path: str):
        """Load FAQ and/or PDF chunks and build TF-IDF index."""
        self.chunks = []
        
        # If directory, read all .jsonl files in it. Otherwise, read the single file.
        if os.path.isdir(chunks_path):
            files_to_read = [os.path.join(chunks_path, f) for f in os.listdir(chunks_path) if f.endswith('.jsonl')]
        else:
            files_to_read = [chunks_path]
            
        for file_path in files_to_read:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.chunks.append(json.loads(line))

        print(f"[RAG] Building index on {len(self.chunks)} chunks from {len(files_to_read)} file(s)...")

        corpus = [self._text_for_indexing(c) for c in self.chunks]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # unigram + bigram
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,  # log normalization
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)  # sparse (n_chunks, vocab)
        print(f"[RAG] Index built. Vocab size: {len(self.vectorizer.vocabulary_):,}")

    # ──────────────────────────────────────
    # Retrieve
    # ──────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 3, department_filter: str = None) -> list[dict]:
        """
        Retrieve top-k most relevant FAQ chunks for a query.

        Args:
            query: User's natural language query.
            top_k: Number of results to return.
            department_filter: If given, only return chunks from this department.

        Returns:
            List of dicts with chunk info + relevance score.
        """
        if self.vectorizer is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        query_vec = self.vectorizer.transform([self._preprocess(query)])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]  # (n_chunks,)

        # Keyword boost: if any keyword appears in query, boost that chunk's score
        query_lower = query.lower()
        for i, chunk in enumerate(self.chunks):
            for kw in chunk.get("keywords", []):
                if kw.lower() in query_lower:
                    scores[i] += 0.2  # boost

        # Department filter
        if department_filter:
            for i, chunk in enumerate(self.chunks):
                if chunk.get("department", "").lower() != department_filter.lower():
                    scores[i] *= 0.3  # penalize wrong dept chunks

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:  # minimum relevance threshold
                results.append({
                    "chunk_id": self.chunks[idx]["chunk_id"],
                    "department": self.chunks[idx]["department"],
                    "question": self.chunks[idx]["question"],
                    "answer": self.chunks[idx]["answer"],
                    "score": float(scores[idx]),
                })

        return results

    # ──────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────
    def save(self, path: str):
        """Save retriever to directory."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "retriever.pkl"), "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
            }, f)
        print(f"[RAG] Retriever saved → {path}/")

    @classmethod
    def load(cls, path: str) -> "MunicipalRetriever":
        """Load retriever from directory."""
        instance = cls()
        with open(os.path.join(path, "retriever.pkl"), "rb") as f:
            data = pickle.load(f)
        instance.chunks = data["chunks"]
        instance.vectorizer = data["vectorizer"]
        instance.tfidf_matrix = data["tfidf_matrix"]
        print(f"[RAG] Retriever loaded from {path}/ ({len(instance.chunks)} chunks)")
        return instance


if __name__ == "__main__":
    # Build and test
    ret = MunicipalRetriever()
    ret.build_index("data/rag")
    ret.save("artifacts/rag_index")

    test_queries = [
        "birth certificate ke liye kya documents chahiye",
        "property tax kaise calculate hota hai",
        "naya water connection chahiye",
        "garbage collection timing kya hai",
        "emergency contact number kya hai",
        "building permission kaise milegi",
    ]

    print("\n── Retrieval Test ──────────────────────────────────")
    for q in test_queries:
        results = ret.retrieve(q, top_k=1)
        if results:
            print(f"Q: {q}")
            print(f"   → {results[0]['question']} (score={results[0]['score']:.3f})")
        print()
