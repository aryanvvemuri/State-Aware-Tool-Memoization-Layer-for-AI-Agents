"""Embedding generation and vector similarity computation for Axiom."""

from __future__ import annotations
import os
import re
import math
import hashlib
from typing import List, Optional
import numpy as np

# Point to local workspace cache if present for offline execution
_local_hf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache"))
if os.path.exists(_local_hf) and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = _local_hf
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from axiom.config import DEFAULT_EMBEDDING_MODEL


class EmbeddingEngine:
    """Manages sentence embedding model and cosine similarity computations."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("AXIOM_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._model = None
        self._cache = {}

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                # Log or keep None to fall back to deterministic n-gram vectorizer
                self._model = None
        return self._model

    def embed(self, text: str) -> List[float]:
        """Generate normalized embedding vector for the input text."""
        cleaned = text.strip()
        if cleaned in self._cache:
            return self._cache[cleaned]

        model = self._get_model()
        if model is not None:
            try:
                vec = model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
                res = vec.tolist()
                self._cache[cleaned] = res
                return res
            except Exception:
                pass

        # Fallback deterministic pseudo-embedding (384-dim, matching MiniLM-L6-v2)
        res = self._deterministic_fallback_embed(cleaned)
        self._cache[cleaned] = res
        return res

    def _deterministic_fallback_embed(self, text: str) -> List[float]:
        """Deterministic 384-dimensional fallback embedding based on hashed token n-grams.
        Ensures high similarity for rephrased identical keywords and low for unrelated text.
        """
        dim = 384
        vec = np.zeros(dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec.tolist()

        for word in words:
            # Hash single word
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[idx] += sign

        # Hash word pairs (bigrams)
        for i in range(len(words) - 1):
            pair = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.sha256(pair.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[idx] += sign * 1.5

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.tolist()

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vector lists."""
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        dot = float(np.dot(a, b))
        sim = dot / (norm_a * norm_b)
        # Clip to [-1.0, 1.0] to guard against floating point inaccuracies
        return max(-1.0, min(1.0, sim))
