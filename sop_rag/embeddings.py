"""
Embedding engine for dense semantic search and similarity calculations.
Utilizes FastEmbed (BAAI/bge-small-en-v1.5) with in-memory caching
and cosine similarity computations.
"""

import numpy as np
from typing import List, Union, Dict


class EmbeddingEngine:
    """Singleton embedding engine for SOPRAG."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        if getattr(self, "_initialized", False):
            return
        self.model_name = model_name
        self._cache: Dict[str, np.ndarray] = {}
        try:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=model_name)
            self._using_fastembed = True
        except Exception as e:
            print(f"[EmbeddingEngine] Warning: FastEmbed initialization failed ({e}), using fallback.")
            self.model = None
            self._using_fastembed = False
        self._initialized = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings, utilizing cache when available."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        uncached_indices = []
        uncached_texts = []
        embeddings = [None] * len(texts)

        for i, text in enumerate(texts):
            clean_text = text.strip()
            if clean_text in self._cache:
                embeddings[i] = self._cache[clean_text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(clean_text)

        if uncached_texts:
            if self._using_fastembed and self.model is not None:
                new_embs = list(self.model.embed(uncached_texts))
                for idx, text, emb in zip(uncached_indices, uncached_texts, new_embs):
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
                    self._cache[text] = emb_norm
                    embeddings[idx] = emb_norm
            else:
                # Deterministic fallback hashing / simple bag-of-words projection
                for idx, text in zip(uncached_indices, uncached_texts):
                    v = np.zeros(384, dtype=np.float32)
                    for word in text.lower().split():
                        h = abs(hash(word)) % 384
                        v[h] += 1.0
                    v_norm = v / (np.linalg.norm(v) + 1e-9)
                    self._cache[text] = v_norm
                    embeddings[idx] = v_norm

        return np.array(embeddings, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two text snippets."""
        emb_a = self.embed_text(text_a)
        emb_b = self.embed_text(text_b)
        return float(np.dot(emb_a, emb_b))

    def batch_similarity(self, query: str, candidates: List[str]) -> List[float]:
        """Compute cosine similarity between a single query and a list of candidates."""
        if not candidates:
            return []
        q_emb = self.embed_text(query)
        cand_embs = self.embed_texts(candidates)
        # Assuming normalized vectors
        sims = np.dot(cand_embs, q_emb)
        # Bound between 0.0 and 1.0 for score stability
        sims = np.clip(sims, 0.0, 1.0)
        return sims.tolist()


def get_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()
