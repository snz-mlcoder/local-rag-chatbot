from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "intfloat/e5-small-v2") -> None:
        # English comments only as requested.
        # This model is strong for retrieval tasks and runs on CPU.
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Normalize embeddings for cosine similarity.
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(vectors, np.ndarray):
            return vectors.astype("float32").tolist()
        return [v.astype("float32").tolist() for v in vectors]
