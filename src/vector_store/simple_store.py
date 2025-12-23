from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SearchHit:
    score: float
    text: str
    metadata: Dict[str, Any]


class SimpleVectorStore:
    """
    A tiny local vector store:
    - Stores embeddings in a .npz file (NumPy)
    - Uses cosine similarity (dot product on normalized vectors)
    """

    def __init__(self, persist_path: str = "data/vectors/store.npz") -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self._embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._texts: List[str] = []
        self._metas: List[Dict[str, Any]] = []

        if self.persist_path.exists():
            self._load()

    def add(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        vecs = np.array(embeddings, dtype=np.float32)
        if self._embeddings.size == 0:
            self._embeddings = vecs
        else:
            self._embeddings = np.vstack([self._embeddings, vecs])

        self._texts.extend(texts)
        self._metas.extend(metadatas)
        self._save()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchHit]:
        if self._embeddings.size == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        # Since embeddings are normalized, cosine similarity = dot product.
        scores = self._embeddings @ q
        top_idx = np.argsort(scores)[::-1][:top_k]

        hits: List[SearchHit] = []
        for idx in top_idx:
            hits.append(
                SearchHit(
                    score=float(scores[idx]),
                    text=self._texts[int(idx)],
                    metadata=self._metas[int(idx)],
                )
            )
        return hits

    def _save(self) -> None:
        np.savez_compressed(
            self.persist_path,
            embeddings=self._embeddings,
            texts=np.array(self._texts, dtype=object),
            metas=np.array(self._metas, dtype=object),
        )

    def _load(self) -> None:
        data = np.load(self.persist_path, allow_pickle=True)
        self._embeddings = data["embeddings"].astype(np.float32)
        self._texts = data["texts"].tolist()
        self._metas = data["metas"].tolist()
