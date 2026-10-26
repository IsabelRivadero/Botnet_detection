from __future__ import annotations

from typing import List, Protocol

import numpy as np


class Embedder(Protocol):

  def encode(self, texts: List[str]) -> np.ndarray:
    ...


class SentenceTransformerEmbedder:
  """Modelo preentrenado y congelado: no aprende de tus datos (sin data leaks)."""

  def __init__(
      self,
      name: str = "paraphrase-multilingual-MiniLM-L12-v2",
      batch_size: int = 256,
  ):
    self.name = name
    self.batch_size = batch_size
    self._model = None

  @property
  def model(self):
    if self._model is None:
      from sentence_transformers import SentenceTransformer  # Import perezoso

      self._model = SentenceTransformer(self.name)
    return self._model

  def encode(self, texts: List[str]) -> np.ndarray:
    if not texts:
      return np.empty((0, 384), dtype=np.float32)

    embeddings = self.model.encode(
        texts,
        batch_size=self.batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


class HashingEmbedder:
  """Embedder determinista y liviano (sin descargas) para pruebas y ejecuciones rápidas."""

  def __init__(self, dim: int = 512):
    from sklearn.feature_extraction.text import HashingVectorizer

    self.dim = dim
    self.vec = HashingVectorizer(
        n_features=dim, alternate_sign=False, norm="l2"
    )

  def encode(self, texts: List[str]) -> np.ndarray:
    if not texts:
      return np.empty((0, self.dim), dtype=np.float32)

    return self.vec.transform(texts).toarray().astype(np.float32)