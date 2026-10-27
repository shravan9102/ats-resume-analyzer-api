"""Lightweight drop-in replacement for SentenceTransformer (no PyTorch).

Uses the same all-MiniLM-L6-v2 model through ONNX Runtime (via `fastembed`),
which needs a few hundred MB less RAM than torch + sentence-transformers.
It exposes only the part of the SentenceTransformer API this project uses:
    encode(text_or_list, convert_to_tensor=False, show_progress_bar=False)
"""
import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger("ats_resume_scorer")


class OnnxEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: str | None = None, threads: int = 1):
        from fastembed import TextEmbedding  # imported lazily: only needed in light mode

        if "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"
        kwargs = {"model_name": model_name, "threads": threads}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self._model = TextEmbedding(**kwargs)
        logger.info("ONNX embedder ready: %s", model_name)

    def encode(self, texts: Union[str, List[str]], convert_to_tensor: bool = False,
               show_progress_bar: bool = False, **_ignored) -> np.ndarray:
        single = isinstance(texts, str)
        batch = [texts] if single else [str(t) for t in texts]
        if not batch:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = np.asarray(list(self._model.embed(batch)), dtype=np.float32)
        return vectors[0] if single else vectors
