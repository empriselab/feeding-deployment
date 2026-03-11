from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

def _query_text(context: Dict[str, Any], corrected: Dict[str, str]) -> str:
    ctx = (
        f"meal={context.get('meal')}; setting={context.get('setting')}; "
        f"time_of_day={context.get('time_of_day')};"
    )
    corr = "; ".join(f"{k}={v}" for k, v in sorted(corrected.items())) if corrected else "none"
    return f"{ctx}\ncorrected_so_far: {corr}"

def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: Dict[str, List[float]] = {}
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, list):
                            self._cache[k] = [float(x) for x in v]
            except Exception:
                self._cache = {}

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        return self._cache.get(self._key(text))

    def set(self, text: str, emb: List[float]) -> None:
        self._cache[self._key(text)] = emb

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)


class EpisodicMemoryModel:
    def __init__(
        self,
        client: OpenAI,
        embed_model: str,
        cache_path: Path,
        retry_fn,
        k_retrieve: int = 5,
    ) -> None:
        self.client = client
        self.embed_model = embed_model
        self.cache = EmbeddingCache(cache_path)
        self._retry = retry_fn
        self.k_retrieve = k_retrieve
        self._history_texts: List[str] = []
        self._last_retrieved: List[str] = []
        
    def add_episode(self, episode_text: str) -> None:
        # Just cache the embedding for this episode text. Retrieval will happen later.
        self._history_texts.append(episode_text)

    def _embed(self, text: str) -> List[float]:
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        def _call() -> Any:
            return self.client.embeddings.create(model=self.embed_model, input=text)

        resp = self._retry(_call)
        emb = [float(x) for x in resp.data[0].embedding]
        self.cache.set(text, emb)
        return emb

    def retrieve(self, context: Dict[str, Any], corrected: Dict[str, str]) -> List[str]:
        query = _query_text(context, corrected)
        if not self._history_texts or self.k_retrieve <= 0:
            return []
        q_emb = self._embed(query)
        scored: List[Tuple[float, str]] = []
        for txt in self._history_texts:
            e_emb = self._embed(txt)
            scored.append((_cosine_sim(q_emb, e_emb), txt))
        scored.sort(key=lambda t: t[0], reverse=True)
        retrieved = [t[1] for t in scored[:self.k_retrieve]]
        retrieved = "\n\n".join(retrieved) if retrieved else ""
        self._last_retrieved = retrieved 
        return retrieved
        
    def get_last_retrieved(self) -> List[str]:
        return self._last_retrieved

    def reset(self) -> None:
        self._history_texts = []
        self.cache.flush()
        