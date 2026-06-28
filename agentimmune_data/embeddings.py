from __future__ import annotations

from collections.abc import Iterable

from .config import Settings


class VoyageEmbedder:
    def __init__(self, settings: Settings):
        if not settings.voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY is required for live embeddings")
        try:
            import voyageai
        except ImportError as exc:
            raise RuntimeError("Install voyageai to use live embeddings") from exc

        self._client = voyageai.Client(api_key=settings.voyage_api_key)
        self._model = settings.voyage_model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        batch = list(texts)
        if not batch:
            return []
        result = self._client.embed(batch, model=self._model, input_type="document")
        return [list(vector) for vector in result.embeddings]
