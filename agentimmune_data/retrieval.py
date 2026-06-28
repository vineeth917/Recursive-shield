from __future__ import annotations

from typing import Any

from .vectors import cosine_similarity


def similar_attacks_local(
    query_vector: list[float],
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    scored = []
    for candidate in candidates:
        vector = candidate.get("embedding")
        if not isinstance(vector, list):
            continue
        scored.append({**candidate, "score": cosine_similarity(query_vector, vector)})
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def similar_attacks_atlas(db: Any, query_vector: list[float], *, top_k: int = 5) -> list[dict[str, Any]]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": "attack_embedding_vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(50, top_k * 10),
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "attack_id": 1,
                "family": 1,
                "seed": 1,
                "model": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(db.attack_embeddings.aggregate(pipeline))
