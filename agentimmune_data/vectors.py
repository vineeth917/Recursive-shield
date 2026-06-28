from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def embedding_text(payload_text: str, source_transcript_id: str, family: str, delivery: str) -> str:
    return "\n".join(
        [
            f"family: {family}",
            f"delivery: {delivery}",
            f"source_transcript_id: {source_transcript_id}",
            "payload:",
            payload_text,
        ]
    )
