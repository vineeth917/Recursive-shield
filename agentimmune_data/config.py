from __future__ import annotations

import os
from dataclasses import dataclass


def _clean_env(value: str | None, default: str = "") -> str:
    return (value if value is not None else default).strip().strip('"').strip("'")


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    mongodb_db: str = "agentimmune"
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-4-large"
    voyage_dimension: int = 1024
    duplicate_threshold: float = 0.92
    split_seed: str = "agentimmune-v1"
    held_out_variant_ratio: float = 0.20
    novel_families: tuple[str, ...] = ("cross_lingual", "sub_audible_over_spoken")


def load_settings() -> Settings:
    novel_raw = os.getenv("NOVEL_HELD_OUT_FAMILIES", "cross_lingual,sub_audible_over_spoken")
    novel = tuple(item.strip() for item in novel_raw.split(",") if item.strip())
    return Settings(
        mongodb_uri=_clean_env(os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")),
        mongodb_db=_clean_env(os.getenv("MONGODB_DB"), "agentimmune"),
        voyage_api_key=_clean_env(os.getenv("VOYAGE_API_KEY")) or None,
        voyage_model=_clean_env(os.getenv("VOYAGE_MODEL"), "voyage-4-large"),
        voyage_dimension=int(_clean_env(os.getenv("VOYAGE_DIMENSION"), "1024")),
        duplicate_threshold=float(_clean_env(os.getenv("ATTACK_DUPLICATE_THRESHOLD"), "0.92")),
        split_seed=_clean_env(os.getenv("SPLIT_SEED"), "agentimmune-v1"),
        held_out_variant_ratio=float(_clean_env(os.getenv("HELD_OUT_VARIANT_RATIO"), "0.20")),
        novel_families=novel,
    )
