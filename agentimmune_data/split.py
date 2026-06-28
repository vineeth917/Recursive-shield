from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentimmune.contracts import AttackSpec
from .vectors import cosine_similarity


@dataclass(frozen=True)
class SplitConfig:
    seed: str
    novel_families: tuple[str, ...]
    dev_ratio: float = 0.15
    held_out_variant_ratio: float = 0.20


def split_for_attack(spec: AttackSpec, config: SplitConfig) -> str:
    if spec.family in config.novel_families:
        return "novel_held_out"
    digest = hashlib.sha256(f"{config.seed}:{spec.family}:{spec.seed}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < config.held_out_variant_ratio:
        return "held_out"
    if value < config.held_out_variant_ratio + config.dev_ratio:
        return "dev"
    return "train"


def build_split(specs: Iterable[AttackSpec], config: SplitConfig) -> dict[str, list[str]]:
    split: dict[str, list[str]] = {"train": [], "dev": [], "held_out": [], "benign": [], "novel_held_out": []}
    for spec in sorted(specs, key=lambda item: item.attack_id):
        split[split_for_attack(spec, config)].append(spec.attack_id)
    return split


def write_split_json(split: Mapping[str, list[str]], path: Path) -> None:
    path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_no_leakage(
    specs: Iterable[AttackSpec],
    split: Mapping[str, list[str]],
    embeddings: Mapping[str, list[float]] | None = None,
    *,
    duplicate_threshold: float = 0.92,
) -> None:
    by_id = {spec.attack_id: spec for spec in specs}
    train = set(split.get("train", []))
    held_out = set(split.get("held_out", []))
    novel_held_out = set(split.get("novel_held_out", []))
    eval_holdout = held_out | novel_held_out

    overlap = train & eval_holdout
    if overlap:
        raise AssertionError(f"attack_id leakage across train/holdout: {sorted(overlap)}")

    train_family_seed = {(by_id[item].family, by_id[item].seed) for item in train if item in by_id}
    held_family_seed = {(by_id[item].family, by_id[item].seed) for item in eval_holdout if item in by_id}
    seed_overlap = train_family_seed & held_family_seed
    if seed_overlap:
        raise AssertionError(f"(family, seed) leakage across train/holdout: {sorted(seed_overlap)}")

    train_families = {by_id[item].family for item in train if item in by_id}
    novel_families = {by_id[item].family for item in novel_held_out if item in by_id}
    family_overlap = train_families & novel_families
    if family_overlap:
        raise AssertionError(f"novel held-out family appears in train: {sorted(family_overlap)}")

    if embeddings:
        for train_id in sorted(train):
            for held_id in sorted(eval_holdout):
                if train_id not in embeddings or held_id not in embeddings:
                    continue
                score = cosine_similarity(embeddings[train_id], embeddings[held_id])
                if score > duplicate_threshold:
                    raise AssertionError(
                        f"near-duplicate leakage: {train_id} vs {held_id} cosine={score:.4f}"
                    )


def split_summary(split: Mapping[str, list[str]]) -> dict[str, Any]:
    return {name: len(ids) for name, ids in split.items()}
