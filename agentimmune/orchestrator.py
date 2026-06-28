from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentimmune.contracts import EvalRun, ModelVersion, Trace
from agentimmune.eval_harness import EvalMetrics, promotion_allowed, run_eval
from agentimmune.guardrail import StubGuardrail
from agentimmune.store import LocalJsonlStore


@dataclass(frozen=True)
class RoundResult:
    model_version: ModelVersion
    eval_run: EvalRun
    metrics: EvalMetrics


def load_split(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


async def run_round(
    split: dict[str, Any],
    previous_metrics: dict[str, float],
    parent_version_id: str | None = None,
    store: LocalJsonlStore | None = None,
    benign_fp_tolerance: float = 0.02,
) -> RoundResult:
    store = store or LocalJsonlStore()
    split_id = str(split.get("split_id", "split_stub"))
    train_hash = train_set_hash(split.get("train", []))
    model_version = train_stub(train_hash, parent_version_id)

    eval_traces = [Trace.model_validate(item) for item in split.get("held_out", []) + split.get("benign", [])]
    metrics = await run_eval(eval_traces, StubGuardrail(model_version.model_version_id))
    candidate_metrics = {
        "held_out_block_rate": metrics.held_out_block_rate,
        "benign_fp_rate": metrics.benign_fp_rate,
    }
    promoted = promotion_allowed(previous_metrics, candidate_metrics, benign_fp_tolerance)
    reason = (
        "held-out block-rate improved and benign FP stayed within tolerance"
        if promoted
        else "promotion gate rejected: no held-out improvement or benign FP regression"
    )

    model_version = model_version.model_copy(
        update={
            "promoted": promoted,
            "metrics": metrics.as_dict(),
        }
    )
    eval_run = EvalRun(
        model_version_id=model_version.model_version_id,
        split_id=split_id,
        metrics=metrics.as_dict(),
        promoted=promoted,
        promotion_reason=reason,
    )

    store.insert_one("model_versions", model_version)
    store.insert_one("eval_runs", eval_run)
    return RoundResult(model_version=model_version, eval_run=eval_run, metrics=metrics)


def train_stub(train_hash: str, parent_version_id: str | None = None) -> ModelVersion:
    version_suffix = train_hash[:10]
    return ModelVersion(
        model_version_id=f"stub-lora-{version_suffix}",
        parent_version_id=parent_version_id,
        adapter_path=f"models/stub-lora-{version_suffix}",
        train_set_hash=train_hash,
        promoted=False,
        metadata={"trainer": "stub", "real_training": False},
    )


def train_set_hash(train_items: list[Any]) -> str:
    payload = json.dumps(train_items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_round_sync(
    split: dict[str, Any],
    previous_metrics: dict[str, float],
    parent_version_id: str | None = None,
    store: LocalJsonlStore | None = None,
    benign_fp_tolerance: float = 0.02,
) -> RoundResult:
    return asyncio.run(
        run_round(
            split=split,
            previous_metrics=previous_metrics,
            parent_version_id=parent_version_id,
            store=store,
            benign_fp_tolerance=benign_fp_tolerance,
        )
    )
