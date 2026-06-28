from __future__ import annotations

from typing import Any

from agentimmune.contracts import EvalRun, ModelVersion


def write_eval_run(db: Any, eval_run: EvalRun) -> None:
    db.eval_runs.update_one(
        {"eval_run_id": eval_run.eval_run_id},
        {"$set": eval_run.model_dump(mode="json")},
        upsert=True,
    )


def write_model_version(db: Any, model_version: ModelVersion) -> None:
    db.model_versions.update_one(
        {"model_version_id": model_version.model_version_id},
        {"$set": model_version.model_dump(mode="json")},
        upsert=True,
    )


def list_eval_runs(db: Any) -> list[dict[str, Any]]:
    return list(db.eval_runs.find({}, {"_id": 0}).sort("created_at", 1))
