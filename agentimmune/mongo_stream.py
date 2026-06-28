from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from agentimmune.contracts import GuardrailDecision, ToolAction, Trace
from agentimmune_data.config import load_settings
from agentimmune_data.db import ensure_collections, get_database


ROOT = Path(__file__).resolve().parents[1]


def stream_enabled() -> bool:
    return os.getenv("AGENTIMMUNE_MONGO_STREAM", "").lower() in {"1", "true", "yes", "on"}


def log_run_started(run: Any) -> None:
    if not stream_enabled():
        return
    _safe_write(
        lambda db: db.traces.update_one(
            {"run_id": run.run_id, "record_kind": "run_state"},
            {
                "$set": {
                    "record_kind": "run_state",
                    "run_id": run.run_id,
                    "scenario": run.scenario,
                    "attack_id": run.attack_id,
                    "audio_path": run.audio_path,
                    "transcript": run.transcript_window,
                    "policy": run.policy.model_dump(mode="json"),
                    "guardrail_enabled": run.guardrail_enabled,
                    "actions": [],
                    "decisions": [],
                    "updated_at": _now(),
                },
                "$setOnInsert": {"created_at": _now()},
            },
            upsert=True,
        )
    )


def log_tool_event(
    *,
    run: Any,
    proposed_action: ToolAction,
    blocked: bool,
    decision: GuardrailDecision | None,
    trace: Trace,
) -> None:
    if not stream_enabled():
        return

    event_id = f"event_{uuid4().hex[:12]}"

    def write(db: Any) -> None:
        event = {
            "event_id": event_id,
            "record_kind": "tool_event",
            "run_id": run.run_id,
            "scenario": run.scenario,
            "attack_id": run.attack_id,
            "proposed_action": proposed_action.model_dump(mode="json"),
            "guardrail_decision": decision.model_dump(mode="json") if decision else None,
            "blocked": blocked,
            "executed": not blocked,
            "oracle": trace.oracle.model_dump(mode="json") if trace.oracle else None,
            "created_at": _now(),
        }
        db.traces.update_one(
            {"run_id": trace.run_id, "record_kind": "run_state"},
            {
                "$set": {
                    **trace.model_dump(mode="json"),
                    "record_kind": "run_state",
                    "scenario": run.scenario,
                    "guardrail_enabled": run.guardrail_enabled,
                    "updated_at": _now(),
                },
            },
            upsert=True,
        )
        db.traces.insert_one(event)

    _safe_write(write)


def _safe_write(write_fn: Any) -> None:
    try:
        db = _database()
        write_fn(db)
    except Exception as exc:
        # Logging must never decide whether a brokerage tool executes. The guardrail
        # decision is made before this point; Mongo is evidence memory.
        print(f"mongo_stream_warning={type(exc).__name__}: {exc}")


@lru_cache(maxsize=1)
def _database() -> Any:
    load_dotenv(ROOT / ".env")
    settings = load_settings()
    db = get_database(settings)
    ensure_collections(db)
    return db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
