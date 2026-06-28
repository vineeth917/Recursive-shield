from __future__ import annotations

import asyncio
from pathlib import Path

from agentimmune.contracts import GuardrailHookPayload, OracleVerdict, Trace
from agentimmune.guardrail import classify_payload
from agentimmune.oracle import evaluate_trace


ROOT = Path(__file__).resolve().parents[1]
HANDOFF_DIR = ROOT / "fixtures" / "task_a_handoff"


def load_trace(name: str) -> Trace:
    return Trace.model_validate_json((HANDOFF_DIR / name).read_text())


def test_task_a_clean_fed_trace_is_safe() -> None:
    trace = load_trace("clean_fed_trace.json")

    label = evaluate_trace(trace)

    assert label.label == OracleVerdict.SAFE
    assert label.forbidden_tool_fired is False
    assert label.induced_by_injection is False


def test_task_a_l1_poisoned_trace_is_unsafe_and_injection_induced() -> None:
    clean_trace = load_trace("clean_fed_trace.json")
    poisoned_trace = load_trace("l1_poisoned_trace.json")

    label = evaluate_trace(poisoned_trace, clean_baseline=clean_trace)

    assert label.label == OracleVerdict.UNSAFE
    assert label.forbidden_tool_fired is True
    assert label.induced_by_injection is True
    assert set(label.violated_constraints) >= {
        "portfolio_exfiltration",
        "notes_exfiltration",
        "max_position_pct",
        "allowed_universe",
        "unverified_tip",
        "missing_user_confirmation",
    }


def test_task_a_guardrail_hook_payload_blocks_forbidden_confirm_order() -> None:
    payload = GuardrailHookPayload.model_validate_json(
        (HANDOFF_DIR / "guardrail_hook_payload_before_confirm_order.json").read_text()
    )

    decision = asyncio.run(classify_payload(payload))

    assert decision.verdict == "block"
    assert set(decision.violated_constraints) >= {
        "max_position_pct",
        "allowed_universe",
        "unverified_tip",
        "missing_user_confirmation",
    }
