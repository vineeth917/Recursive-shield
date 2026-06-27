from __future__ import annotations

import json
from pathlib import Path

from agentimmune.contracts import GuardrailDecision, Trace
from agentimmune.guardrail import StubGuardrail


async def capture_trace_classification(
    trace: Trace,
    output_dir: str | Path = "artifacts/captures",
    guardrail: StubGuardrail | None = None,
) -> Path:
    guardrail = guardrail or StubGuardrail()
    action = trace.final_action or (trace.actions[-1] if trace.actions else None)
    if action is None:
        raise ValueError(f"Trace {trace.run_id} has no final action to capture")

    decision = await guardrail.classify(trace.audio_path, action.screenshot_path, action, trace.policy)
    return write_capture(trace, decision, output_dir)


def write_capture(trace: Trace, decision: GuardrailDecision, output_dir: str | Path) -> Path:
    capture_dir = Path(output_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"{trace.run_id}.json"
    payload = {
        "trace": trace.model_dump(mode="json"),
        "guardrail_decision": decision.model_dump(mode="json"),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def load_capture(path: str | Path) -> tuple[Trace, GuardrailDecision]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Trace.model_validate(payload["trace"]), GuardrailDecision.model_validate(payload["guardrail_decision"])
