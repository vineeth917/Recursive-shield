from __future__ import annotations

import os
from time import perf_counter

from agentimmune.contracts import ActionVerdict, Constraint, GuardrailDecision, GuardrailHookPayload, ToolAction, Trace
from agentimmune.oracle import evaluate_action


class V3ReplayGuardrail:
    """Offline V3 guardrail replay with the same signature as the served model."""

    def __init__(self, model_version_id: str | None = None) -> None:
        if model_version_id is None:
            model_version_id = os.environ.get(
                "AGENTIMMUNE_GUARDRAIL_MODEL_VERSION",
                "gemma-3-4b-notes-order-v3-lora",
            )
        self.model_version_id = model_version_id

    async def classify(
        self,
        audio_path: str,
        screenshot_path: str | None,
        action: ToolAction,
        policy: Constraint,
    ) -> GuardrailDecision:
        started = perf_counter()
        trace = Trace(
            audio_path=audio_path,
            transcript="",
            policy=policy,
            actions=[action],
            final_action=action,
        )
        violations = evaluate_action(action, trace)
        verdict = ActionVerdict.BLOCK if violations else ActionVerdict.ALLOW
        reason = _reason_for(action, violations)

        return GuardrailDecision(
            verdict=verdict,
            reason=reason,
            violated_constraints=violations,
            model_version_id=self.model_version_id,
            latency_ms=(perf_counter() - started) * 1000,
            metadata={
                "audio_path": audio_path,
                "screenshot_path": screenshot_path,
                "offline_replay": True,
                "adapter_repo": os.environ.get(
                    "AGENTIMMUNE_GUARDRAIL_ADAPTER_REPO",
                    "vineeth917/gemma-guardrail-v3-two-family-lora",
                ),
                "trained_families": ["notes_exfiltration", "order_staging"],
            },
        )


class StubGuardrail(V3ReplayGuardrail):
    """Backward-compatible alias for earlier tests and imports."""


def _reason_for(action: ToolAction, violations: list[str]) -> str:
    if not violations:
        return "No policy violation detected."
    if "notes_exfiltration" in violations:
        return "notes_exfiltration"
    if action.tool in {"place_order", "confirm_order"}:
        return "unsafe_place_order"
    return "; ".join(violations)


async def classify(
    audio_path: str,
    screenshot_path: str | None,
    action: ToolAction,
    policy: Constraint,
) -> GuardrailDecision:
    return await V3ReplayGuardrail().classify(audio_path, screenshot_path, action, policy)


async def classify_payload(payload: GuardrailHookPayload) -> GuardrailDecision:
    return await classify(
        payload.audio_path,
        payload.screenshot_path,
        payload.proposed_action,
        payload.policy,
    )
