from __future__ import annotations

from time import perf_counter

from agentimmune.contracts import ActionVerdict, Constraint, GuardrailDecision, GuardrailHookPayload, ToolAction, Trace
from agentimmune.oracle import evaluate_action


class StubGuardrail:
    """Drop-in guardrail with the same signature the served model will expose."""

    def __init__(self, model_version_id: str = "stub-v0") -> None:
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
        reason = "Policy violation: " + ", ".join(violations) if violations else "Allowed by stub policy check."

        return GuardrailDecision(
            verdict=verdict,
            reason=reason,
            violated_constraints=violations,
            model_version_id=self.model_version_id,
            latency_ms=(perf_counter() - started) * 1000,
            metadata={
                "audio_path": audio_path,
                "screenshot_path": screenshot_path,
                "stub": True,
            },
        )


async def classify(
    audio_path: str,
    screenshot_path: str | None,
    action: ToolAction,
    policy: Constraint,
) -> GuardrailDecision:
    return await StubGuardrail().classify(audio_path, screenshot_path, action, policy)


async def classify_payload(payload: GuardrailHookPayload) -> GuardrailDecision:
    return await classify(
        payload.audio_path,
        payload.screenshot_path,
        payload.proposed_action,
        payload.policy,
    )
