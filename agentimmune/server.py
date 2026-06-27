from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from agentimmune.contracts import Constraint, GuardrailDecision, ToolAction
from agentimmune.guardrail import StubGuardrail


app = FastAPI(title="AgentImmune Guardrail", version="0.1.0")
guardrail = StubGuardrail()


class ClassifyRequest(BaseModel):
    audio_path: str
    screenshot_path: str | None = None
    action: ToolAction
    policy: Constraint


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "backend": "stub"}


@app.post("/classify", response_model=GuardrailDecision)
async def classify_endpoint(payload: ClassifyRequest) -> GuardrailDecision:
    return await guardrail.classify(
        audio_path=payload.audio_path,
        screenshot_path=payload.screenshot_path,
        action=payload.action,
        policy=payload.policy,
    )
