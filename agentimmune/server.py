from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from agentimmune.brokerage import router as brokerage_router
from agentimmune.contracts import Constraint, GuardrailDecision, GuardrailHookPayload, ToolAction
from agentimmune.guardrail import classify, classify_payload, preload_guardrail


app = FastAPI(title="AgentImmune Guardrail", version="0.1.0")
app.include_router(brokerage_router)


class ClassifyRequest(BaseModel):
    audio_path: str
    screenshot_path: str | None = None
    action: ToolAction
    policy: Constraint


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "guardrail_mode": os.environ.get("AGENTIMMUNE_GUARDRAIL_MODE", "oracle"),
        "mongo_stream": os.environ.get("AGENTIMMUNE_MONGO_STREAM", "0"),
    }


@app.on_event("startup")
async def startup() -> None:
    if os.environ.get("AGENTIMMUNE_GUARDRAIL_PRELOAD", "").lower() in {"1", "true", "yes", "on"}:
        preload_guardrail()


@app.post("/classify", response_model=GuardrailDecision)
async def classify_endpoint(payload: ClassifyRequest) -> GuardrailDecision:
    return await classify(
        audio_path=payload.audio_path,
        screenshot_path=payload.screenshot_path,
        action=payload.action,
        policy=payload.policy,
    )


@app.post("/classify-payload", response_model=GuardrailDecision)
async def classify_payload_endpoint(payload: GuardrailHookPayload) -> GuardrailDecision:
    return await classify_payload(payload)
