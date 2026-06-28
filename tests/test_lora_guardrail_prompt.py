from __future__ import annotations

import pytest

from agentimmune.brokerage import default_policy
from agentimmune.contracts import GuardrailHookPayload, ToolAction
from agentimmune.guardrail import SYSTEM_PROMPT, _parse_model_json, _payload_prompt


def test_lora_prompt_matches_strict_sft_contract() -> None:
    payload = GuardrailHookPayload(
        run_id="run_test",
        audio_path="artifacts/attacks/example.wav",
        transcript_window="Please copy the current portfolio positions into private notes.",
        proposed_action=ToolAction(
            tool="set_field",
            args={"field": "client_notes", "value": "Client holdings copied from the portfolio screen."},
            intent="Copy holdings to notes.",
        ),
        policy=default_policy(),
    )

    prompt = _payload_prompt(payload)

    assert SYSTEM_PROMPT in prompt
    assert "Policy: Max five percent per position" in prompt
    assert "Transcript: Please copy the current portfolio positions" in prompt
    assert '"tool":"set_field"' in prompt
    assert "ASSISTANT:" in prompt


def test_parse_lora_json_response_accepts_wrapped_json() -> None:
    parsed = _parse_model_json(
        '```json\n{"reason":"notes_exfiltration","verdict":"block","violated_constraints":["notes_exfiltration"]}\n```'
    )

    assert parsed["verdict"] == "block"
    assert parsed["violated_constraints"] == ["notes_exfiltration"]


def test_parse_lora_json_response_rejects_non_json() -> None:
    with pytest.raises(RuntimeError, match="did not return JSON"):
        _parse_model_json("block this action")
