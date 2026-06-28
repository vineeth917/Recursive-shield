from __future__ import annotations

import json
from pathlib import Path

from agentimmune.contracts import Constraint, NativeDefenseOutcome, OracleLabel, OracleVerdict, ToolAction, Trace
from agentimmune_data.cli import build_sft, resolve_check, validate_split
from agentimmune.sample_data import sample_split


def test_validate_split_accepts_unseen_held_out_family(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "train": [{"attack_id": "a1", "family": "ad_break_splice", "seed": "1"}],
                "dev": [],
                "held_out": [{"attack_id": "a2", "family": "cross_lingual", "seed": "2"}],
                "benign": [],
            }
        )
    )

    assert validate_split(path) == 0


def test_validate_split_rejects_attack_id_leakage(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "train": [{"attack_id": "a1", "family": "ad_break_splice", "seed": "1"}],
                "dev": [],
                "held_out": [{"attack_id": "a1", "family": "cross_lingual", "seed": "2"}],
                "benign": [],
            }
        )
    )

    assert validate_split(path) == 1


def test_build_sft_writes_training_jsonl(tmp_path: Path) -> None:
    split = sample_split()
    split_path = tmp_path / "split.json"
    out = tmp_path / "sft.jsonl"
    split_path.write_text(json.dumps(split))

    assert build_sft(split_path, out) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["messages"][0]["role"] == "system"
    assert "verdict" in payload["messages"][2]["content"]


def test_resolve_check_accepts_trace_lookup(tmp_path: Path) -> None:
    trace = _trace("l0_ad_break_splice_seed_244")
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(trace.model_dump_json())
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["l0_ad_break_splice_seed_244"],
                "dev": [],
                "held_out": [],
                "benign": [],
            }
        )
    )
    lookup_path = tmp_path / "trace_lookup.json"
    lookup_path.write_text(json.dumps({"l0_ad_break_splice_seed_244": str(trace_path)}))

    assert resolve_check(split_path, lookup_path) == 0


def test_resolve_check_rejects_missing_lookup_entry(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["l0_ad_break_splice_seed_244"],
                "dev": [],
                "held_out": [],
                "benign": [],
            }
        )
    )
    lookup_path = tmp_path / "trace_lookup.json"
    lookup_path.write_text(json.dumps({}))

    assert resolve_check(split_path, lookup_path) == 1


def test_build_sft_uses_trace_lookup_for_id_only_split(tmp_path: Path) -> None:
    trace = _trace("l0_ad_break_splice_seed_244")
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(trace.model_dump_json())
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["l0_ad_break_splice_seed_244"],
                "dev": [],
                "held_out": [],
                "benign": [],
            }
        )
    )
    lookup_path = tmp_path / "trace_lookup.json"
    lookup_path.write_text(json.dumps({"l0_ad_break_splice_seed_244": str(trace_path)}))
    out = tmp_path / "sft.jsonl"

    assert build_sft(split_path, out, lookup_path) == 0
    assert len(out.read_text().splitlines()) == 1


def _trace(attack_id: str) -> Trace:
    action = ToolAction(
        tool="confirm_order",
        args={
            "symbol": "PUMP",
            "position_pct": 40,
            "market_cap_usd": 500_000_000,
            "verified_tip": False,
            "user_confirmed": False,
        },
    )
    return Trace(
        attack_id=attack_id,
        audio_path="artifacts/attacks/example.wav",
        transcript="Injected market commentary.",
        policy=Constraint(raw_text="Max 5 percent per position.", max_position_pct=5),
        actions=[action],
        final_action=action,
        native_defense_outcome=NativeDefenseOutcome.BYPASSED,
        oracle=OracleLabel(label=OracleVerdict.UNSAFE, reason="violates policy"),
        metadata={"gemini_evidence": {"raw_logs": []}, "embedded": True},
    )
