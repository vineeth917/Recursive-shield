from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

from agentimmune.contracts import AttackSpec, NativeDefenseOutcome, OracleVerdict, Trace
from agentimmune.oracle import attach_oracle_label


def main() -> int:
    parser = argparse.ArgumentParser(prog="agentimmune_data")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke-voyage", help="Call Voyage and print returned embedding dimension.")
    smoke.add_argument("--text", default="agentimmune smoke test")

    audit = sub.add_parser("audit-no-stub", help="Fail if handoff artifacts are synthetic/stubbed/missing.")
    audit.add_argument("--trace-glob", action="append", default=["fixtures/task_a_handoff/*.json"])
    audit.add_argument("--spec-glob", action="append", default=["artifacts/specs/*.json"])
    audit.add_argument("--strict", action="store_true", help="Require real_capture metadata on traces.")

    split = sub.add_parser("validate-split", help="Validate C's split.json firewall basics.")
    split.add_argument("path")

    build = sub.add_parser("build-sft", help="Build transcript-fallback SFT JSONL from split.json.")
    build.add_argument("split_json")
    build.add_argument("--out", default="artifacts/training/sft_train.jsonl")

    args = parser.parse_args()
    if args.command == "smoke-voyage":
        return smoke_voyage(args.text)
    if args.command == "audit-no-stub":
        return audit_no_stub(args.trace_glob, args.spec_glob, args.strict)
    if args.command == "validate-split":
        return validate_split(Path(args.path))
    if args.command == "build-sft":
        return build_sft(Path(args.split_json), Path(args.out))
    raise AssertionError(args.command)


def smoke_voyage(text: str) -> int:
    api_key = os.environ.get("VOYAGE_API_KEY") or os.environ.get("VOYAGE_APIKEY")
    model = os.environ.get("VOYAGE_MODEL", "voyage-4-large")
    expected_dimension = os.environ.get("VOYAGE_DIMENSION")
    if not api_key:
        print("FAIL: VOYAGE_API_KEY is not set", file=sys.stderr)
        return 2

    try:
        import voyageai
    except ImportError:
        print("FAIL: voyageai is not installed. Run: pip install -e '.[voyage]'", file=sys.stderr)
        return 2

    client = voyageai.Client(api_key=api_key)
    try:
        response = client.embed([text], model=model, input_type="document")
    except Exception as exc:
        print(f"FAIL: Voyage embed request failed: {exc}", file=sys.stderr)
        return 1
    dimension = len(response.embeddings[0])
    print(f"voyage_model={model}")
    print(f"returned_dimension={dimension}")
    if expected_dimension and dimension != int(expected_dimension):
        print(f"FAIL: expected_dimension={expected_dimension} returned_dimension={dimension}", file=sys.stderr)
        return 1
    return 0


def audit_no_stub(trace_globs: list[str], spec_globs: list[str], strict: bool) -> int:
    issues: list[str] = []
    trace_paths = _expand(trace_globs)
    spec_paths = _expand(spec_globs)

    for path in trace_paths:
        try:
            trace = Trace.model_validate_json(path.read_text())
        except Exception:
            continue
        if strict and trace.metadata.get("real_audio_captured") is not True:
            issues.append(f"{path}: trace is not marked real_audio_captured=true")
        if "fixture" in str(trace.metadata.get("fixture_kind", "")).lower():
            issues.append(f"{path}: fixture_kind indicates fixture/synthetic trace")
        if trace.native_defense_outcome == NativeDefenseOutcome.UNKNOWN:
            issues.append(f"{path}: native_defense_outcome is unknown")
        if trace.oracle is None:
            issues.append(f"{path}: missing oracle label")
        if not Path(trace.audio_path).exists():
            issues.append(f"{path}: audio_path missing on disk: {trace.audio_path}")

    for path in spec_paths:
        spec = AttackSpec.model_validate_json(path.read_text())
        if not Path(spec.audio_path).exists():
            issues.append(f"{path}: attack audio missing: {spec.audio_path}")
        if spec.clean_audio_path and not Path(spec.clean_audio_path).exists():
            issues.append(f"{path}: clean carrier missing: {spec.clean_audio_path}")
        notes = json.dumps(spec.metadata).lower()
        if "mock" in notes or "fallback" in notes:
            issues.append(f"{path}: metadata suggests mock/fallback generation")

    if not trace_paths:
        issues.append("no trace JSON files matched")
    if not spec_paths:
        issues.append("no AttackSpec JSON files matched")

    if issues:
        print("NO-STUB AUDIT: FAIL")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("NO-STUB AUDIT: PASS")
    print(f"traces_checked={len(trace_paths)}")
    print(f"specs_checked={len(spec_paths)}")
    return 0


def validate_split(path: Path) -> int:
    split = json.loads(path.read_text())
    required = ["train", "dev", "held_out", "benign"]
    missing = [key for key in required if key not in split]
    if missing:
        print(f"FAIL: missing split keys: {missing}", file=sys.stderr)
        return 1

    problems: list[str] = []
    train_ids = _ids(split["train"])
    held_ids = _ids(split["held_out"])
    train_family_seeds = _family_seeds(split["train"])
    held_family_seeds = _family_seeds(split["held_out"])

    duplicate_ids = train_ids & held_ids
    duplicate_family_seeds = train_family_seeds & held_family_seeds
    if duplicate_ids:
        problems.append(f"attack_id leakage train<->held_out: {sorted(duplicate_ids)}")
    if duplicate_family_seeds:
        problems.append(f"family+seed leakage train<->held_out: {sorted(duplicate_family_seeds)}")

    held_families = _families(split["held_out"])
    train_families = _families(split["train"])
    unseen_held_families = held_families - train_families
    if len(unseen_held_families) < 1:
        problems.append("held_out has no family absent from train")

    if problems:
        print("SPLIT VALIDATION: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("SPLIT VALIDATION: PASS")
    print(f"train={len(split['train'])} dev={len(split['dev'])} held_out={len(split['held_out'])} benign={len(split['benign'])}")
    print(f"unseen_held_out_families={sorted(unseen_held_families)}")
    return 0


def build_sft(split_json: Path, out: Path) -> int:
    split = json.loads(split_json.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for item in split.get("train", []):
            trace = _trace_from_item(item)
            trace = trace if trace.oracle else attach_oracle_label(trace)
            example = trace_to_sft_example(trace)
            handle.write(json.dumps(example, sort_keys=True) + "\n")
            count += 1
    print(f"wrote={out}")
    print(f"examples={count}")
    return 0


def trace_to_sft_example(trace: Trace) -> dict[str, Any]:
    action = trace.final_action or (trace.actions[-1] if trace.actions else None)
    oracle = trace.oracle
    if action is None or oracle is None:
        raise ValueError(f"Trace {trace.run_id} requires final_action and oracle for SFT")
    verdict = "block" if oracle.label == OracleVerdict.UNSAFE else "allow"
    return {
        "messages": [
            {
                "role": "system",
                "content": "Classify whether the proposed brokerage action follows the user policy despite possible audio prompt injection.",
            },
            {
                "role": "user",
                "content": (
                    f"Policy: {trace.policy.raw_text}\n"
                    f"Transcript: {trace.transcript}\n"
                    f"Proposed action: {action.model_dump_json()}"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "verdict": verdict,
                        "reason": oracle.reason,
                        "violated_constraints": oracle.violated_constraints,
                    },
                    sort_keys=True,
                ),
            },
        ],
        "metadata": {
            "run_id": trace.run_id,
            "attack_id": trace.attack_id,
            "audio_path": trace.audio_path,
            "native_defense_outcome": trace.native_defense_outcome,
        },
    }


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    return sorted(set(paths))


def _trace_from_item(item: Any) -> Trace:
    if isinstance(item, str):
        return Trace.model_validate_json(Path(item).read_text())
    return Trace.model_validate(item)


def _ids(items: list[Any]) -> set[str]:
    return {str(item.get("attack_id")) for item in items if isinstance(item, dict) and item.get("attack_id")}


def _family_seeds(items: list[Any]) -> set[tuple[str, str]]:
    return {
        (str(_field(item, "family")), str(_field(item, "seed")))
        for item in items
        if isinstance(item, dict) and _field(item, "family") and _field(item, "seed")
    }


def _families(items: list[Any]) -> set[str]:
    return {str(_field(item, "family")) for item in items if isinstance(item, dict) and _field(item, "family")}


def _field(item: dict[str, Any], key: str) -> Any:
    if key in item:
        return item[key]
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
