from __future__ import annotations

import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "fixtures" / "person_b_attack_traces" / "person_b_attack_traces_20260628T025027Z.zip"
PREFIX = "artifacts/person_b_attack_traces/20260628T025027Z"
ATTACK_IDS = [
    "l0_ad_break_splice_seed_244",
    "l0_constraint_override_seed_795",
    "l0_spoofed_host_directive_seed_515",
    "l1_constraint_override_seed_638",
    "l1_fake_breaking_news_urgency_seed_676",
    "l1_ad_break_splice_seed_230",
    "l0_fake_breaking_news_urgency_seed_996",
    "l1_spoofed_host_directive_seed_752",
]


def read_json(archive: zipfile.ZipFile, relpath: str) -> dict:
    return json.loads(archive.read(f"{PREFIX}/{relpath}").decode("utf-8"))


def test_person_b_attack_bundle_contains_all_split_ids() -> None:
    with zipfile.ZipFile(BUNDLE) as archive:
        names = set(archive.namelist())
        assert f"{PREFIX}/manifest.json" in names
        assert f"{PREFIX}/trace_lookup.json" in names
        for attack_id in ATTACK_IDS:
            assert f"{PREFIX}/traces/{attack_id}.json" in names
            assert f"{PREFIX}/gemini_action_logs/{attack_id}.json" in names


def test_person_b_attack_bundle_traces_are_labeled_and_resolve_ids() -> None:
    with zipfile.ZipFile(BUNDLE) as archive:
        manifest = read_json(archive, "manifest.json")
        assert manifest["model_id"] == "gemini-3.5-flash"
        assert manifest["native_defense_on"] is True

        lookup = read_json(archive, "trace_lookup.json")
        assert set(lookup) == set(ATTACK_IDS)

        for attack_id in ATTACK_IDS:
            trace = read_json(archive, f"traces/{attack_id}.json")
            assert trace["attack_id"] == attack_id
            assert trace["oracle"]["label"] in {"safe", "unsafe"}
            assert trace["native_defense_outcome"] in {"caught", "bypassed"}
            assert trace["metadata"]["gemini_response"]["model"] == "gemini-3.5-flash"
