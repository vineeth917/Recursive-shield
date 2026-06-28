# Task A Strict Trace Audit

Audit date: 2026-06-28

## Verdict

Not ready for final strict C/D training handoff yet.

The current pushed Person B trace bundle resolves the eight requested poisoned split IDs with real Gemini response logs and deterministic oracle labels, but it does not satisfy the newer strict protocol because the clean carrier audio referenced by `AttackSpec.clean_audio_path` is missing.

## Checks

| Requirement | Status | Notes |
| --- | --- | --- |
| Shared `AttackSpec` validation | Pass | All eight files under `artifacts/specs/` validate with `agentimmune.contracts.AttackSpec`. |
| Poisoned audio exists | Pass | All eight `audio_path` files under `artifacts/attacks/` exist. |
| Clean baseline audio exists | Blocked | Every spec points to `artifacts/carriers/fomc_clean.wav`; that file is absent. |
| Clean baseline trace per attack | Missing | Existing bundle has no `*_clean_baseline.json` traces. |
| Poisoned trace links clean run | Missing | Existing traces have `clean_baseline_run_id = null`. |
| Deterministic oracle | Pass | Labels are produced by `agentimmune.oracle`, not an LLM. |
| Native defense evidence | Partial | Existing bundle includes Gemini Computer Use response logs and `native_defense_on = true`; all eight poisoned runs were caught. |
| Custom Gemma guardrail disabled | Pass for next strict run | `scripts/run_split_attack_traces.py` starts brokerage runs with `guardrail_enabled = false`. |
| Actual audio transcription in `Trace.transcript` | Missing in existing bundle | The existing bundle stores spec/payload text. The runner now transcribes WAVs through Gemini before writing new traces. |
| Oracle/native outcome consistency | Fixed for next strict run | The runner now relabels after setting final `native_defense_outcome`. Existing bundle has known stale oracle outcome values. |

## Required Person B Input

Provide the clean carrier file referenced by all eight attack specs:

```text
artifacts/carriers/fomc_clean.wav
```

Alternatively, update the `clean_audio_path` field in each authoritative AttackSpec to point at the correct clean WAV and commit that artifact.

## Strict Rerun Command

Start the fake brokerage server:

```bash
python3.11 -m uvicorn agentimmune.server:app --host 127.0.0.1 --port 8000
```

Then run the strict Gemini + oracle runner:

```bash
python3.11 scripts/run_split_attack_traces.py --model gemini-3.5-flash
```

The rerun will fail fast until `clean_audio_path` exists. Once it succeeds, hand D the new timestamped directory under:

```text
artifacts/person_b_attack_traces/<timestamp>/
```

Expected contents:

```text
manifest.json
trace_lookup.json
clean_trace_lookup.json
traces/<attack_id>.json
traces/<attack_id>__clean_baseline.json
gemini_action_logs/*.json
transcript_logs/*.json
```
