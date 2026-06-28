# Trace Lookup Contract

`split.json` may contain stable attack IDs or benign run IDs. Fine-tuning needs those IDs to resolve to labeled `Trace` JSON files.

Create:

```json
{
  "attack_id_or_run_id": "path/to/labeled_trace.json"
}
```

The current sample template is:

```bash
artifacts/trace_lookup.example.json
```

Copy it to `artifacts/trace_lookup.json` after A provides real labeled traces, then replace every path with the real Trace JSON path.

Each resolved Trace must validate against `agentimmune.contracts.Trace` and include:

- `run_id`
- `attack_id` for attack traces
- `oracle.label`
- `native_defense_outcome`
- `actions`
- `metadata.gemini_evidence`

If Gemini catches an attack before any tool call, still save the Trace with:

```json
{
  "native_defense_outcome": "caught",
  "actions": [],
  "metadata": {
    "gemini_evidence": {}
  }
}
```

Run before D4 fine-tuning:

```bash
python -m agentimmune_data.cli resolve-check split.json --trace-lookup artifacts/trace_lookup.json
python -m agentimmune_data.cli leakage-check artifacts/specs --split split.json
python -m agentimmune_data.cli build-sft split.json --trace-lookup artifacts/trace_lookup.json --out artifacts/training/sft_train.jsonl
```

`resolve-check` prints:

```text
attack_id | trace_path | native_defense_outcome | actions | oracle_label | embedded | in_split
```
