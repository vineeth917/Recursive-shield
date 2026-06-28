# AgentImmune / Recursive Shield

Shared spine for the hackathon system: contracts, oracle stubs, guardrail stubs, eval harness, and later the fine-tune/eval/promote pipeline.

## D1 smoke test

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/smoke_d1.py
pytest
```

The pure D1 scaffold was also smoke-tested locally on Python 3.12.8, but team stack lock remains Python 3.11.

Teammate branches can import either:

```python
from agentimmune.contracts import Trace, AttackSpec
```

or the short compatibility import:

```python
from contracts import Trace, AttackSpec
```

## D2 dry-run

```bash
python scripts/run_d2_dryrun.py
```

This uses stub training plus the stub guardrail, then writes local stand-ins for Atlas collections:

- `artifacts/local_store/model_versions.jsonl`
- `artifacts/local_store/eval_runs.jsonl`

## D3 stub serving and capture

```bash
pip install -e ".[serving]"
uvicorn agentimmune.server:app --host 0.0.0.0 --port 8000
python scripts/capture_demo_stub.py
```

`/classify` accepts the same guardrail payload A will call before `confirm_order`:

```json
{
  "audio_path": "artifacts/audio/poisoned.wav",
  "screenshot_path": "artifacts/screens/order.png",
  "action": {"tool": "confirm_order", "args": {}},
  "policy": {"raw_text": "max 5%", "max_position_pct": 5}
}
```

`/classify-payload` accepts A's full `GuardrailHookPayload` shape:

```json
{
  "run_id": "run_l1",
  "audio_path": "artifacts/audio/poisoned.wav",
  "screenshot_path": "artifacts/screens/order.png",
  "transcript_window": "recent transcript text",
  "proposed_action": {"tool": "confirm_order", "args": {}},
  "policy": {"raw_text": "max 5%", "max_position_pct": 5},
  "recent_actions": [],
  "metadata": {}
}
```

## GPU training box

On Lightning AI / Colab GPU:

```bash
pip install -e ".[dev,mongo,voyage,google]"
pip install -r requirements/gpu-training.txt
```

Do not install `requirements/gpu-training.txt` on local Mac unless you specifically want to debug dependency resolution.

## Data Handoff Checks

Voyage smoke test, using secrets from your shell only:

```bash
export VOYAGE_API_KEY="..."
export VOYAGE_MODEL="voyage-4-large"
export VOYAGE_DIMENSION="1024"
python -m agentimmune_data.cli smoke-voyage
```

Fail the build if current handoff artifacts are still synthetic/missing:

```bash
python -m agentimmune_data.cli audit-no-stub --strict
```

Validate C's future `split.json`:

```bash
python -m agentimmune_data.cli validate-split path/to/split.json
```

Build C's frozen split and run the leakage firewall:

```bash
python -m agentimmune_data.cli split artifacts/specs --benign fixtures/task_a_handoff/clean_fed_trace.json --out split.json
python -m agentimmune_data.cli leakage-check artifacts/specs --split split.json
```

Task C pins:

- Mongo URI: provide with `MONGODB_URI` or `MONGO_URI` secret only
- Database: `agentimmune`
- Collections: `traces`, `attacks`, `attack_embeddings`, `model_versions`, `eval_runs`
- Vector index: `attack_embedding_vector_index` on `attack_embeddings.embedding`
- Voyage model: `voyage-4-large`
- Embedding dimension: `1024`
- Duplicate threshold tau: `0.92`

Build transcript-fallback SFT JSONL for D4:

```bash
python -m agentimmune_data.cli build-sft path/to/split.json --trace-lookup artifacts/trace_lookup.json --out artifacts/training/sft_train.jsonl
```

When `split.json` contains IDs only, create `artifacts/trace_lookup.json`:

```json
{
  "attack_id_or_run_id": "path/to/labeled_trace.json"
}
```

Check every split ID resolves to a labeled `Trace` before D4:

```bash
python -m agentimmune_data.cli resolve-check split.json --trace-lookup artifacts/trace_lookup.json
```

Run the data firewall before training:

```bash
python -m agentimmune_data.cli leakage-check artifacts/specs --split split.json
python -m agentimmune_data.cli resolve-check split.json --trace-lookup artifacts/trace_lookup.json
python -m agentimmune_data.cli audio-audit artifacts/specs --trace-lookup artifacts/trace_lookup.json
python -m agentimmune_data.cli handoff-report split.json --trace-lookup artifacts/trace_lookup.json --specs artifacts/specs
```

`audio-audit` intentionally fails if placeholder/fake audio slips in: duplicate WAV hashes across attack IDs, missing clean carriers, empty ASR transcript, or payload-text transcripts without ASR provenance.

Mongo proof and embeddings use secrets from the shell only:

```bash
python -m agentimmune_data.cli mongo-proof
python -m agentimmune_data.cli embed-attacks artifacts/specs
```

See `docs/trace_lookup_contract.md` and `artifacts/trace_lookup.example.json`.
