from __future__ import annotations

from pathlib import Path

from agentimmune.orchestrator import run_round_sync
from agentimmune.sample_data import sample_split
from agentimmune.store import LocalJsonlStore


def main() -> None:
    store = LocalJsonlStore(Path("artifacts/local_store"))
    result = run_round_sync(
        split=sample_split(),
        previous_metrics={"held_out_block_rate": 0.0, "benign_fp_rate": 0.0},
        store=store,
    )

    print(f"model_version_id={result.model_version.model_version_id}")
    print(f"train_set_hash={result.model_version.train_set_hash}")
    print(f"promoted={result.model_version.promoted}")
    print(f"held_out_block_rate={result.metrics.held_out_block_rate:.2f}")
    print(f"benign_fp_rate={result.metrics.benign_fp_rate:.2f}")
    print("wrote=artifacts/local_store/model_versions.jsonl")
    print("wrote=artifacts/local_store/eval_runs.jsonl")


if __name__ == "__main__":
    main()
