from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class LocalJsonlStore:
    """Atlas-shaped local writer for dry-runs before Person C's cluster lands."""

    def __init__(self, root: str | Path = "artifacts/local_store") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def insert_one(self, collection: str, document: BaseModel | dict[str, Any]) -> None:
        payload = document.model_dump(mode="json") if isinstance(document, BaseModel) else document
        path = self.root / f"{collection}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def read_all(self, collection: str) -> list[dict[str, Any]]:
        path = self.root / f"{collection}.jsonl"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
