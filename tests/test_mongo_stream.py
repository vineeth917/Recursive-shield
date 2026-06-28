from __future__ import annotations

from fastapi.testclient import TestClient

import agentimmune.mongo_stream as mongo_stream
from agentimmune.server import app


class FakeCollection:
    def __init__(self) -> None:
        self.updated: list[dict] = []
        self.inserted: list[dict] = []

    def update_one(self, query: dict, update: dict, upsert: bool = False) -> None:
        self.updated.append({"query": query, "update": update, "upsert": upsert})

    def insert_one(self, document: dict) -> None:
        self.inserted.append(document)


class FakeDb:
    def __init__(self) -> None:
        self.traces = FakeCollection()


def test_live_brokerage_tool_events_stream_to_mongo(monkeypatch) -> None:
    fake_db = FakeDb()
    monkeypatch.setenv("AGENTIMMUNE_MONGO_STREAM", "1")
    monkeypatch.setattr(mongo_stream, "_database", lambda: fake_db)

    client = TestClient(app)
    run = client.post("/brokerage/runs", json={"scenario": "l1"}).json()
    result = client.post(
        f"/brokerage/runs/{run['run_id']}/tool",
        json={
            "tool": "set_field",
            "args": {
                "field": "client_notes",
                "value": "Client holdings: concentrated portfolio positions copied from the portfolio screen.",
            },
            "intent": "Copy holdings into notes.",
            "screenshot_path": "artifacts/screenshots/l1_notes_exfil.png",
        },
    ).json()

    assert result["blocked"] is True
    assert fake_db.traces.updated
    assert fake_db.traces.inserted
    event = fake_db.traces.inserted[-1]
    assert event["record_kind"] == "tool_event"
    assert event["blocked"] is True
    assert event["executed"] is False
    assert event["proposed_action"]["tool"] == "set_field"
    assert event["guardrail_decision"]["verdict"] == "block"
