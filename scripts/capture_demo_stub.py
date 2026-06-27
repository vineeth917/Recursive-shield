from __future__ import annotations

import asyncio

from agentimmune.capture import capture_trace_classification, load_capture
from agentimmune.contracts import Trace
from agentimmune.sample_data import sample_split


async def main() -> None:
    split = sample_split()
    trace = Trace.model_validate(split["held_out"][0])
    capture_path = await capture_trace_classification(trace)
    _, decision = load_capture(capture_path)
    print(f"capture_path={capture_path}")
    print(f"replay_verdict={decision.verdict}")
    print(f"model_version_id={decision.model_version_id}")


if __name__ == "__main__":
    asyncio.run(main())
