# ASSUMPTIONS.md — Task D / Lead

## Current status
- D1 scaffold is self-contained and does not require Gemini, MongoDB, Voyage, MiniMax, or GPU access.
- The guardrail interface is frozen as `classify(audio_path, screenshot_path, action, policy) -> GuardrailDecision`.
- The top-level `contracts.py` re-export exists so A/B/C can `import contracts` immediately.

## Open assumptions
- Audio-native Gemma-4 fine-tune speed is unverified; transcript-only SFT remains the fallback.
- Real D4 rounds require Person C's frozen `split.json`, benign set `D`, and labeled traces.
- Real attack potency requires Person A's oracle plus Person B's playable `AttackSpec` audio.
- MongoDB Atlas URI, Voyage model id/dimension, and vector threshold are pending from Person C.

## Weakest link
- The L1 bypass and labeled dataset are upstream dependencies. Until A/B/C hand off traces and `split.json`, Task D can only dry-run the pipeline with stub data.
