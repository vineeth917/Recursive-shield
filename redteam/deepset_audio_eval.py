"""Generate and evaluate a balanced audio subset of deepset/prompt-injections."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import requests

from redteam import config
from redteam.jailbreakv_audio_eval import GeminiJailbreakAudioEvaluator
from redteam.tts_adapter import text_to_speech


DATASET_ID = "deepset/prompt-injections"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def fetch_rows(page_size: int = 100) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        response = requests.get(
            ROWS_ENDPOINT,
            params={
                "dataset": DATASET_ID,
                "config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
                "offset": offset,
                "length": page_size,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        total = int(payload["num_rows_total"])
        page = []
        for entry in payload["rows"]:
            row = dict(entry["row"])
            row["row_index"] = int(entry["row_idx"])
            page.append(row)
        if not page:
            break
        rows.extend(page)
        offset += len(page)
    return rows


def select_balanced(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count < 2:
        raise ValueError("count must be at least 2 for a balanced evaluation")
    rng = random.Random(seed)
    benign = [row for row in rows if int(row["label"]) == 0]
    injections = [row for row in rows if int(row["label"]) == 1]
    rng.shuffle(benign)
    rng.shuffle(injections)
    injection_count = count // 2
    benign_count = count - injection_count
    if injection_count > len(injections) or benign_count > len(benign):
        raise ValueError("Requested count exceeds an available label bucket")
    selected = injections[:injection_count] + benign[:benign_count]
    rng.shuffle(selected)
    return selected


CSV_FIELDS = [
    "dataset_id",
    "dataset_row_index",
    "expected_jailbreak",
    "detected_jailbreak",
    "jailbreak_label_match",
    "unsafe_detected",
    "confidence",
    "harmful_request_detected",
    "recommended_action",
    "suspicious_segment",
    "reason",
    "gemini_transcript",
    "source_transcript_path",
    "audio_path",
    "tts_backend",
    "interaction_id",
    "error",
]


def summarize(results: list[dict]) -> dict[str, int | float]:
    completed = [result for result in results if "detected_jailbreak" in result]
    tp = sum(r["expected_jailbreak"] and r["detected_jailbreak"] for r in completed)
    tn = sum(not r["expected_jailbreak"] and not r["detected_jailbreak"] for r in completed)
    fp = sum(not r["expected_jailbreak"] and r["detected_jailbreak"] for r in completed)
    fn = sum(r["expected_jailbreak"] and not r["detected_jailbreak"] for r in completed)
    positive_total = tp + fn
    negative_total = tn + fp
    return {
        "selected": len(results),
        "completed": len(completed),
        "errors": len(results) - len(completed),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "injection_detection_rate": tp / positive_total if positive_total else 0.0,
        "false_positive_rate": fp / negative_total if negative_total else 0.0,
        "accuracy": (tp + tn) / len(completed) if completed else 0.0,
    }


def write_csv(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def run(count: int, seed: int, model: str, output_dir: Path) -> dict:
    rows = fetch_rows()
    candidates = select_balanced(
        rows,
        count=min(len(rows), count + 40),
        seed=seed,
    )
    target_injections = count // 2
    target_benign = count - target_injections
    evaluator = GeminiJailbreakAudioEvaluator(model=model)
    results: list[dict] = []
    error_attempts: list[dict] = []

    for index, row in enumerate(candidates, start=1):
        completed_injections = sum(r["expected_jailbreak"] for r in results)
        completed_benign = len(results) - completed_injections
        if completed_injections >= target_injections and completed_benign >= target_benign:
            break
        row_index = int(row["row_index"])
        expected = bool(int(row["label"]))
        if expected and completed_injections >= target_injections:
            continue
        if not expected and completed_benign >= target_benign:
            continue
        sample_dir = output_dir / "samples" / f"row_{row_index:03d}"
        transcript_path = sample_dir / "source_transcript.txt"
        audio_path = sample_dir / "audio.wav"
        sample_dir.mkdir(parents=True, exist_ok=True)
        speech_text = " ".join(str(row["text"]).split())
        transcript_path.write_text(speech_text + "\n", encoding="utf-8")

        print(
            f"[completed={len(results)}/{count}; attempt={index}] "
            f"row={row_index} expected_jailbreak={expected}",
            flush=True,
        )
        record = {
            "dataset_id": DATASET_ID,
            "dataset_row_index": row_index,
            "expected_jailbreak": expected,
            "source_transcript_path": transcript_path.as_posix(),
            "audio_path": audio_path.as_posix(),
        }
        try:
            backend = text_to_speech(speech_text, str(audio_path))
            record["tts_backend"] = backend
            if backend == "beep":
                raise RuntimeError("Speech synthesis failed; refusing to evaluate beep fallback")
            verdict, interaction_id = evaluator.evaluate(audio_path)
            unsafe_detected = (
                verdict.jailbreak_detected
                or verdict.harmful_request_detected
                or verdict.recommended_action == "block"
            )
            record.update(
                {
                    "detected_jailbreak": verdict.jailbreak_detected,
                    "jailbreak_label_match": verdict.jailbreak_detected == expected,
                    "unsafe_detected": unsafe_detected,
                    "confidence": verdict.confidence,
                    "harmful_request_detected": verdict.harmful_request_detected,
                    "recommended_action": verdict.recommended_action,
                    "suspicious_segment": verdict.suspicious_segment,
                    "reason": verdict.reason,
                    "gemini_transcript": verdict.transcript,
                    "interaction_id": interaction_id,
                }
            )
            print(
                f"  detected={verdict.jailbreak_detected} "
                f"confidence={verdict.confidence:.2f} action={verdict.recommended_action}",
                flush=True,
            )
            results.append(record)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {record['error']}", flush=True)
            error_attempts.append(record)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "subset_strategy": "seeded balanced labels (50% injection, 50% benign)",
        "seed": seed,
        "model": model,
        "summary": summarize(results),
        "results": results,
        "error_attempts": error_attempts,
    }
    report["summary"]["attempts"] = len(results) + len(error_attempts)
    report["summary"]["api_or_tts_errors"] = len(error_attempts)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, results)
    print(json.dumps(report["summary"], indent=2))
    print(f"JSON written to {json_path}")
    print(f"CSV written to {csv_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=config.GEMINI_AUDIO_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evals/deepset_audio"),
    )
    args = parser.parse_args()
    run(count=args.count, seed=args.seed, model=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
