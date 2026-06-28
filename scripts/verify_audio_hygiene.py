import os
import json
import hashlib
import wave
import speech_recognition as sr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
MANIFEST_PATH = ARTIFACTS_DIR / "specs" / "l1_manifest.json"

def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return round(frames / float(rate), 2)

def run_asr(path: Path) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(str(path)) as source:
        audio = r.record(source)
    try:
        # Use pocket sphinx or google speech offline wrapper
        text = r.recognize_google(audio)
        return text
    except Exception as e:
        return f"[ASR Failed: {e}]"

def main():
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    hashes = set()
    all_unique = True
    
    print("\n" + "="*80)
    print("RUNNING AUDIO HYGIENE & ASR VERIFICATION REPORT")
    print("="*80)
    
    report_rows = []

    for item in manifest:
        audio_path = ROOT / item["audio_path"]
        payload_path = ROOT / item["payload_path"]
        
        # Verify file existence
        if not audio_path.exists():
            print(f"Error: missing attack file at {audio_path}")
            continue
            
        sha = get_sha256(audio_path)
        duration = get_duration(audio_path)
        
        # Check uniqueness of hashes
        if sha in hashes:
            all_unique = False
            print(f"Duplicate hash detected: {item['attack_id']} uses already seen audio!")
        hashes.add(sha)
        
        # Run ASR on the raw payload audio for clear speech checking
        print(f"Running ASR on payload: {payload_path.name}...")
        asr_transcript = run_asr(payload_path)
        item["ASR_transcript"] = asr_transcript
        
        report_rows.append(
            f"| {item['attack_id']} | {item['family']} | {duration}s | `{sha[:10]}...` | {asr_transcript} |"
        )
        
    # Write updated manifest back
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print markdown report
    print("\n### Audio Quality & ASR Manifest Report")
    print("| Attack ID | Family | Duration | SHA-256 (prefix) | ASR Transcript |")
    print("|---|---|---|---|---|")
    for row in report_rows:
        print(row)
        
    print("\n" + "="*80)
    print(f"Hash uniqueness verification check: {'PASS' if all_unique else 'FAIL'}")
    print("="*80)

if __name__ == "__main__":
    main()
