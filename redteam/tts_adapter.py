from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any

import requests

from redteam import config


def generate_mock_wav(
    output_path: str,
    text: str = "This is a fallback audio instruction.",
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
) -> str:
    """Generate local speech for tests, with a clearly identified tone fallback.

    Production/campaign generation must call :func:`synthesize_minimax` or
    ``text_to_speech(..., strict=True)`` so this fallback can never silently enter
    an evaluation dataset.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        print(f"Generated spoken audio offline via pyttsx3: {output_path}")
        return "pyttsx3"
    except Exception as exc:
        print(f"pyttsx3 offline synthesis failed: {exc}. Generating labeled test tone.")
        nchannels = 1
        sampwidth = 2
        nframes = int(duration_sec * sample_rate)
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, "NONE", "not compressed"))
            for index in range(nframes):
                value = int(16000.0 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
                wav_file.writeframesraw(struct.pack("<h", value))
        print(f"Generated labeled test-tone fallback at {output_path}")
        return "beep"


def synthesize_minimax(
    text: str,
    output_path: str,
    *,
    voice_id: str | None = None,
    model: str | None = None,
    language_boost: str = "auto",
    timeout_sec: float = 60.0,
) -> dict[str, Any]:
    """Synthesize a real WAV through MiniMax and fail hard on any API/audio error."""
    if not config.MINIMAX_API_KEY or config.MINIMAX_API_KEY == "your_minimax_api_key_here":
        raise RuntimeError("MINIMAX_API_KEY is not configured; strict speech generation cannot continue")

    voice_id = voice_id or config.MINIMAX_VOICE_ID
    model = model or config.MINIMAX_MODEL_ID
    group_query = f"?GroupId={config.MINIMAX_GROUP_ID}" if config.MINIMAX_GROUP_ID else ""
    endpoint = f"https://api.minimax.io/v1/t2a_v2{group_query}"
    request_body = {
        "model": model,
        "text": text,
        "stream": False,
        "language_boost": language_boost,
        "output_format": "hex",
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "format": "wav",
            "sample_rate": 32000,
            "bitrate": 128000,
            "channel": 1,
        },
    }
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {config.MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        },
        json=request_body,
        timeout=timeout_sec,
    )
    response.raise_for_status()
    try:
        response_body = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("MiniMax returned a non-JSON response for hex output") from exc

    base_resp = response_body.get("base_resp") or {}
    if int(base_resp.get("status_code", -1)) != 0:
        raise RuntimeError(
            f"MiniMax TTS failed with status {base_resp.get('status_code')}: "
            f"{base_resp.get('status_msg', 'unknown error')}"
        )
    audio_hex = str((response_body.get("data") or {}).get("audio") or "")
    if not audio_hex:
        raise RuntimeError("MiniMax returned success without audio data")
    try:
        audio_bytes = bytes.fromhex(audio_hex)
    except ValueError as exc:
        raise RuntimeError("MiniMax audio payload was not valid hexadecimal") from exc
    if not (audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE"):
        raise RuntimeError("MiniMax output is not a RIFF/WAVE file")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(audio_bytes)
    try:
        with wave.open(str(temporary), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            if frames <= 0 or sample_rate <= 0:
                raise RuntimeError("MiniMax WAV has no playable frames")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)

    evidence = {
        "provider": "minimax",
        "endpoint": "https://api.minimax.io/v1/t2a_v2",
        "model": model,
        "voice_id": voice_id,
        "language_boost": language_boost,
        "http_status": response.status_code,
        "api_status_code": int(base_resp.get("status_code", -1)),
        "api_status_message": base_resp.get("status_msg"),
        "trace_id": response_body.get("trace_id"),
        "output_path": destination.as_posix(),
        "bytes": len(audio_bytes),
        "frames": frames,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "duration_sec": frames / sample_rate,
        "extra_info": response_body.get("extra_info") or {},
    }
    print(
        f"MiniMax TTS success: path={destination.as_posix()} voice={voice_id} "
        f"duration={evidence['duration_sec']:.2f}s trace_id={evidence['trace_id']}"
    )
    return evidence


def synthesize_windows_sapi(
    text: str,
    output_path: str,
    *,
    voice_id: str = "Microsoft David Desktop",
    rate: int = 0,
    volume: int = 100,
) -> dict[str, Any]:
    """Create real spoken WAV audio with Windows System.Speech.

    The transcript is supplied on stdin, avoiding shell interpolation of attack text.
    This path is deterministic, offline, and never produces the legacy sine fallback.
    """
    if not -10 <= rate <= 10:
        raise ValueError("Windows SAPI rate must be between -10 and 10")
    if not 0 <= volume <= 100:
        raise ValueError("Windows SAPI volume must be between 0 and 100")

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    powershell = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$text = [Console]::In.ReadToEnd()
$OutputPath = $env:AGENTIMMUNE_TTS_OUTPUT
$VoiceName = $env:AGENTIMMUNE_TTS_VOICE
$Rate = [int]$env:AGENTIMMUNE_TTS_RATE
$Volume = [int]$env:AGENTIMMUNE_TTS_VOLUME
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speaker.SelectVoice($VoiceName)
    $speaker.Rate = $Rate
    $speaker.Volume = $Volume
    $speaker.SetOutputToWaveFile($OutputPath)
    $speaker.Speak($text)
} finally {
    $speaker.Dispose()
}
""".strip()
    command = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        powershell,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "AGENTIMMUNE_TTS_OUTPUT": str(destination),
            "AGENTIMMUNE_TTS_VOICE": voice_id,
            "AGENTIMMUNE_TTS_RATE": str(rate),
            "AGENTIMMUNE_TTS_VOLUME": str(volume),
        }
    )
    completed = subprocess.run(
        command,
        input=text,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Windows System.Speech failed: {completed.stderr.strip()}")
    if not destination.exists():
        raise RuntimeError("Windows System.Speech returned success without creating a WAV")
    with wave.open(str(destination), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
    if frames <= 0 or sample_rate <= 0:
        raise RuntimeError("Windows System.Speech WAV has no playable frames")

    evidence = {
        "provider": "windows_system_speech",
        "tool": "System.Speech.Synthesis.SpeechSynthesizer",
        "command": (
            "powershell -NoProfile -NonInteractive -Command "
            "<SpeechSynthesizer reads transcript from stdin and writes OutputPath>"
        ),
        "voice_id": voice_id,
        "rate": rate,
        "volume": volume,
        "output_path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "frames": frames,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "duration_sec": frames / sample_rate,
    }
    print(
        f"Windows System.Speech success: path={destination.as_posix()} voice={voice_id} "
        f"duration={evidence['duration_sec']:.2f}s"
    )
    return evidence


def text_to_speech(
    text: str,
    output_path: str,
    voice_id: str | None = None,
    *,
    strict: bool = False,
) -> str:
    """Synthesize speech, optionally prohibiting every local/tone fallback."""
    try:
        synthesize_minimax(text, output_path, voice_id=voice_id)
        return "minimax"
    except Exception as exc:
        if strict:
            raise
        print(f"MiniMax TTS unavailable ({exc}); attempting the labeled local fallback.")
        return generate_mock_wav(output_path, text=text)
