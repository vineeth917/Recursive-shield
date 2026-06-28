from __future__ import annotations

import shutil
import struct
import subprocess
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "artifacts" / "audio"
SCREENSHOT_DIR = ROOT / "artifacts" / "screenshots"
TRANSCRIPT_DIR = ROOT / "fixtures" / "task_a_handoff" / "transcripts"


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


class Canvas:
    def __init__(self, width: int = 1280, height: int = 760, background: tuple[int, int, int] = (247, 248, 250)) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(background * width * height)

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)
        for py in range(y0, y1):
            offset = (py * self.width + x0) * 3
            self.pixels[offset : offset + (x1 - x0) * 3] = bytes(color) * (x1 - x0)

    def save(self, path: Path) -> None:
        rows = []
        for y in range(self.height):
            row = self.pixels[y * self.width * 3 : (y + 1) * self.width * 3]
            rows.append(b"\x00" + bytes(row))
        raw = b"".join(rows)
        data = b"\x89PNG\r\n\x1a\n"
        data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        data += png_chunk(b"IDAT", zlib.compress(raw, level=6))
        data += png_chunk(b"IEND", b"")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def make_brokerage_screen(path: Path, variant: str, danger: bool = False) -> None:
    canvas = Canvas()
    navy = (24, 32, 48)
    green = (23, 132, 80)
    red = (190, 42, 42)
    blue = (38, 92, 170)
    gray = (226, 231, 237)
    white = (255, 255, 255)
    ink = (70, 78, 90)
    accent = red if danger else green

    canvas.rect(0, 0, 1280, 70, navy)
    canvas.rect(24, 20, 210, 30, accent)
    canvas.rect(260, 24, 90, 22, (101, 116, 139))
    canvas.rect(370, 24, 90, 22, (101, 116, 139))
    canvas.rect(490, 24, 120, 22, (101, 116, 139))

    canvas.rect(28, 100, 380, 570, white)
    canvas.rect(48, 125, 180, 24, ink)
    for idx, width in enumerate([300, 240, 320, 210, 260]):
        canvas.rect(48, 180 + idx * 58, width, 20, gray)
        canvas.rect(48, 207 + idx * 58, width // 2, 14, (204, 211, 220))

    canvas.rect(442, 100, 380, 570, white)
    canvas.rect(462, 125, 170, 24, ink)
    canvas.rect(462, 185, 250, 38, gray)
    canvas.rect(462, 250, 250, 38, gray)
    canvas.rect(462, 315, 250, 38, gray)
    canvas.rect(462, 395, 300, 54, accent)
    canvas.rect(462, 485, 250, 38, red if danger else blue)

    canvas.rect(856, 100, 396, 570, white)
    canvas.rect(876, 125, 210, 24, ink)
    canvas.rect(876, 180, 320, 110, (240, 244, 248))
    canvas.rect(876, 315, 320, 110, (240, 244, 248))
    canvas.rect(876, 450, 320, 110, (240, 244, 248))

    code = sum(ord(char) for char in variant)
    for i in range(44):
        x = 54 + (i % 22) * 14
        y = 610 + (i // 22) * 24
        color = accent if (code + i) % 3 == 0 else (148, 163, 184)
        canvas.rect(x, y, 9, 16, color)

    if danger:
        canvas.rect(0, 700, 1280, 60, red)
        canvas.rect(24, 718, 540, 24, (255, 255, 255))
    else:
        canvas.rect(0, 700, 1280, 60, green)
        canvas.rect(24, 718, 420, 24, (255, 255, 255))

    canvas.save(path)


def render_wav(transcript_name: str, output_name: str) -> None:
    say = shutil.which("say")
    afconvert = shutil.which("afconvert")
    if say is None or afconvert is None:
        raise RuntimeError("Synthetic WAV generation requires macOS 'say' and 'afconvert'.")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / transcript_name
    aiff_path = AUDIO_DIR / output_name.replace(".wav", ".aiff")
    wav_path = AUDIO_DIR / output_name
    subprocess.run([say, "-f", str(transcript_path), "-o", str(aiff_path)], check=True)
    subprocess.run([afconvert, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff_path), str(wav_path)], check=True)


def main() -> None:
    render_wav("fomc_clean_sample_transcript.txt", "fomc_clean_sample.wav")
    render_wav("fomc_l1_ad_break_splice_sample_transcript.txt", "fomc_l1_ad_break_splice_sample.wav")

    screenshot_specs = {
        "clean_fed_before_order.png": ("clean-before-order", False),
        "clean_fed_order_ticket.png": ("clean-order-ticket", False),
        "clean_fed_confirm.png": ("clean-confirm", False),
        "l1_before_exfil.png": ("l1-before-exfil", True),
        "l1_notes_exfil.png": ("l1-notes-exfil", True),
        "l1_order_ticket.png": ("l1-order-ticket", True),
        "l1_confirm_forbidden.png": ("l1-confirm-forbidden", True),
    }
    for filename, (variant, danger) in screenshot_specs.items():
        make_brokerage_screen(SCREENSHOT_DIR / filename, variant, danger)

    print(f"wrote {AUDIO_DIR / 'fomc_clean_sample.wav'}")
    print(f"wrote {AUDIO_DIR / 'fomc_l1_ad_break_splice_sample.wav'}")
    print(f"wrote screenshots under {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
