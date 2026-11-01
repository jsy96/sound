#!/usr/bin/env python3
"""Generate bilingual voice-design candidates with macOS built-in voices.

This is a dependency-free acoustic baseline for the paper's human-in-the-loop
workflow. It does not claim to reproduce the paper's learned speaker space.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


VOICE_PAIRS = {
    "eddy": {
        "zh": "Eddy (中文（中国大陆）)",
        "en": "Eddy (英语（美国）)",
    },
    "flo": {
        "zh": "Flo (中文（中国大陆）)",
        "en": "Flo (英语（美国）)",
    },
}


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def pitch_filter(semitones: float) -> tuple[str, float]:
    ratio = 2 ** (semitones / 12.0)
    # asetrate changes pitch and duration; atempo restores the original tempo.
    return f"asetrate=44100*{ratio:.8f},aresample=44100,atempo={1/ratio:.8f}", ratio


def make_candidate(
    zh_text: str,
    en_text: str,
    out_path: Path,
    pair: dict[str, str],
    rate: int,
    pitch: float,
    brightness: float,
    workdir: Path,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    zh_aiff = workdir / "zh.aiff"
    en_aiff = workdir / "en.aiff"
    run(["say", "-v", pair["zh"], "-r", str(rate), "-o", str(zh_aiff), zh_text])
    run(["say", "-v", pair["en"], "-r", str(rate), "-o", str(en_aiff), en_text])

    pitch, _ = pitch_filter(pitch)
    # Positive brightness emphasizes upper mids; negative values darken them.
    chain = (
        f"{pitch},highpass=f=70,lowpass=f=14000,"
        f"equalizer=f=1800:t=q:w=1:g={brightness:.2f}"
    )
    filter_complex = (
        f"[0:a]{chain}[zh];"
        f"[1:a]{chain}[en];"
        "[zh][en]concat=n=2:v=0:a=1,loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(zh_aiff),
            "-i",
            str(en_aiff),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ar",
            "44100",
            "-ac",
            "1",
            str(out_path),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chinese",
        default="你好，我想设计一种自然、温暖、清晰的声音。",
        help="Chinese probe sentence",
    )
    parser.add_argument(
        "--english",
        default="Hello, I am designing a voice that sounds natural, warm, and clear.",
        help="English probe sentence",
    )
    parser.add_argument("--outdir", type=Path, default=Path("candidates"))
    args = parser.parse_args()

    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        raise SystemExit("This prototype requires macOS 'say' and ffmpeg.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []
    candidates = 0
    with tempfile.TemporaryDirectory(prefix="voice-design-") as temp:
        workroot = Path(temp)
        for base, pair in VOICE_PAIRS.items():
            for pitch in (-3.0, 0.0, 3.0):
                for brightness in (-3.0, 0.0, 3.0):
                    rate = 180
                    stem = f"{base}_p{pitch:+.0f}_b{brightness:+.0f}".replace("+", "plus").replace("-", "minus")
                    out_path = args.outdir / f"{stem}.wav"
                    make_candidate(
                        args.chinese,
                        args.english,
                        out_path,
                        pair,
                        rate,
                        pitch,
                        brightness,
                        workroot / stem,
                    )
                    metadata.append(
                        {
                            "file": out_path.name,
                            "base_voice": base,
                            "pitch_semitones": pitch,
                            "brightness_db": brightness,
                            "rate_wpm": rate,
                        }
                    )
                    candidates += 1

    (args.outdir / "metadata.json").write_text(
        json.dumps(
            {
                "chinese": args.chinese,
                "english": args.english,
                "candidates": metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    for item in metadata:
        rows.append(
            "<tr>"
            f"<td>{item['file']}</td>"
            f"<td>{item['base_voice']}</td>"
            f"<td>{item['pitch_semitones']:+.0f} st</td>"
            f"<td>{item['brightness_db']:+.0f} dB</td>"
            f"<td><audio controls preload='none' src='{item['file']}'></audio></td>"
            "</tr>"
        )
    html = """<!doctype html>
<meta charset="utf-8">
<title>Voice design candidates</title>
<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:.55rem;text-align:left}audio{width:280px}</style>
<h1>Bilingual voice-design candidates</h1>
<p>Listen, compare, and record the filenames you prefer. Each file contains the Chinese probe followed by the English probe.</p>
<table><thead><tr><th>File</th><th>Base</th><th>Pitch</th><th>Brightness</th><th>Player</th></tr></thead><tbody>
""" + "\n".join(rows) + "\n</tbody></table>\n"
    (args.outdir / "index.html").write_text(html, encoding="utf-8")
    print(f"Generated {candidates} candidates in {args.outdir}")


if __name__ == "__main__":
    main()
