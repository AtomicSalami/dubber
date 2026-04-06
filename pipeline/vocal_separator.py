"""
pipeline/vocal_separator.py — Vocal isolation using Demucs.

Strips background music and noise from the original audio so that
reference clips extracted for XTTS voice cloning contain clean speech only.

Results are cached per video_id in processing/<video_id>_vocals/ so
re-runs don't pay the Demucs cost again.

Falls back to the original audio on failure (Demucs not installed, etc.).
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from config import TMP_DIR, PROCESSING_DIR


def separate_vocals(
    video_path: str | Path,
    video_id: str,
    processing_dir: Path = PROCESSING_DIR,
) -> str:
    """
    Run Demucs htdemucs (two-stem: vocals + no_vocals) on the video's audio.

    Args:
        video_path:     Source video file.
        video_id:       Stable ID used for cache directory naming.
        processing_dir: Where to cache the separated vocals.

    Returns:
        Path to the vocals.wav file (or original audio WAV on failure).
    """
    vocals_dir = processing_dir / f"{video_id}_vocals"

    # ── Cache check ───────────────────────────────────────────────────────────
    cached = list(vocals_dir.rglob("vocals.wav"))
    if cached:
        print(f"  [{video_id}] Vocals already separated, reusing: {cached[0]}")
        return str(cached[0])

    # ── Extract stereo 44.1kHz (Demucs works best with this) ─────────────────
    raw_audio = str(TMP_DIR / f"{video_id}_raw.wav")
    os.system(
        f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le '
        f'-ar 44100 -ac 2 "{raw_audio}" -loglevel error'
    )

    print(f"  [{video_id}] Separating vocals with Demucs...")
    print(f"             (first run downloads ~80MB model, takes 2-5 min)")

    result = subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "--out", str(vocals_dir),
            raw_audio,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  [{video_id}] WARNING: Demucs failed — using original audio instead.")
        print(f"             stderr: {result.stderr[-400:]}")
        return raw_audio

    found = list(vocals_dir.rglob("vocals.wav"))
    if found:
        print(f"  [{video_id}] Vocals separated -> {found[0]}")
        return str(found[0])

    print(f"  [{video_id}] WARNING: vocals.wav not found — using original audio.")
    return raw_audio