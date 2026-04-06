"""
config.py — Central configuration for the dubbing pipeline.
All constants, environment variables, and directory paths live here.
"""

import os
import torch
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Directories ───────────────────────────────────────────────────────────────

TMP_DIR        = Path(tempfile.gettempdir())
VIDEO_DIR      = Path("video")
PROCESSING_DIR = Path("processing")
OUTPUT_DIR     = Path("output")

for _d in [VIDEO_DIR, PROCESSING_DIR, OUTPUT_DIR]:
    _d.mkdir(exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model identifiers ─────────────────────────────────────────────────────────

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-medium")
CLAUDE_MODEL  = "claude-sonnet-4-20250514"
XTTS_MODEL    = "tts_models/multilingual/multi-dataset/xtts_v2"

# ── Pipeline tunables ─────────────────────────────────────────────────────────

VALIDATE_BATCH_SIZE = 5     # segments per Claude validation call
MAX_SEGMENT_DURATION = 15.0  # seconds — segments longer than this get semantically split

# Reference clip selection (for XTTS voice cloning)
REF_CLIP_MIN_DUR = 6.0   # seconds
REF_CLIP_MAX_DUR = 22.0  # seconds
REF_CLIP_COUNT   = 3     # how many reference clips to extract

# XTTS character limit per synthesis call (hard limit is 71; use 60 to be safe)
TTS_MAX_CHARS = 60

# F0 transfer clamping
F0_MAX_SEMITONE_SHIFT = 4   # ± semitones
F0_RANGE_FACTOR_MIN   = 0.5
F0_RANGE_FACTOR_MAX   = 2.0

# ── File handling ─────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

# ── Supported languages ───────────────────────────────────────────────────────

SUPPORTED_LANGUAGES: dict[str, dict] = {
    "english": {
        "name":      "English",
        "tts_code":  "en",
        "whisper":   "english",
    },
    "japanese": {
        "name":      "Japanese",
        "tts_code":  "ja",
        "whisper":   "japanese",
        # Extra TTS instructions injected into the translation system prompt
        "tts_notes": (
            " CRITICAL: Output MUST be written entirely in hiragana and katakana only — "
            "NO kanji, NO Arabic numerals, NO Roman letters. "
            "Write all numbers as Japanese reading words in hiragana "
            "(e.g. 4ヶ月 -> よんかげつ, 2.5 -> にてんご, 1000 -> せん). "
            "Use katakana only for loanwords (e.g. チーズ is fine as katakana). "
            "Everything else must be hiragana."
        ),
    },
}

def get_language_config(lang_key: str) -> dict:
    if lang_key not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{lang_key}'. "
            f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    return SUPPORTED_LANGUAGES[lang_key]


# ── API helpers ───────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env or environment.")
    return key