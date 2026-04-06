"""
main.py — CLI entry point for the dubbing pipeline.

Usage:
  python main.py --lang japanese              # full pipeline
  python main.py --lang japanese --step transcribe
  python main.py --lang japanese --step validate
  python main.py --lang japanese --step translate
  python main.py --lang japanese --step dub

Steps are idempotent: re-running a completed step is safe and fast.
State is persisted in processing/<video_id>.json between runs.
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", message=".*chunk_length_s.*")
warnings.filterwarnings("ignore", message=".*seq2seq.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")

import torch

from config import (
    DEVICE, PROCESSING_DIR, WHISPER_MODEL, XTTS_MODEL,
    get_api_key, get_language_config,
)
from models import VideoRecord
from utils.video import scan_and_id_videos


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular video dubbing pipeline.")
    parser.add_argument(
        "--lang",
        default="japanese",
        choices=["english", "japanese"],
        help="Target dubbing language (default: japanese)",
    )
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "transcribe", "validate", "translate", "dub"],
        help="Run a specific step only (default: all)",
    )
    return parser


def load_or_init_record(video_id: str, video_path: Path) -> VideoRecord:
    record = VideoRecord.load(video_id, PROCESSING_DIR)
    if record is None:
        record = VideoRecord(video_id=video_id, video_path=str(video_path))
        record.save(PROCESSING_DIR)
    return record


def patch_tortoise_import():
    """
    Fix a broken import in older TTS/Coqui installs that reference a
    removed transformers utility (isin_mps_friendly).
    Safe to call unconditionally — skips silently if not needed.
    """
    import site
    roots = [Path(sys.prefix)] + [Path(p) for p in site.getsitepackages()]
    for root in roots:
        target = root / "TTS" / "tts" / "layers" / "tortoise" / "autoregressive.py"
        if target.exists():
            txt = target.read_text(encoding="utf-8")
            if "isin_mps_friendly" in txt:
                target.write_text(
                    txt.replace(
                        "from transformers.pytorch_utils import isin_mps_friendly as isin",
                        "from torch import isin",
                    ),
                    encoding="utf-8",
                )
            break


def main():
    args    = build_arg_parser().parse_args()
    run_all = args.step == "all"

    # ── Discover videos ───────────────────────────────────────────────────────
    print("\nScanning video folder...")
    videos = scan_and_id_videos()
    if not videos:
        print("No videos found in video/ folder. Drop some in and try again.")
        return
    print(f"Found {len(videos)} video(s): {[vid for vid, _ in videos]}")

    api_key  = get_api_key()
    lang_cfg = get_language_config(args.lang)
    print(f"Target language: {lang_cfg['name']}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Transcription
    # ══════════════════════════════════════════════════════════════════════════
    if run_all or args.step == "transcribe":
        from transformers import pipeline as hf_pipeline
        from pipeline.transcriber import TranscriberStep

        print(f"\nLoading Whisper ({WHISPER_MODEL})...")
        whisper_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            return_timestamps=True,
        )
        step = TranscriberStep(whisper_pipe, PROCESSING_DIR)
        print("\n── Step 1: Transcription ──────────────────────────────────────")
        for video_id, video_path in videos:
            record = load_or_init_record(video_id, video_path)
            step.run_if_needed(record)

        del whisper_pipe
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Validation
    # ══════════════════════════════════════════════════════════════════════════
    if run_all or args.step == "validate":
        from pipeline.validator import ValidatorStep

        step = ValidatorStep(api_key, PROCESSING_DIR)
        print("\n── Step 2: Validation ─────────────────────────────────────────")
        for video_id, video_path in videos:
            record = load_or_init_record(video_id, video_path)
            step.run_if_needed(record)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Translation
    # ══════════════════════════════════════════════════════════════════════════
    if run_all or args.step == "translate":
        from pipeline.translator import TranslatorStep

        step = TranslatorStep(api_key, args.lang, PROCESSING_DIR)
        print(f"\n── Step 3: Translation -> {lang_cfg['name']} ──────────────────")
        for video_id, video_path in videos:
            record = load_or_init_record(video_id, video_path)
            step.run_if_needed(record)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Dubbing
    # ══════════════════════════════════════════════════════════════════════════
    if run_all or args.step == "dub":
        import os
        from TTS.api import TTS as CoquiTTS
        from pipeline.dubber import DubberStep

        patch_tortoise_import()
        os.environ["COQUI_TOS_AGREED"] = "1"

        print(f"\nLoading XTTS v2 ({XTTS_MODEL})...")
        xtts_model = CoquiTTS(model_name=XTTS_MODEL).to(DEVICE)
        print("XTTS v2 loaded.")

        step = DubberStep(xtts_model, args.lang, PROCESSING_DIR)
        print("\n── Step 4: Dubbing ────────────────────────────────────────────")
        for video_id, video_path in videos:
            record = load_or_init_record(video_id, video_path)
            step.run_if_needed(record)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()