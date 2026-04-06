"""
pipeline/translator.py — Step 3: Translation.

Translates validated Russian segments into a target language using Claude.
Translation is dubbing-aware:
  - Must fit within the time slot when spoken aloud
  - High-energy source = short punchy output
  - Never word-for-word; must sound natural as spoken language
  - Language-specific constraints (e.g. Japanese hiragana-only for TTS)
"""

from __future__ import annotations

from pathlib import Path

from pipeline.base import PipelineStep
from models import VideoRecord, Segment
from utils.claude import claude_call, parse_json_response
from config import PROCESSING_DIR, get_language_config


def _build_system_prompt(lang_key: str) -> str:
    cfg       = get_language_config(lang_key)
    lang_name = cfg["name"]
    base = (
        f"You are a professional dubbing translator specializing in Russian to {lang_name}. "
        f"Translations will be spoken aloud by a TTS voice, so they MUST: "
        f"(1) match the energy and mood exactly — high energy = short punchy sentences, "
        f"(2) fit within the time shown in parentheses when spoken aloud, "
        f"(3) sound completely natural as spoken {lang_name} — never word-for-word, "
        f"(4) be concise — err shorter not longer. "
        f"Return ONLY a JSON array of strings, one per segment, no markdown, no explanation."
    )
    extra = cfg.get("tts_notes", "")
    return base + extra


class TranslatorStep(PipelineStep):
    """
    Translates all validated segments into target_lang using Claude.

    Sends all untranslated segments for a video in a single API call
    (the list can be large; Claude handles it well and it keeps latency low).
    Falls back to chunked calls if needed in future.
    """

    name = "translate"

    def __init__(
        self,
        api_key: str,
        target_lang: str,
        processing_dir: Path = PROCESSING_DIR,
    ):
        super().__init__(processing_dir)
        self.api_key     = api_key
        self.target_lang = target_lang
        self._system     = _build_system_prompt(target_lang)

    def is_complete(self, record: VideoRecord) -> bool:
        return record.status.translated.get(self.target_lang, False)

    def run(self, record: VideoRecord) -> VideoRecord:
        lang_name = get_language_config(self.target_lang)["name"]
        todo      = record.segments_needing_translation(self.target_lang)

        if not todo:
            print(
                f"  [{record.video_id}] All segments already translated "
                f"to {lang_name} (or nothing to translate)."
            )
            record.status.translated[self.target_lang] = True
            record.save(self.processing_dir)
            return record

        print(f"  [{record.video_id}] Translating {len(todo)} segments to {lang_name}...")

        numbered = "\n".join(
            f"[{i+1}] ({s.duration:.1f}s) {s.content_validated}"
            for i, s in enumerate(todo)
        )
        user_prompt = (
            f"Translate these {len(todo)} Russian video segments to {lang_name}.\n\n"
            f"{numbered}\n\n"
            f"Return a JSON array with exactly {len(todo)} strings."
        )

        raw          = claude_call(self.api_key, self._system, user_prompt, max_tokens=4096)
        translations = parse_json_response(raw)

        if len(translations) != len(todo):
            raise RuntimeError(
                f"[{record.video_id}] Claude returned {len(translations)} translations "
                f"for {len(todo)} segments"
            )

        for seg, tr in zip(todo, translations):
            seg.set_translation(self.target_lang, tr.strip())

        record.status.translated[self.target_lang] = True
        record.save(self.processing_dir)
        print(f"  [{record.video_id}] Translation to {lang_name} complete.")
        return record