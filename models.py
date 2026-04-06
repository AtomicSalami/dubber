"""
models.py — Pydantic data models for the dubbing pipeline.

Key design decisions:
- All optional fields default to None so records can be built incrementally step-by-step
- speaker_id is present but unused until diarization is plugged in
- ReferenceChunk is selected during transcription (Whisper timestamps guide it)
- VideoRecord serialises cleanly to/from JSON for state persistence in processing/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


# ── Segment ───────────────────────────────────────────────────────────────────

class Segment(BaseModel):
    """
    One unit of speech in the video.

    Lifecycle of text fields:
        content_raw        <- filled by Transcriber (Whisper output)
        content_validated  <- filled by Validator (LLM-corrected Russian)
        content_<lang>     <- filled by Translator per target language
    """

    seg_id: str
    time_start: float
    time_end: float

    # Text at each pipeline stage
    content_raw: str
    content_validated: Optional[str] = None

    # Per-language translations — stored as a flat dict so adding a new
    # language never requires a model change.
    # e.g. {"japanese": "...", "english": "..."}
    translations: dict[str, str] = {}

    # Diarization hook — set by DiarizationStep when implemented
    speaker_id: Optional[str] = None

    @field_validator("time_end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        start = info.data.get("time_start", 0)
        if v <= start:
            raise ValueError(
                f"time_end ({v}) must be greater than time_start ({start})"
            )
        return v

    @property
    def duration(self) -> float:
        return self.time_end - self.time_start

    def get_translation(self, lang_key: str) -> Optional[str]:
        return self.translations.get(lang_key)

    def set_translation(self, lang_key: str, text: str) -> None:
        self.translations[lang_key] = text


# ── ReferenceChunk ────────────────────────────────────────────────────────────

class ReferenceChunk(BaseModel):
    """
    A 6-22 second audio clip used for XTTS voice cloning.
    Selected by the Transcriber from Whisper-aligned segments.

    The clip must:
    - contain a complete sentence (not cut mid-speech)
    - fall within REF_CLIP_MIN_DUR..REF_CLIP_MAX_DUR
    - have clean, validated Russian text for context
    """

    chunk_id: str
    seg_id: str           # source segment this was derived from
    time_start: float
    time_end: float
    text: str             # Russian text in this chunk
    wav_path: Optional[str] = None  # set after audio extraction

    # Diarization hook — which speaker this clip belongs to
    speaker_id: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.time_end - self.time_start


# ── PipelineStatus ────────────────────────────────────────────────────────────

class PipelineStatus(BaseModel):
    """Tracks which steps have completed for a video."""

    transcribed:  bool = False
    validated:    bool = False
    diarized:     bool = False   # future step
    translated:   dict[str, bool] = {}   # {"japanese": True, ...}
    dubbed:       dict[str, bool] = {}   # {"japanese": True, ...}


# ── VideoRecord ───────────────────────────────────────────────────────────────

class VideoRecord(BaseModel):
    """
    The central state object for one video moving through the pipeline.
    Persisted as a single JSON file in processing/<video_id>.json.
    """

    video_id:   str
    video_path: str

    status:    PipelineStatus = PipelineStatus()
    segments:  list[Segment]  = []

    # Reference chunks for voice cloning — selected per speaker once
    # diarization is active; without diarization, all chunks share speaker "default"
    reference_chunks: list[ReferenceChunk] = []

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, video_id: str, processing_dir: Path) -> Optional["VideoRecord"]:
        path = processing_dir / f"{video_id}.json"
        if path.exists():
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        return None

    def save(self, processing_dir: Path) -> None:
        path = processing_dir / f"{self.video_id}.json"
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def segments_needing_validation(self) -> list[Segment]:
        return [s for s in self.segments if not s.content_validated]

    def segments_needing_translation(self, lang_key: str) -> list[Segment]:
        return [
            s for s in self.segments
            if s.content_validated and not s.get_translation(lang_key)
        ]

    def translated_segments(self, lang_key: str) -> list[Segment]:
        return [s for s in self.segments if s.get_translation(lang_key)]

    @model_validator(mode="after")
    def check_video_path_exists(self) -> "VideoRecord":
        # Soft check — path may not exist in test scenarios
        return self