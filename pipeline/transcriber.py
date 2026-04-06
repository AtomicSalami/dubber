"""
pipeline/transcriber.py — Step 1: Transcription + Reference Chunk Selection.

Responsibilities:
  1. Extract audio from the video file
  2. Run Whisper to get timestamped segments
  3. Clean hallucinations from Whisper output
  4. Semantically split long segments at sentence boundaries
  5. Select reference chunks for XTTS voice cloning
     (Whisper timestamps guide chunk boundaries so clips are never mid-sentence)

WhisperX diarization hook:
  The `speaker_id` field on Segment and ReferenceChunk is set to None here.
  When a DiarizationStep is added, it will fill speaker_id on existing segments
  without requiring any changes to this file.
"""

from __future__ import annotations

import re
import uuid
import soundfile as sf
from pathlib import Path

from pipeline.base import PipelineStep
from models import VideoRecord, Segment, ReferenceChunk
from utils.audio import extract_audio
from config import (
    TMP_DIR, PROCESSING_DIR, WHISPER_MODEL, DEVICE,
    MAX_SEGMENT_DURATION, REF_CLIP_MIN_DUR, REF_CLIP_MAX_DUR, REF_CLIP_COUNT,
)


# ── Hallucination cleaning ─────────────────────────────────────────────────────

_REPEAT_PATTERN = re.compile(r"(.{2,20})\1{4,}", re.IGNORECASE)


def clean_hallucinations(text: str) -> str:
    """
    Remove Whisper hallucination artefacts: repeated n-gram loops.
    E.g. "да да да да да да" or looped phrases Whisper outputs on silence.
    """
    cleaned = _REPEAT_PATTERN.sub("", text)
    return re.sub(r" {2,}", " ", cleaned).strip()


# ── Semantic splitting ────────────────────────────────────────────────────────

def split_semantically(seg: Segment, max_dur: float = MAX_SEGMENT_DURATION) -> list[Segment]:
    """
    If a segment is longer than max_dur, split it at sentence boundaries
    and distribute the time proportionally to character counts.

    Ensures no clip is ever cut mid-sentence, which also benefits
    reference chunk selection downstream.
    """
    if seg.duration <= max_dur:
        return [seg]

    sentences = re.split(r"(?<=[.!?])\s+", seg.content_raw.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return [seg]

    total_chars = sum(len(s) for s in sentences)
    chunks: list[Segment] = []
    cur = seg.time_start

    for i, sent in enumerate(sentences):
        end = cur + seg.duration * (len(sent) / total_chars)
        if i == len(sentences) - 1:
            end = seg.time_end

        chunks.append(Segment(
            seg_id=f"{seg.seg_id}_{i}",
            time_start=round(cur, 3),
            time_end=round(end, 3),
            content_raw=sent,
            speaker_id=seg.speaker_id,
        ))
        cur = end

    return chunks


# ── Reference chunk selection ─────────────────────────────────────────────────

def select_reference_chunks(
    segments: list[Segment],
    n: int = REF_CLIP_COUNT,
    min_dur: float = REF_CLIP_MIN_DUR,
    max_dur: float = REF_CLIP_MAX_DUR,
) -> list[ReferenceChunk]:
    """
    Pick the best N segments for XTTS voice cloning reference clips.

    Selection criteria:
    - Must have content_raw (validated text preferred but not required here —
      validation happens in the next step; we use raw text as placeholder)
    - Duration between min_dur and max_dur seconds
    - Prefer longer segments (more voice data → better voice clone)
    - Sentence boundaries are already guaranteed by split_semantically,
      so no clip will be cut mid-word

    When diarization is active, this function should be called once per
    speaker, filtering segments by speaker_id. The stub for that lives
    in pipeline/diarizer.py.
    """
    candidates = [
        s for s in segments
        if min_dur <= s.duration <= max_dur
    ]
    candidates.sort(key=lambda s: s.duration, reverse=True)
    selected = candidates[:n]

    return [
        ReferenceChunk(
            chunk_id=f"ref_{uuid.uuid4().hex[:8]}",
            seg_id=seg.seg_id,
            time_start=seg.time_start,
            time_end=seg.time_end,
            text=seg.content_raw,
            speaker_id=seg.speaker_id,
        )
        for seg in selected
    ]


# ── Transcriber step ──────────────────────────────────────────────────────────

class TranscriberStep(PipelineStep):
    """
    Runs Whisper on the video's audio track to produce timestamped Segments
    and selects ReferenceChunks for voice cloning.

    The whisper_pipe is injected at construction time so the heavy model load
    happens once in main.py and is shared across all videos.
    """

    name = "transcribe"

    def __init__(self, whisper_pipe, processing_dir: Path = PROCESSING_DIR):
        super().__init__(processing_dir)
        self.whisper_pipe = whisper_pipe

    def is_complete(self, record: VideoRecord) -> bool:
        return record.status.transcribed and bool(record.segments)

    def run(self, record: VideoRecord) -> VideoRecord:
        print(f"  [{record.video_id}] Transcribing {record.video_path}...")

        # ── 1. Extract mono 16kHz audio ──────────────────────────────────────
        audio_path = str(TMP_DIR / f"{record.video_id}_whisper.wav")
        extract_audio(record.video_path, audio_path, sample_rate=16000, channels=1)

        audio_array, sr = sf.read(audio_path, dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # ── 2. Run Whisper ───────────────────────────────────────────────────
        result = self.whisper_pipe(
            {"array": audio_array, "sampling_rate": sr},
            generate_kwargs={"language": "russian", "task": "transcribe"},
            chunk_length_s=30,
            stride_length_s=5,
        )

        # ── 3. Build raw segments ────────────────────────────────────────────
        raw_segments: list[Segment] = []
        for i, chunk in enumerate(result.get("chunks", [])):
            text = chunk["text"].strip()
            if not text:
                continue

            text = clean_hallucinations(text)
            if not text:
                continue

            ts    = chunk.get("timestamp", (0, 0))
            start = ts[0] if ts[0] is not None else 0
            end   = ts[1] if ts[1] is not None else start + 3.0

            raw_segments.append(Segment(
                seg_id=f"{record.video_id}_{i:04d}",
                time_start=round(start, 3),
                time_end=round(end, 3),
                content_raw=text,
            ))

        # ── 4. Semantic split on long segments ───────────────────────────────
        segments: list[Segment] = []
        for seg in raw_segments:
            segments.extend(split_semantically(seg))

        # ── 5. Select reference chunks for voice cloning ─────────────────────
        ref_chunks = select_reference_chunks(segments)
        if not ref_chunks:
            print(
                f"  [{record.video_id}] WARNING: No segments in "
                f"{REF_CLIP_MIN_DUR}-{REF_CLIP_MAX_DUR}s range for reference clips. "
                f"Falling back to first {REF_CLIP_COUNT} segments."
            )
            ref_chunks = select_reference_chunks(
                segments, min_dur=0, max_dur=9999
            )[:REF_CLIP_COUNT]

        # ── 6. Persist ───────────────────────────────────────────────────────
        record.segments         = segments
        record.reference_chunks = ref_chunks
        record.status.transcribed = True
        record.save(self.processing_dir)

        print(
            f"  [{record.video_id}] {len(segments)} segments, "
            f"{len(ref_chunks)} reference chunks selected."
        )
        return record