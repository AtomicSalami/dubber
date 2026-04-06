"""
pipeline/validator.py — Step 2: LLM Transcript Validation.

Sends batches of raw Whisper segments to Claude for correction:
  - Orthographic errors and mishearings
  - Missing punctuation affecting meaning
  - Obvious word substitutions (similar-sounding words confused by Whisper)
  - Hallucinated or nonsensical text

Also updates ReferenceChunk.text to use the validated (cleaner) text,
which gives XTTS better context about the clip content.
"""

from __future__ import annotations

import time
from pathlib import Path

from pipeline.base import PipelineStep
from models import VideoRecord, Segment
from utils.claude import claude_call, parse_json_response
from config import PROCESSING_DIR, VALIDATE_BATCH_SIZE


_SYSTEM_PROMPT = """You are a Russian language expert and transcription validator.
You will receive batches of Russian speech segments auto-transcribed by Whisper.

Your job is to fix:
1. Orthographic errors (wrong letters, mishearings — e.g. "Рестно" -> "Честно")
2. Missing punctuation that affects meaning
3. Obvious word substitutions (Whisper confused similar-sounding words)
4. Run-on hallucinations or clearly nonsensical text (remove it)

Rules:
- Keep the original meaning and style — do NOT rephrase or improve the text
- If a segment looks correct, return it unchanged
- Return ONLY a JSON array of corrected strings, one per input segment
- No explanations, no markdown, just the JSON array"""


class ValidatorStep(PipelineStep):
    """
    Validates and corrects raw Whisper transcripts using Claude.

    Segments are sent in batches (default: 5 per call) to stay within
    a comfortable token budget while keeping latency reasonable.

    After validation:
    - Each Segment.content_validated is set
    - Matching ReferenceChunk.text fields are updated to the cleaner text
    """

    name = "validate"

    def __init__(self, api_key: str, processing_dir: Path = PROCESSING_DIR):
        super().__init__(processing_dir)
        self.api_key = api_key

    def is_complete(self, record: VideoRecord) -> bool:
        return record.status.validated

    def run(self, record: VideoRecord) -> VideoRecord:
        todo = record.segments_needing_validation()
        if not todo:
            print(f"  [{record.video_id}] All segments already validated.")
            record.status.validated = True
            record.save(self.processing_dir)
            return record

        print(
            f"  [{record.video_id}] Validating {len(todo)} segments "
            f"in batches of {VALIDATE_BATCH_SIZE}..."
        )

        for batch_start in range(0, len(todo), VALIDATE_BATCH_SIZE):
            batch = todo[batch_start : batch_start + VALIDATE_BATCH_SIZE]
            self._validate_batch(record.video_id, batch)

            batch_num = batch_start // VALIDATE_BATCH_SIZE + 1
            done      = min(batch_start + VALIDATE_BATCH_SIZE, len(todo))
            print(f"    Batch {batch_num} done ({done}/{len(todo)} segments)")
            time.sleep(0.5)  # gentle rate-limit

        # Update reference chunk text to use validated (cleaner) versions
        self._refresh_reference_chunk_text(record)

        record.status.validated = True
        record.save(self.processing_dir)
        print(f"  [{record.video_id}] Validation complete.")
        return record

    def _validate_batch(self, video_id: str, batch: list[Segment]) -> None:
        numbered = "\n".join(
            f"[{i+1}] ({s.duration:.1f}s) {s.content_raw}"
            for i, s in enumerate(batch)
        )
        user_prompt = (
            f"Validate and correct these {len(batch)} Russian transcription segments:\n\n"
            f"{numbered}\n\n"
            f"Return a JSON array with exactly {len(batch)} corrected strings."
        )
        raw         = claude_call(self.api_key, _SYSTEM_PROMPT, user_prompt)
        corrections = parse_json_response(raw)

        if len(corrections) != len(batch):
            raise RuntimeError(
                f"[{video_id}] Claude returned {len(corrections)} corrections "
                f"for {len(batch)} segments"
            )

        for seg, corrected in zip(batch, corrections):
            seg.content_validated = corrected.strip()

    def _refresh_reference_chunk_text(self, record: VideoRecord) -> None:
        """
        Update each ReferenceChunk's text field to the validated transcript
        for the segment it was derived from.
        """
        seg_map = {s.seg_id: s for s in record.segments}
        for chunk in record.reference_chunks:
            source_seg = seg_map.get(chunk.seg_id)
            if source_seg and source_seg.content_validated:
                chunk.text = source_seg.content_validated