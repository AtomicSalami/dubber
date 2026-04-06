"""
pipeline/diarizer.py — Step (future): Speaker Diarization via WhisperX.

This is a fully-typed stub that defines the interface for diarization.
When you're ready to implement it, everything the rest of the pipeline needs
is already wired:
  - Segment.speaker_id (set to None until this step runs)
  - ReferenceChunk.speaker_id (same)
  - PipelineStatus.diarized flag
  - VideoRecord.reference_chunks list (re-populated per speaker here)

WhisperX was chosen over pyannote because:
  - Performs transcription + diarization in one pass (fewer models to load)
  - Free and open-source (no HuggingFace access token required for the base)
  - Word-level timestamps → more accurate speaker boundary alignment

─────────────────────────────────────────────────────────────────────────────
Implementation guide (when you're ready):

  pip install whisperx

  import whisperx

  # 1. Load model
  model = whisperx.load_model("large-v2", device, compute_type="float16")

  # 2. Transcribe with word timestamps
  audio  = whisperx.load_audio(audio_path)
  result = model.transcribe(audio, batch_size=16)

  # 3. Align word timestamps
  model_a, metadata = whisperx.load_align_model(language_code="ru", device=device)
  result = whisperx.align(result["segments"], model_a, metadata, audio, device)

  # 4. Diarize
  diarize_model  = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
  diarize_segs   = diarize_model(audio)
  result         = whisperx.assign_word_speakers(diarize_segs, result)

  # 5. Map WhisperX speaker labels onto our Segment objects by time overlap
  for seg in record.segments:
      seg.speaker_id = _find_speaker(result["segments"], seg.time_start, seg.time_end)

  # 6. Re-run select_reference_chunks() once per speaker_id and
  #    rebuild record.reference_chunks as a flat list
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

from pipeline.base import PipelineStep
from models import VideoRecord
from config import PROCESSING_DIR


class DiarizationStep(PipelineStep):
    """
    Stub for WhisperX speaker diarization.

    Plugging this step into main.py:

        diarizer = DiarizationStep(hf_token="hf_...", processing_dir=PROCESSING_DIR)
        steps    = [transcriber, validator, diarizer, translator, dubber]

    The step sets speaker_id on every Segment and ReferenceChunk,
    then sets record.status.diarized = True so downstream steps can
    behave differently per speaker (e.g. separate voice cloning per speaker).
    """

    name = "diarize"

    def __init__(
        self,
        hf_token: str | None = None,
        processing_dir: Path = PROCESSING_DIR,
    ):
        super().__init__(processing_dir)
        self.hf_token = hf_token

    def is_complete(self, record: VideoRecord) -> bool:
        return record.status.diarized

    def run(self, record: VideoRecord) -> VideoRecord:
        raise NotImplementedError(
            "DiarizationStep is a stub. "
            "See the implementation guide in pipeline/diarizer.py."
        )