"""
pipeline/base.py — Abstract base class for every pipeline step.

Adding a new step (e.g. diarization) means:
  1. Create a new file in pipeline/
  2. Subclass PipelineStep
  3. Implement run()
  4. Register the step in main.py

Nothing else needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from models import VideoRecord
from config import PROCESSING_DIR


class PipelineStep(ABC):
    """
    Base class for all pipeline steps.

    Each step:
    - Receives a VideoRecord (the full pipeline state for one video)
    - Mutates it in place (adding segments, translations, etc.)
    - Saves the updated record back to disk
    - Returns the (mutated) record

    Steps are idempotent: running a completed step again should be a no-op,
    checking the relevant status flag and returning early.
    """

    # Human-readable name shown in CLI progress output
    name: str = "unnamed_step"

    def __init__(self, processing_dir: Path = PROCESSING_DIR):
        self.processing_dir = processing_dir

    @abstractmethod
    def run(self, record: VideoRecord) -> VideoRecord:
        """
        Execute this step for one video.

        Args:
            record: Current pipeline state for the video.

        Returns:
            Updated VideoRecord (same object, mutated in place).
        """
        ...

    def is_complete(self, record: VideoRecord) -> bool:
        """
        Return True if this step has already been completed for this record.
        Override in subclasses for custom completion logic.
        Default implementation always returns False (re-run every time).
        """
        return False

    def run_if_needed(self, record: VideoRecord) -> VideoRecord:
        """
        Run this step only if it hasn't been completed yet.
        This is the recommended entry point from main.py.
        """
        if self.is_complete(record):
            print(f"  [{record.video_id}] Step '{self.name}' already done, skipping.")
            return record
        return self.run(record)