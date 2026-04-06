"""
utils/video.py — Video discovery and ID assignment.
Scans the video/ folder and ensures every file has a stable UUID-based stem.
"""

import uuid
from pathlib import Path

from config import VIDEO_DIR, VIDEO_EXTENSIONS


def scan_and_id_videos(video_dir: Path = VIDEO_DIR) -> list[tuple[str, Path]]:
    """
    Scan video_dir for supported video files.

    If a file's stem is already a valid 12-char hex ID it is kept as-is.
    Otherwise the file is renamed with a new hex ID so the stem becomes the
    stable video_id used throughout the pipeline.

    Returns:
        List of (video_id, Path) tuples sorted by filename.
    """
    videos: list[tuple[str, Path]] = []

    for f in sorted(video_dir.iterdir()):
        if f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        # Check if stem is already a valid 12-char hex ID
        if _is_hex_id(f.stem):
            video_id = f.stem
        else:
            video_id = uuid.uuid4().hex[:12]
            new_path = f.with_name(video_id + f.suffix)
            f.rename(new_path)
            f = new_path
            print(f"  Assigned ID {video_id} -> {f.name}")

        videos.append((video_id, f))

    return videos


def _is_hex_id(stem: str) -> bool:
    """Return True if stem looks like a 12-char lowercase hex ID."""
    if len(stem) != 12:
        return False
    try:
        int(stem, 16)
        return True
    except ValueError:
        return False