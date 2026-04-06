"""
utils/audio.py — Audio extraction and processing helpers.
These are pure functions with no pipeline state — easy to unit-test and reuse.
"""

import os
from pathlib import Path

from pydub import AudioSegment


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Extract audio from a video file using ffmpeg.

    Args:
        video_path:   Source video file.
        output_path:  Destination WAV path.
        sample_rate:  Output sample rate in Hz (16000 = Whisper standard).
        channels:     1 = mono (default), 2 = stereo.

    Returns:
        str path to the written WAV file.
    """
    os.system(
        f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le '
        f'-ar {sample_rate} -ac {channels} "{output_path}" -loglevel error'
    )
    return str(output_path)


def extract_segment_audio(
    source_audio_path: str | Path,
    time_start: float,
    time_end: float,
    output_path: str | Path,
    sample_rate: int = 22050,
    channels: int = 1,
) -> str:
    """
    Slice a time window from an audio file, peak-normalise, and save as WAV.

    Args:
        source_audio_path: Full audio file to slice from.
        time_start:        Segment start in seconds.
        time_end:          Segment end in seconds.
        output_path:       Destination WAV path.
        sample_rate:       Output sample rate (22050 = XTTS default).
        channels:          Output channels.

    Returns:
        str path to the written WAV file.
    """
    duration = time_end - time_start
    os.system(
        f'ffmpeg -y -ss {time_start} -t {duration} -i "{source_audio_path}" '
        f'-vn -acodec pcm_s16le -ar {sample_rate} -ac {channels} '
        f'"{output_path}" -loglevel error'
    )
    clip = AudioSegment.from_wav(str(output_path))
    clip = peak_normalise(clip)
    clip.export(str(output_path), format="wav")
    return str(output_path)


def peak_normalise(clip: AudioSegment, target_db: float = -3.0) -> AudioSegment:
    """
    Normalise an AudioSegment so its peak level hits target_db.
    Only amplifies quiet clips; clips that are already at or above 0 dBFS
    are attenuated to 0 dBFS to avoid clipping.
    """
    peak_db = clip.max_dBFS
    if peak_db < target_db:
        clip = clip.apply_gain(target_db - peak_db)
    elif peak_db > 0:
        clip = clip.apply_gain(-peak_db)
    return clip


def speed_adjust(input_path: str | Path, output_path: str | Path, tempo: float) -> str:
    """
    Time-stretch audio using ffmpeg's atempo filter.
    atempo is limited to [0.5, 2.0]; values outside this range are chained.

    Args:
        input_path:  Source WAV.
        output_path: Destination WAV.
        tempo:       Speed multiplier (e.g. 1.5 = 50% faster).

    Returns:
        str path to output file (may be input_path if tempo ≈ 1.0).
    """
    if abs(tempo - 1.0) < 0.01:
        return str(input_path)

    # Chain atempo filters if outside [0.5, 2.0]
    filters = _build_atempo_chain(tempo)
    filter_str = ",".join(f"atempo={f:.6f}" for f in filters)
    os.system(
        f'ffmpeg -y -i "{input_path}" -filter:a "{filter_str}" '
        f'"{output_path}" -loglevel error'
    )
    return str(output_path)


def _build_atempo_chain(tempo: float) -> list[float]:
    """
    ffmpeg atempo only accepts values in [0.5, 2.0].
    Chain multiple filters to achieve tempos outside this range.
    """
    filters = []
    remaining = tempo
    while remaining > 2.0:
        filters.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        filters.append(0.5)
        remaining /= 0.5
    filters.append(remaining)
    return filters