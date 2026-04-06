"""
pipeline/dubber.py — Step 4: Dubbing (TTS synthesis + audio assembly).

Pipeline for each video:
  1. Separate vocals (Demucs) for clean reference clips
  2. Extract reference WAV clips from vocals track
  3. Build a silent audio timeline matching the original video length
  4. For each translated segment:
     a. Synthesise speech with XTTS v2 (voice-cloned from reference clips)
     b. Transfer F0 (pitch contour) from the original speaker
     c. Speed-adjust if the TTS output is longer than the available time slot
     d. Overlay onto the silent timeline at the correct timestamp
  5. Export final WAV and merge back into the video as MP4
"""

from __future__ import annotations

import os
from pathlib import Path

from pydub import AudioSegment

from pipeline.base import PipelineStep
from pipeline.vocal_separator import separate_vocals
from models import VideoRecord, ReferenceChunk
from utils.audio import extract_audio, extract_segment_audio, peak_normalise, speed_adjust
from config import (
    TMP_DIR, PROCESSING_DIR, OUTPUT_DIR,
    TTS_MAX_CHARS, get_language_config,
)


# ── TTS text splitting ────────────────────────────────────────────────────────

def split_tts_text(text: str, lang_code: str, max_chars: int = TTS_MAX_CHARS) -> list[str]:
    """
    Split text into chunks that fit within XTTS's per-call character limit.

    Strategy (in order):
      1. If text fits, return as-is
      2. Split at sentence boundaries (。！？ for Japanese; .!? for others)
      3. Split at phrase boundaries (、 / ,)
      4. Hard split at max_chars as last resort

    Ensures no chunk ever exceeds max_chars.
    """
    import re

    if len(text) <= max_chars:
        return [text]

    if lang_code == "ja":
        sentence_re   = re.compile(r"(?<=[。！？])")
        phrase_delim  = "、"
    else:
        sentence_re   = re.compile(r"(?<=[.!?])\s+")
        phrase_delim  = ","

    parts = [p.strip() for p in sentence_re.split(text) if p.strip()]

    chunks: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            chunks.append(part)
            continue

        sub_parts = part.split(phrase_delim)
        current   = ""
        for sp in sub_parts:
            candidate = current + (phrase_delim if current else "") + sp
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(sp) > max_chars:
                    for i in range(0, len(sp), max_chars):
                        chunks.append(sp[i : i + max_chars])
                    current = ""
                else:
                    current = sp
        if current:
            chunks.append(current)

    return [c for c in chunks if c.strip()]


# ── XTTS synthesis ────────────────────────────────────────────────────────────

def synthesize_text(
    xtts_model,
    text: str,
    lang_code: str,
    ref_wavs: list[str],
    out_path: str,
) -> str:
    """
    Synthesise text to WAV using XTTS v2, automatically splitting if the text
    exceeds the 71-char hard limit. Concatenates chunks seamlessly.

    Args:
        xtts_model: Loaded CoquiTTS XTTS model instance.
        text:       Text to synthesise.
        lang_code:  BCP-47 language code (e.g. "ja", "en").
        ref_wavs:   List of reference WAV paths for voice cloning.
        out_path:   Destination WAV path.

    Returns:
        str path to output WAV.
    """
    chunks = split_tts_text(text, lang_code)

    if len(chunks) == 1:
        xtts_model.tts_to_file(
            text=chunks[0],
            file_path=out_path,
            speaker_wav=ref_wavs,
            language=lang_code,
        )
        return out_path

    # Synthesise each chunk, then concatenate
    tmp_dir     = Path(out_path).parent
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = str(tmp_dir / f"{Path(out_path).stem}_chunk{i:03d}.wav")
        xtts_model.tts_to_file(
            text=chunk,
            file_path=chunk_path,
            speaker_wav=ref_wavs,
            language=lang_code,
        )
        chunk_paths.append(chunk_path)

    combined = AudioSegment.empty()
    for cp in chunk_paths:
        combined += AudioSegment.from_wav(cp)
    combined.export(out_path, format="wav")

    for cp in chunk_paths:
        try:
            os.remove(cp)
        except OSError:
            pass

    return out_path


# ── F0 pitch transfer ─────────────────────────────────────────────────────────

def transfer_f0(
    original_wav_path: str,
    tts_wav_path: str,
    out_path: str,
) -> str:
    """
    Transfer pitch character from the original speaker to TTS output using
    Praat's "Change gender" (a single high-quality PSOLA pass).

    Shifts TTS median F0 toward the original speaker's median (clamped to
    ±4 semitones) and scales pitch range (expressiveness) to match.

    Falls back to unmodified TTS on any error (parselmouth not installed, etc.).

    Returns:
        str path to the pitch-adjusted WAV (or tts_wav_path on fallback).
    """
    import shutil

    try:
        import numpy as np
        import parselmouth
        from parselmouth.praat import call

        orig_snd = parselmouth.Sound(original_wav_path)
        tts_snd  = parselmouth.Sound(tts_wav_path)

        def f0_stats(snd, floor=75, ceiling=500):
            pitch  = snd.to_pitch(time_step=0.01,
                                   pitch_floor=floor, pitch_ceiling=ceiling)
            values = np.array([
                pitch.get_value_at_time(t) or 0.0
                for t in pitch.xs()
            ])
            voiced = values[values > 0]
            if len(voiced) < 3:
                return None, None
            return float(np.median(voiced)), float(np.std(np.log(voiced)))

        orig_median, orig_log_std = f0_stats(orig_snd)
        tts_median,  tts_log_std  = f0_stats(tts_snd)

        if orig_median is None or tts_median is None:
            shutil.copy2(tts_wav_path, out_path)
            return out_path

        # Pitch range (expressiveness) ratio, clamped
        if tts_log_std and tts_log_std > 0:
            range_factor = float(np.clip(orig_log_std / tts_log_std, 0.5, 2.0))
        else:
            range_factor = 1.0

        # Median shift, clamped to ±4 semitones
        ratio     = float(np.clip(orig_median / tts_median, 2**(-4/12), 2**(4/12)))
        new_pitch = tts_median * ratio
        semitones = 12 * np.log2(ratio)

        print(
            f"    F0: tts {tts_median:.0f}Hz -> target {new_pitch:.0f}Hz "
            f"({semitones:+.1f} st), range factor {range_factor:.2f}x"
        )

        tts_shifted = call(
            tts_snd, "Change gender",
            75,           # pitch floor
            500,          # pitch ceiling
            1.0,          # formant shift (1.0 = none)
            new_pitch,    # new median pitch (Hz)
            range_factor, # pitch range factor
            1.0,          # duration factor (1.0 = none)
        )
        tts_shifted.save(out_path, "WAV")
        return out_path

    except Exception as e:
        print(f"    WARNING: F0 transfer failed ({e}), using original TTS.")
        import shutil
        shutil.copy2(tts_wav_path, out_path)
        return out_path


# ── Reference clip extraction ─────────────────────────────────────────────────

def extract_reference_wavs(
    record: VideoRecord,
    vocals_path: str,
    processing_dir: Path,
) -> list[str]:
    """
    Extract reference WAV clips from the separated vocals track.
    Results are cached in processing/<video_id>_refs/.

    Returns:
        List of paths to reference WAV files.
    """
    ref_dir = processing_dir / f"{record.video_id}_refs"
    ref_dir.mkdir(exist_ok=True)

    ref_wavs: list[str] = []
    for i, chunk in enumerate(record.reference_chunks):
        ref_path = ref_dir / f"ref_{i:02d}.wav"
        if not ref_path.exists():
            extract_segment_audio(
                vocals_path,
                chunk.time_start,
                chunk.time_end,
                str(ref_path),
            )
        ref_wavs.append(str(ref_path))
        print(
            f"  [{record.video_id}] Ref {i+1}: "
            f"{chunk.time_start:.1f}s-{chunk.time_end:.1f}s "
            f"({chunk.duration:.1f}s)"
        )

    return ref_wavs


# ── Dubber step ───────────────────────────────────────────────────────────────

class DubberStep(PipelineStep):
    """
    Synthesises dubbed audio for all translated segments and assembles
    the final WAV + MP4 output files.

    The xtts_model is injected at construction so the heavy load happens
    once in main.py.
    """

    name = "dub"

    def __init__(
        self,
        xtts_model,
        target_lang: str,
        processing_dir: Path = PROCESSING_DIR,
    ):
        super().__init__(processing_dir)
        self.xtts_model  = xtts_model
        self.target_lang = target_lang

    def is_complete(self, record: VideoRecord) -> bool:
        return record.status.dubbed.get(self.target_lang, False)

    def run(self, record: VideoRecord) -> VideoRecord:
        lang_cfg   = get_language_config(self.target_lang)
        lang_code  = lang_cfg["tts_code"]
        translated = record.translated_segments(self.target_lang)

        if not translated:
            print(
                f"  [{record.video_id}] No {self.target_lang} translations found, skip."
            )
            return record

        # ── 1. Vocal separation ───────────────────────────────────────────────
        vocals_path = separate_vocals(
            record.video_path, record.video_id, self.processing_dir
        )

        # ── 2. Extract reference clips ────────────────────────────────────────
        ref_wavs = extract_reference_wavs(record, vocals_path, self.processing_dir)
        if not ref_wavs:
            print(f"  [{record.video_id}] ERROR: No reference WAVs — cannot dub.")
            return record

        # ── 3. Silent timeline ────────────────────────────────────────────────
        orig_wav = str(TMP_DIR / f"{record.video_id}_orig.wav")
        extract_audio(record.video_path, orig_wav)
        orig_track    = AudioSegment.from_wav(orig_wav)
        total_ms      = len(orig_track)
        final_audio   = AudioSegment.silent(duration=total_ms)

        seg_dir = TMP_DIR / f"{record.video_id}_segs"
        seg_dir.mkdir(exist_ok=True)

        # ── 4. Synthesise and overlay each segment ────────────────────────────
        print(f"  [{record.video_id}] Synthesising {len(translated)} segments...")

        for i, seg in enumerate(translated):
            seg_path = str(seg_dir / f"seg_{i:04d}.wav")

            try:
                synthesize_text(
                    self.xtts_model,
                    seg.get_translation(self.target_lang),
                    lang_code,
                    ref_wavs,
                    seg_path,
                )
            except Exception as e:
                print(f"    WARNING: seg {i} synthesis failed: {e}")
                continue

            tts_audio     = AudioSegment.from_file(seg_path)
            seg_start_ms  = int(seg.time_start * 1000)
            seg_end_ms    = int(seg.time_end   * 1000)
            target_dur_ms = seg_end_ms - seg_start_ms

            if len(tts_audio) == 0 or target_dur_ms == 0:
                continue

            # ── F0 pitch transfer ─────────────────────────────────────────────
            orig_seg_path = str(seg_dir / f"seg_{i:04d}_orig.wav")
            extract_segment_audio(orig_wav, seg.time_start, seg.time_end, orig_seg_path)

            f0_path = str(seg_dir / f"seg_{i:04d}_f0.wav")
            transfer_f0(orig_seg_path, seg_path, f0_path)
            if os.path.exists(f0_path):
                tts_audio = AudioSegment.from_file(f0_path)

            # ── Speed-adjust if TTS is longer than slot ───────────────────────
            speed_ratio = len(tts_audio) / target_dur_ms
            if speed_ratio > 1.1:
                adj_path = str(seg_dir / f"seg_{i:04d}_adj.wav")
                tempo    = min(speed_ratio, 2.0)
                source   = f0_path if os.path.exists(f0_path) else seg_path
                speed_adjust(source, adj_path, tempo)
                if os.path.exists(adj_path):
                    tts_audio = AudioSegment.from_file(adj_path)

            # ── Overlay at timestamp ──────────────────────────────────────────
            if seg_start_ms < total_ms:
                final_audio = final_audio.overlay(tts_audio, position=seg_start_ms)

            if (i + 1) % 5 == 0:
                print(f"    ... {i+1}/{len(translated)} done")

        # ── 5. Export WAV ─────────────────────────────────────────────────────
        out_wav = OUTPUT_DIR / f"{record.video_id}_{self.target_lang}.wav"
        final_audio.export(str(out_wav), format="wav")
        print(f"  [{record.video_id}] WAV -> {out_wav}")

        # ── 6. Merge audio back into video (MP4) ──────────────────────────────
        out_mp4 = OUTPUT_DIR / f"{record.video_id}_{self.target_lang}.mp4"
        os.system(
            f'ffmpeg -y -i "{record.video_path}" -i "{out_wav}" '
            f'-c:v copy -map 0:v:0 -map 1:a:0 -shortest '
            f'"{out_mp4}" -loglevel error'
        )
        print(f"  [{record.video_id}] MP4 -> {out_mp4}")

        record.status.dubbed[self.target_lang] = True
        record.save(self.processing_dir)
        return record