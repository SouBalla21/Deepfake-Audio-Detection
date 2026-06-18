"""Audio loading and defensive preprocessing."""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf

from .config import AudioConfig


class AudioValidationError(ValueError):
    """Raised when an input cannot be used for speech analysis."""


AudioSource = str | Path | bytes | BinaryIO


def _load_with_soundfile(source: AudioSource) -> tuple[np.ndarray, int]:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    audio, sample_rate = sf.read(source, always_2d=False, dtype="float32")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def load_audio(source: AudioSource, config: AudioConfig) -> np.ndarray:
    """Load, mono-convert, resample, trim, and peak-normalize audio."""
    try:
        audio, sample_rate = _load_with_soundfile(source)
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                audio, sample_rate = librosa.load(
                    source, sr=None, mono=False, dtype=np.float32
                )
        except Exception as exc:
            raise AudioValidationError(
                "Could not decode this audio file. Use a valid WAV or MP3 file."
            ) from exc

    if audio.ndim == 2:
        channel_axis = 0 if audio.shape[0] <= 8 else 1
        audio = np.mean(audio, axis=channel_axis)
    audio = np.nan_to_num(np.ravel(audio), nan=0.0, posinf=0.0, neginf=0.0)

    if sample_rate <= 0 or audio.size == 0:
        raise AudioValidationError("The audio file is empty or has an invalid sample rate.")
    if sample_rate != config.sample_rate:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=config.sample_rate
        )

    audio, _ = librosa.effects.trim(audio, top_db=config.top_db)
    minimum_samples = int(config.min_duration * config.sample_rate)
    if audio.size < minimum_samples:
        raise AudioValidationError(
            f"At least {config.min_duration:.2f} seconds of audible audio is required."
        )
    if float(np.sqrt(np.mean(np.square(audio)))) < 1e-5:
        raise AudioValidationError("The recording is silent or too quiet to analyze.")

    max_samples = int(config.max_duration * config.sample_rate)
    audio = audio[:max_samples]
    peak = float(np.max(np.abs(audio)))
    return (audio / max(peak, 1e-8)).astype(np.float32)


def split_segments(audio: np.ndarray, config: AudioConfig) -> list[np.ndarray]:
    """Split audio into fixed analysis windows and pad the final window."""
    size = int(config.segment_duration * config.sample_rate)
    if audio.size <= size:
        return [np.pad(audio, (0, size - audio.size))]
    segments = []
    for start in range(0, audio.size, size):
        segment = audio[start : start + size]
        if segment.size < size // 2:
            break
        segments.append(np.pad(segment, (0, size - segment.size)))
    return segments
