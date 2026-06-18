from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from deepfake_audio.audio import AudioValidationError, load_audio, split_segments
from deepfake_audio.config import AudioConfig
from deepfake_audio.features import extract_features


def wav_bytes(duration: float = 1.0, sample_rate: int = 22_050) -> bytes:
    time = np.arange(int(duration * sample_rate)) / sample_rate
    audio = 0.4 * np.sin(2 * np.pi * 220 * time)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def test_load_audio_resamples_and_normalizes() -> None:
    config = AudioConfig()
    audio = load_audio(wav_bytes(), config)
    assert 15_500 <= len(audio) <= 16_100
    assert np.max(np.abs(audio)) == pytest.approx(1.0, abs=1e-5)


def test_extract_features_is_finite_and_fixed_size() -> None:
    config = AudioConfig()
    first = extract_features(wav_bytes(1.0), config)
    second = extract_features(wav_bytes(2.5), config)
    assert first.shape == second.shape
    assert first.ndim == 1
    assert first.size > 1_000
    assert np.all(np.isfinite(first))


def test_short_and_invalid_audio_are_rejected() -> None:
    config = AudioConfig()
    with pytest.raises(AudioValidationError):
        load_audio(wav_bytes(0.1), config)
    with pytest.raises(AudioValidationError):
        load_audio(b"not audio", config)


def test_segment_split_pads_consistently() -> None:
    config = AudioConfig(segment_duration=1.0)
    segments = split_segments(np.ones(24_000, dtype=np.float32), config)
    assert len(segments) == 2
    assert all(len(segment) == 16_000 for segment in segments)

