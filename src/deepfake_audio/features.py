"""Feature extraction for synthetic-speech artifacts."""

from __future__ import annotations

import librosa
import numpy as np
from scipy.stats import kurtosis, skew

from .audio import AudioSource, load_audio, split_segments
from .config import AudioConfig

FEATURE_VERSION = 2


def _summary(matrix: np.ndarray) -> np.ndarray:
    matrix = np.atleast_2d(matrix)
    return np.concatenate(
        [
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1),
            np.median(matrix, axis=1),
            np.percentile(matrix, 10, axis=1),
            np.percentile(matrix, 90, axis=1),
            np.nan_to_num(skew(matrix, axis=1, bias=False)),
            np.nan_to_num(kurtosis(matrix, axis=1, bias=False)),
        ]
    )


def extract_segment_features(audio: np.ndarray, config: AudioConfig) -> np.ndarray:
    """Extract cepstral, spectral, temporal, and harmonic descriptors."""
    kwargs = {
        "y": audio,
        "sr": config.sample_rate,
        "n_fft": config.n_fft,
        "hop_length": config.hop_length,
        "win_length": config.win_length,
    }
    mel = librosa.feature.melspectrogram(
        **kwargs, n_mels=config.n_mels, power=2.0, fmin=20, fmax=7_600
    )
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=config.n_mfcc)
    delta = librosa.feature.delta(mfcc, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")

    magnitude = np.abs(
        librosa.stft(
            audio,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
        )
    )
    spectral = np.vstack(
        [
            librosa.feature.spectral_centroid(S=magnitude, sr=config.sample_rate),
            librosa.feature.spectral_bandwidth(S=magnitude, sr=config.sample_rate),
            librosa.feature.spectral_rolloff(
                S=magnitude, sr=config.sample_rate, roll_percent=0.85
            ),
            librosa.feature.spectral_flatness(S=magnitude),
            librosa.feature.zero_crossing_rate(
                audio, frame_length=config.n_fft, hop_length=config.hop_length
            ),
            librosa.feature.rms(
                S=magnitude, frame_length=config.n_fft, hop_length=config.hop_length
            ),
        ]
    )
    contrast = librosa.feature.spectral_contrast(
        S=magnitude, sr=config.sample_rate, n_bands=6
    )
    chroma = librosa.feature.chroma_stft(
        S=magnitude, sr=config.sample_rate, tuning=0.0
    )

    autocorrelation = librosa.autocorrelate(audio, max_size=config.sample_rate // 50)
    ac_peak = float(np.max(autocorrelation[1:]) / max(autocorrelation[0], 1e-8))
    clipping = float(np.mean(np.abs(audio) >= 0.99))
    crest = float(np.max(np.abs(audio)) / max(np.sqrt(np.mean(audio**2)), 1e-8))

    features = np.concatenate(
        [
            _summary(mfcc),
            _summary(delta),
            _summary(delta2),
            _summary(log_mel),
            _summary(spectral),
            _summary(contrast),
            _summary(chroma),
            np.array([ac_peak, clipping, crest], dtype=np.float64),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def extract_features(source: AudioSource, config: AudioConfig) -> np.ndarray:
    """Return one robust feature vector by aggregating all audio segments."""
    audio = load_audio(source, config)
    segment_features = np.vstack(
        [extract_segment_features(segment, config) for segment in split_segments(audio, config)]
    )
    if len(segment_features) == 1:
        spread = np.zeros_like(segment_features[0])
    else:
        spread = np.std(segment_features, axis=0)
    return np.concatenate([np.mean(segment_features, axis=0), spread]).astype(
        np.float32
    )
