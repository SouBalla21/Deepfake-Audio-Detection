"""Shared project configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_audio_detector.joblib"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    min_duration: float = 0.35
    max_duration: float = 12.0
    segment_duration: float = 2.0
    top_db: int = 35
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 64
    n_mfcc: int = 24

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

