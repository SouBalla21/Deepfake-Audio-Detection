"""High-level inference API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio import AudioSource
from .config import DEFAULT_MODEL_PATH, AudioConfig
from .features import extract_features
from .modeling import load_artifact


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    deepfake_probability: float
    genuine_probability: float
    threshold: float


class AudioDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run `python train.py` first."
            )
        self.artifact = load_artifact(self.model_path)
        self.config = AudioConfig(**self.artifact.audio_config)

    def predict(self, source: AudioSource) -> Prediction:
        features = extract_features(source, self.config).reshape(1, -1)
        probability = float(self.artifact.estimator.predict_proba(features)[0, 1])
        is_deepfake = probability >= self.artifact.threshold
        confidence = probability if is_deepfake else 1.0 - probability
        return Prediction(
            label="Deepfake" if is_deepfake else "Genuine",
            confidence=confidence,
            deepfake_probability=probability,
            genuine_probability=1.0 - probability,
            threshold=self.artifact.threshold,
        )
