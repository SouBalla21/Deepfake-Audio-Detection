from __future__ import annotations

from pathlib import Path

import numpy as np

from deepfake_audio.config import AudioConfig
from deepfake_audio.inference import AudioDetector
from deepfake_audio.modeling import save_artifact


class StubEstimator:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile([0.2, 0.8], (len(features), 1))


def test_detector_returns_consistent_probabilities(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "model.joblib"
    save_artifact(path, StubEstimator(), AudioConfig(), 0.6, {}, 10)
    monkeypatch.setattr(
        "deepfake_audio.inference.extract_features",
        lambda source, config: np.zeros(10, dtype=np.float32),
    )
    result = AudioDetector(path).predict(b"placeholder")
    assert result.label == "Deepfake"
    assert result.confidence == 0.8
    assert result.genuine_probability + result.deepfake_probability == 1.0
