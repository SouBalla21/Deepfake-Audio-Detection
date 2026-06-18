"""Model creation and artifact serialization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from .config import AudioConfig


@dataclass
class ModelArtifact:
    estimator: Any
    audio_config: dict[str, Any]
    threshold: float
    metrics: dict[str, Any]
    feature_count: int
    trained_at: str
    version: str = "1.0"


def build_estimator(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        criterion="entropy",
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def save_artifact(
    path: Path,
    estimator: Any,
    config: AudioConfig,
    threshold: float,
    metrics: dict[str, Any],
    feature_count: int,
) -> ModelArtifact:
    artifact = ModelArtifact(
        estimator=estimator,
        audio_config=config.to_dict(),
        threshold=float(threshold),
        metrics=metrics,
        feature_count=int(feature_count),
        trained_at=datetime.now(UTC).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    return artifact


def load_artifact(path: str | Path) -> ModelArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, ModelArtifact):
        raise TypeError("Unsupported or corrupted model artifact.")
    return artifact


def predict_probabilities(estimator: Any, features: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.predict_proba(features), dtype=np.float64)[:, 1]

