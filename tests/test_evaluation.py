from __future__ import annotations

import numpy as np

from deepfake_audio.evaluation import compute_eer, evaluate, select_threshold


def test_perfect_classifier_metrics() -> None:
    truth = np.array([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.1, 0.9, 0.99])
    threshold = select_threshold(truth, probabilities)
    metrics = evaluate(truth, probabilities, threshold)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["eer"] == 0.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_eer_is_bounded() -> None:
    truth = np.array([0, 1, 0, 1, 0, 1])
    probabilities = np.array([0.3, 0.7, 0.8, 0.2, 0.4, 0.6])
    eer, threshold = compute_eer(truth, probabilities)
    assert 0 <= eer <= 1
    assert np.isfinite(threshold)

