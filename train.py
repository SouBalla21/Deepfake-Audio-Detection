#!/usr/bin/env python3
"""Train and evaluate the deepfake audio detector."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

from deepfake_audio.config import DEFAULT_MODEL_PATH, AudioConfig
from deepfake_audio.evaluation import evaluate, save_report, select_threshold
from deepfake_audio.features import FEATURE_VERSION, extract_features
from deepfake_audio.modeling import (
    build_estimator,
    predict_probabilities,
    save_artifact,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def discover_files(data_dir: Path, split: str) -> tuple[list[Path], np.ndarray]:
    files: list[Path] = []
    labels: list[int] = []
    for label_name, label in (("real", 0), ("genuine", 0), ("fake", 1), ("deepfake", 1)):
        directory = data_dir / split / label_name
        if not directory.exists():
            continue
        for pattern in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
            matches = sorted(directory.glob(pattern))
            files.extend(matches)
            labels.extend([label] * len(matches))
    if not files:
        raise FileNotFoundError(
            f"No audio found in {data_dir / split}. Expected real/ and fake/ folders."
        )
    return files, np.asarray(labels, dtype=np.int8)


def cached_features(
    split: str,
    files: list[Path],
    labels: np.ndarray,
    config: AudioConfig,
    cache_dir: Path,
    jobs: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / f"{split}_features.joblib"
    signature = [(path.as_posix(), path.stat().st_size) for path in files]
    if cache_path.exists() and not force:
        cache = joblib.load(cache_path)
        if (
            cache.get("signature") == signature
            and cache.get("feature_version") == FEATURE_VERSION
            and cache.get("audio_config") == config.to_dict()
        ):
            return cache["features"], cache["labels"]

    results = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(extract_features)(path, config)
        for path in tqdm(files, desc=f"Extracting {split} features")
    )
    features = np.vstack(results)
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "features": features,
            "labels": labels,
            "signature": signature,
            "feature_version": FEATURE_VERSION,
            "audio_config": config.to_dict(),
        },
        cache_path,
        compress=3,
    )
    return features, labels


def main() -> None:
    args = parse_args()
    config = AudioConfig()
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        files, labels = discover_files(args.data_dir, split)
        datasets[split] = cached_features(
            split,
            files,
            labels,
            config,
            args.cache_dir,
            args.jobs,
            args.force_features,
        )

    x_train, y_train = datasets["train"]
    x_validation, y_validation = datasets["validation"]
    x_test, y_test = datasets["test"]

    estimator = build_estimator()
    estimator.fit(x_train, y_train)
    validation_probabilities = predict_probabilities(estimator, x_validation)
    threshold = select_threshold(y_validation, validation_probabilities)
    validation_metrics = evaluate(y_validation, validation_probabilities, threshold)

    test_probabilities = predict_probabilities(estimator, x_test)
    test_metrics = evaluate(y_test, test_probabilities, threshold)
    metrics = {
        "dataset": "Fake-or-Real (FoR), for-norm deterministic subset",
        "splits": {
            "train": int(len(y_train)),
            "validation": int(len(y_validation)),
            "test": int(len(y_test)),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    save_artifact(
        args.model_path,
        estimator,
        config,
        threshold,
        metrics,
        x_train.shape[1],
    )
    save_report(args.report_dir, test_metrics, y_test, test_probabilities)
    (args.report_dir / "training_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps(test_metrics, indent=2))
    print(f"\nModel saved to {args.model_path}")


if __name__ == "__main__":
    main()
