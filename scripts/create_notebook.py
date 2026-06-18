#!/usr/bin/env python3
"""Generate the executable project notebook from maintainable source cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Deepfake Audio Detection: Training and Evaluation

This notebook is the reproducible experiment companion to Auralis. It uses the
same production modules as `train.py` and `predict.py`, preventing notebook/code
drift. The workflow covers dataset inspection, preprocessing, feature
extraction, model training, threshold selection, held-out evaluation, and
inference.

**Labels:** `0 = Genuine`, `1 = Deepfake`."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from deepfake_audio.audio import load_audio
from deepfake_audio.config import AudioConfig, DEFAULT_MODEL_PATH
from deepfake_audio.features import extract_features
from deepfake_audio.inference import AudioDetector

print(f"Project root: {ROOT}")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Dataset

The project uses a deterministic balanced subset of the official Fake-or-Real
`for-norm` train, validation, and test directories. The test split is not used
for model fitting or threshold selection."""
        ),
        nbf.v4.new_code_cell(
            """rows = []
for split in ("train", "validation", "test"):
    for label in ("real", "fake"):
        folder = ROOT / "data" / "raw" / split / label
        rows.append({"split": split, "label": label, "files": len(list(folder.glob("*")))})
dataset_summary = pd.DataFrame(rows)
display(dataset_summary)
display(dataset_summary.pivot(index="split", columns="label", values="files"))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Preprocessing and Features

Recordings are decoded, converted to mono, resampled to 16 kHz, trimmed,
peak-normalized, capped at 12 seconds, and divided into 2-second windows.
Each window yields MFCC/delta, log-mel, spectral, chroma, energy, periodicity,
crest-factor, and clipping descriptors. Distribution statistics are aggregated
across windows."""
        ),
        nbf.v4.new_code_cell(
            """sample_path = next((ROOT / "data" / "raw" / "train" / "real").glob("*"))
config = AudioConfig()
audio = load_audio(sample_path, config)
features = extract_features(sample_path, config)

print(f"Sample: {sample_path.name}")
print(f"Duration: {len(audio) / config.sample_rate:.2f} seconds")
print(f"Feature dimensions: {features.shape[0]}")

time = pd.Series(range(len(audio))) / config.sample_rate
plt.figure(figsize=(12, 3))
plt.plot(time, audio, color="#22b8cf", linewidth=0.7)
plt.title("Preprocessed waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Train

The production training command extracts/caches features, fits a class-balanced
Extra Trees ensemble, chooses its decision threshold on validation EER, saves
the artifact, and evaluates once on the untouched test split."""
        ),
        nbf.v4.new_code_cell(
            """command = [
    sys.executable,
    str(ROOT / "train.py"),
    "--data-dir", str(ROOT / "data" / "raw"),
    "--model-path", str(ROOT / "models" / "deepfake_audio_detector.joblib"),
    "--report-dir", str(ROOT / "reports"),
    "--cache-dir", str(ROOT / "data" / "cache"),
]
completed = subprocess.run(
    command, cwd=ROOT, check=True, capture_output=True, text=True
)
print(completed.stdout.split("\\nModel saved")[0])"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Held-out Evaluation

EER is the operating point where false acceptance and false rejection are
equal (or closest on the empirical ROC curve). Lower is better. Accuracy and
F1 use the threshold selected on validation data."""
        ),
        nbf.v4.new_code_cell(
            """metrics = json.loads((ROOT / "reports" / "training_summary.json").read_text())
test = metrics["test"]
summary = pd.DataFrame(
    {
        "Metric": ["Accuracy", "F1", "ROC-AUC", "EER", "Genuine accuracy", "Deepfake accuracy"],
        "Value": [
            test["accuracy"],
            test["f1"],
            test["roc_auc"],
            test["eer"],
            test["per_class_accuracy"]["genuine"],
            test["per_class_accuracy"]["deepfake"],
        ],
    }
)
display(summary.style.format({"Value": "{:.3%}"}))
display(pd.DataFrame(test["classification_report"]).T.round(3))"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(ROOT / "reports" / "confusion_matrix.png"), width=600))
display(Image(filename=str(ROOT / "reports" / "roc_curve.png"), width=600))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Inference

The saved artifact contains the estimator, preprocessing configuration,
validation-selected threshold, training timestamp, and metrics."""
        ),
        nbf.v4.new_code_cell(
            """detector = AudioDetector(DEFAULT_MODEL_PATH)
prediction = detector.predict(sample_path)
pd.DataFrame(
    {
        "field": ["prediction", "confidence", "genuine_probability", "deepfake_probability"],
        "value": [
            prediction.label,
            f"{prediction.confidence:.2%}",
            f"{prediction.genuine_probability:.2%}",
            f"{prediction.deepfake_probability:.2%}",
        ],
    }
)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Conclusion

This experiment keeps the official split boundaries intact and reports both
classical classification and biometric-style detection metrics. Results should
still be revalidated on new generators, codecs, languages, and acoustic
conditions before high-stakes use."""
        ),
    ]

    output = Path("notebooks/deepfake_audio_training.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
