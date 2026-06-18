#!/usr/bin/env python3
"""Classify an audio recording as Genuine or Deepfake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepfake_audio.config import DEFAULT_MODEL_PATH
from deepfake_audio.inference import AudioDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Path to a WAV, MP3, FLAC, or OGG file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")

    result = AudioDetector(args.model).predict(args.audio)
    payload = {
        "prediction": result.label,
        "confidence": round(result.confidence, 6),
        "deepfake_probability": round(result.deepfake_probability, 6),
        "genuine_probability": round(result.genuine_probability, 6),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Prediction : {result.label}")
        print(f"Confidence : {result.confidence:.2%}")
        print(f"Genuine    : {result.genuine_probability:.2%}")
        print(f"Deepfake   : {result.deepfake_probability:.2%}")


if __name__ == "__main__":
    main()

