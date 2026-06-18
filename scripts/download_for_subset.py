#!/usr/bin/env python3
"""Download a reproducible balanced subset of the Fake-or-Real dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
from pathlib import Path
from urllib.parse import quote

import requests
from remotezip import RemoteZip
from tqdm import tqdm

DATASET_API = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "mohammedabdeldayem/the-fake-or-real-dataset"
)
FILE_API = DATASET_API + "/{member}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-per-class", type=int, default=300)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def category(name: str) -> tuple[str, str] | None:
    normalized = name.lower()
    if not normalized.endswith((".wav", ".mp3", ".flac")):
        return None
    if not normalized.startswith("for-norm/for-norm/"):
        return None
    split_map = {"training": "train", "validation": "validation", "testing": "test"}
    for source_split, target_split in split_map.items():
        for label in ("real", "fake"):
            if f"/{source_split}/{label}/" in normalized:
                return target_split, label
    return None


def main() -> None:
    args = parse_args()
    quotas = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
    }
    response = requests.get(DATASET_API, allow_redirects=False, timeout=30)
    response.raise_for_status()
    archive_url = response.headers["location"]

    rng = random.Random(args.seed)
    with RemoteZip(archive_url) as archive:
        grouped: dict[tuple[str, str], list[str]] = {}
        for name in archive.namelist():
            key = category(name)
            if key:
                grouped.setdefault(key, []).append(name)

        selected: list[tuple[str, str, str]] = []
        for split, count in quotas.items():
            for label in ("real", "fake"):
                candidates = sorted(grouped.get((split, label), []))
                if len(candidates) < count:
                    raise RuntimeError(
                        f"Only {len(candidates)} files found for {split}/{label}; "
                        f"{count} requested."
                    )
                for name in rng.sample(candidates, count):
                    selected.append((split, label, name))

    manifest = []
    tasks: list[tuple[str, Path]] = []
    for index, (split, label, member) in enumerate(selected, start=1):
        suffix = Path(member).suffix.lower()
        destination = args.output / split / label / f"{index:05d}{suffix}"
        manifest.append(
            {
                "path": destination.as_posix(),
                "split": split,
                "label": label,
                "source_member": member,
            }
        )
        tasks.append((member, destination))

    def download(task: tuple[str, Path]) -> None:
        member, destination = task
        if destination.exists() and destination.stat().st_size > 0:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = FILE_API.format(member=quote(member, safe=""))
        for attempt in range(6):
            try:
                file_response = requests.get(url, timeout=90)
                file_response.raise_for_status()
                destination.write_bytes(file_response.content)
                return
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep((2**attempt) + random.random())

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(
            tqdm(
                executor.map(download, tasks),
                total=len(tasks),
                desc="Downloading FoR audio",
            )
        )

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} files and manifest to {args.output}")


if __name__ == "__main__":
    main()
