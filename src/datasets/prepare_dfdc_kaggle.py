from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.runtime import set_seed


COMPETITION = "deepfake-detection-challenge"


def ensure_kaggle_credentials() -> Path | None:
    if "KAGGLE_API_TOKEN" in __import__("os").environ:
        return None
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "Missing Kaggle credentials. Set KAGGLE_API_TOKEN or place kaggle.json "
            "at ~/.kaggle/kaggle.json before running this script."
        )
    return kaggle_json


def run_kaggle(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def extract_archives(download_dir: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_dirs: list[Path] = []
    for archive in sorted(download_dir.glob("*.zip")):
        target = extract_dir / archive.stem
        if target.exists():
            extracted_dirs.append(target)
            continue
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as handle:
            handle.extractall(target)
        extracted_dirs.append(target)
    return extracted_dirs


def load_metadata(extracted_dirs: list[Path]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for directory in extracted_dirs:
        for metadata_path in directory.rglob("metadata.json"):
            with metadata_path.open("r", encoding="utf-8") as handle:
                merged.update(json.load(handle))
    return merged


def copy_video_subset(
    metadata: dict[str, dict],
    extracted_dirs: list[Path],
    real_root: Path,
    fake_root: Path,
    limit_real: int,
    limit_fake: int,
) -> pd.DataFrame:
    real_root.mkdir(parents=True, exist_ok=True)
    fake_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    real_count = 0
    fake_count = 0

    def find_video(filename: str) -> Path | None:
        for directory in extracted_dirs:
            candidate = next(directory.rglob(filename), None)
            if candidate is not None:
                return candidate
        return None

    for filename, item in sorted(metadata.items()):
        label = item.get("label", "").upper()
        if label == "REAL":
            if real_count >= limit_real:
                continue
            destination_root = real_root
            mapped_label = 0
        elif label == "FAKE":
            if fake_count >= limit_fake:
                continue
            destination_root = fake_root
            mapped_label = 1
        else:
            continue

        source = find_video(filename)
        if source is None:
            continue

        destination = destination_root / filename
        if not destination.exists():
            shutil.copy2(source, destination)

        rows.append(
            {
                "video_id": Path(filename).stem,
                "video_path": str(destination.resolve()),
                "label": mapped_label,
                "split": "test",
                "dataset": "dfdc",
            }
        )
        if mapped_label == 0:
            real_count += 1
        else:
            fake_count += 1
        if real_count >= limit_real and fake_count >= limit_fake:
            break

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official DFDC data from Kaggle into scratch storage.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--download-dir", default="/home/hice1/yyang3119/scratch/deepfake_data/dfdc/kaggle_zips")
    parser.add_argument("--extract-dir", default="/home/hice1/yyang3119/scratch/deepfake_data/dfdc/kaggle_extracted")
    parser.add_argument("--limit-real", type=int, default=None)
    parser.add_argument("--limit-fake", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    download_dir = resolve_path(args.download_dir)
    extract_dir = resolve_path(args.extract_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    limit_real = args.limit_real or config["data"]["dfdc_test_real"]
    limit_fake = args.limit_fake or config["data"]["dfdc_test_fake"]

    if not args.skip_download:
        ensure_kaggle_credentials()
        run_kaggle(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-p",
                str(download_dir),
            ],
            cwd=resolve_path("."),
        )

    extracted_dirs = extract_archives(download_dir, extract_dir)
    metadata = load_metadata(extracted_dirs)
    split_df = copy_video_subset(
        metadata=metadata,
        extracted_dirs=extracted_dirs,
        real_root=resolve_path(config["paths"]["dfdc_real_dir"]),
        fake_root=resolve_path(config["paths"]["dfdc_fake_dir"]),
        limit_real=limit_real,
        limit_fake=limit_fake,
    )

    output_path = resolve_path(config["paths"]["dfdc_split_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    print(f"Saved DFDC split CSV to {output_path}")
    print(split_df['label'].value_counts().to_dict())


if __name__ == "__main__":
    main()
