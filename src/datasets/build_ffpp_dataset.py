from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.runtime import set_seed


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def discover_videos(root: Path, label: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        rows.append(
            {
                "video_id": path.stem,
                "video_path": str(path.resolve()),
                "label": label,
            }
        )
    return rows


def sample_rows(df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(df) < count:
        raise ValueError(f"Requested {count} items but only found {len(df)} at source root.")
    return df.sample(n=count, random_state=seed).reset_index(drop=True)


def assign_split(df: pd.DataFrame, split: str, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    df["split"] = split
    df["dataset"] = dataset_name
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FF++ video split CSV.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    real_root = resolve_path(config["paths"]["ffpp_real_dir"])
    fake_root = resolve_path(config["paths"]["ffpp_fake_dir"])

    real_df = pd.DataFrame(discover_videos(real_root, label=0))
    fake_df = pd.DataFrame(discover_videos(fake_root, label=1))

    splits = []
    split_specs = {
        "train": (config["data"]["ffpp_train_real"], config["data"]["ffpp_train_fake"]),
        "val": (config["data"]["ffpp_val_real"], config["data"]["ffpp_val_fake"]),
        "test": (config["data"]["ffpp_test_real"], config["data"]["ffpp_test_fake"]),
    }

    remaining_real = real_df.copy()
    remaining_fake = fake_df.copy()
    for idx, (split_name, (real_count, fake_count)) in enumerate(split_specs.items()):
        real_sample = sample_rows(remaining_real, real_count, seed=config["seed"] + idx)
        fake_sample = sample_rows(remaining_fake, fake_count, seed=config["seed"] + 100 + idx)
        remaining_real = remaining_real[~remaining_real["video_id"].isin(real_sample["video_id"])].copy()
        remaining_fake = remaining_fake[~remaining_fake["video_id"].isin(fake_sample["video_id"])].copy()
        splits.append(assign_split(real_sample, split_name, "ffpp"))
        splits.append(assign_split(fake_sample, split_name, "ffpp"))

    split_df = pd.concat(splits, ignore_index=True)
    output_path = resolve_path(config["paths"]["ffpp_split_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    print(f"Saved FF++ split manifest to {output_path}")


if __name__ == "__main__":
    main()
