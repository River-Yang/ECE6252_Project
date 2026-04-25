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
                "split": "test",
                "dataset": "dfdc",
            }
        )
    return rows


def sample_rows(df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(df) < count:
        raise ValueError(f"Requested {count} items but only found {len(df)} at source root.")
    return df.sample(n=count, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DFDC test split CSV.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    real_root = resolve_path(config["paths"]["dfdc_real_dir"])
    fake_root = resolve_path(config["paths"]["dfdc_fake_dir"])

    real_df = pd.DataFrame(discover_videos(real_root, label=0))
    fake_df = pd.DataFrame(discover_videos(fake_root, label=1))

    real_df = sample_rows(real_df, config["data"]["dfdc_test_real"], config["seed"])
    fake_df = sample_rows(fake_df, config["data"]["dfdc_test_fake"], config["seed"] + 1)
    split_df = pd.concat([real_df, fake_df], ignore_index=True)

    output_path = resolve_path(config["paths"]["dfdc_split_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    print(f"Saved DFDC split manifest to {output_path}")


if __name__ == "__main__":
    main()
