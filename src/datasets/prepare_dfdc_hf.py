from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.utils.config import load_config, resolve_path
from src.utils.runtime import set_seed


def parse_label(metadata_value: object) -> int | None:
    if isinstance(metadata_value, dict):
        label_value = metadata_value.get("label")
        if isinstance(label_value, str):
            lowered = label_value.lower()
            if lowered == "real":
                return 0
            if lowered == "fake":
                return 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and organize DFDC videos from a Hugging Face mirror.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--limit-real", type=int, default=None)
    parser.add_argument("--limit-fake", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    dataset_name = args.dataset_name or config["hf_datasets"]["dfdc"]["name"]
    cache_dir = resolve_path(config["paths"]["hf_cache_dir"])
    metadata_path = hf_hub_download(dataset_name, "metadata.json", repo_type="dataset", cache_dir=str(cache_dir))
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    real_root = resolve_path(config["paths"]["dfdc_real_dir"])
    fake_root = resolve_path(config["paths"]["dfdc_fake_dir"])
    real_root.mkdir(parents=True, exist_ok=True)
    fake_root.mkdir(parents=True, exist_ok=True)

    real_count = 0
    fake_count = 0
    manifest_rows: list[dict[str, object]] = []

    for filename in sorted(metadata):
        label = parse_label(metadata[filename])
        if label is None:
            continue
        if label == 0 and args.limit_real is not None and real_count >= args.limit_real:
            continue
        if label == 1 and args.limit_fake is not None and fake_count >= args.limit_fake:
            continue

        local_path = hf_hub_download(dataset_name, filename, repo_type="dataset", cache_dir=str(cache_dir))
        destination_root = real_root if label == 0 else fake_root
        destination = destination_root / filename
        if not destination.exists():
            shutil.copy2(local_path, destination)

        manifest_rows.append(
            {
                "video_id": Path(filename).stem,
                "video_path": str(destination.resolve()),
                "label": label,
                "split": "test",
                "dataset": "dfdc",
            }
        )
        if label == 0:
            real_count += 1
        else:
            fake_count += 1

    split_csv = resolve_path(config["paths"]["dfdc_split_csv"])
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(manifest_rows).to_csv(split_csv, index=False)
    print(f"Saved DFDC videos to {real_root.parent}")
    print(f"Saved DFDC split manifest to {split_csv}")
    print(f"real={real_count} fake={fake_count}")


if __name__ == "__main__":
    main()
