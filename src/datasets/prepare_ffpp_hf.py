from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.utils.config import load_config, resolve_path
from src.utils.runtime import set_seed


def normalize_split(split_name: str) -> str:
    return "val" if split_name == "validation" else split_name


def map_label(category: str, real_categories: set[str], fake_categories: set[str]) -> int | None:
    if category in real_categories:
        return 0
    if category in fake_categories:
        return 1
    return None


def export_split(
    hf_split,
    split_name: str,
    output_root: Path,
    real_categories: set[str],
    fake_categories: set[str],
    save_images: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_key = normalize_split(split_name)
    iterable = tqdm(hf_split, desc=f"Preparing HF FF++ {split_name}", leave=False)

    for sample in iterable:
        category = sample["category"]
        mapped_label = map_label(category, real_categories, fake_categories)
        if mapped_label is None:
            continue

        label_name = "real" if mapped_label == 0 else "fake"
        frame_id = f"{int(sample['frame_number']):05d}"
        video_id = str(sample["video_id"])
        output_path = output_root / split_key / label_name / video_id / f"{frame_id}.jpg"
        if save_images:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sample["image"].save(output_path)

        rows.append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "face_path": str(output_path.resolve()),
                "label": mapped_label,
                "split": split_key,
                "dataset": "ffpp",
                "source_label": int(sample["label"]),
                "source_label_text": sample["label_text"],
                "source_category": category,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FF++ face images from Hugging Face.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--output-root", default="/home/hice1/yyang3119/scratch/deepfake_data/ffpp/faces/images")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--no-save-images", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    dataset_name = args.dataset_name or config["hf_datasets"]["ffpp"]["name"]
    output_root = resolve_path(args.output_root)
    manifest_out = resolve_path(args.manifest_out or config["paths"]["ffpp_face_manifest"])
    cache_dir = resolve_path(config["paths"]["hf_cache_dir"])

    real_categories = set(config["hf_datasets"]["ffpp"]["real_categories"])
    fake_categories = set(config["hf_datasets"]["ffpp"]["fake_categories"])
    save_images = not args.no_save_images and bool(config["hf_datasets"]["ffpp"]["save_images"])

    dataset = load_dataset(dataset_name, cache_dir=str(cache_dir))
    manifest_frames = []
    for split_name in ["train", "validation", "test"]:
        manifest_frames.append(
            export_split(
                hf_split=dataset[split_name],
                split_name=split_name,
                output_root=output_root,
                real_categories=real_categories,
                fake_categories=fake_categories,
                save_images=save_images,
            )
        )

    manifest_df = pd.concat(manifest_frames, ignore_index=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_out, index=False)

    split_df = (
        manifest_df[["video_id", "label", "split", "dataset"]]
        .drop_duplicates()
        .sort_values(["split", "label", "video_id"])
        .reset_index(drop=True)
    )
    split_out = resolve_path(config["paths"]["ffpp_split_csv"])
    split_out.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_out, index=False)

    print(f"Saved FF++ HF face manifest to {manifest_out}")
    print(f"Saved FF++ HF split manifest to {split_out}")


if __name__ == "__main__":
    main()
