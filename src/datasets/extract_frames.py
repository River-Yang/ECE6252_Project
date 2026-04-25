from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config, resolve_path
from src.utils.runtime import append_log


def sample_frame_indices(total_frames: int, frames_per_video: int) -> list[int]:
    if total_frames <= 0:
        return []
    positions = np.linspace(0, total_frames - 1, num=frames_per_video + 2, dtype=int)[1:-1]
    return sorted(set(int(position) for position in positions))


def save_frame(frame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def extract_frames_for_manifest(
    split_df: pd.DataFrame,
    output_root: Path,
    frames_per_video: int,
    log_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in tqdm(split_df.to_dict("records"), desc=f"Extracting frames -> {output_root.name}"):
        video_path = Path(row["video_path"])
        capture = cv2.VideoCapture(str(video_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(total_frames, frames_per_video)
        if not indices:
            append_log(f"[extract_frames] no frames for {video_path}", log_path)
            capture.release()
            continue

        extracted = 0
        for frame_idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = capture.read()
            if not success:
                append_log(f"[extract_frames] failed read {video_path} @ frame {frame_idx}", log_path)
                continue
            frame_id = f"{frame_idx:05d}"
            label_name = "real" if int(row["label"]) == 0 else "fake"
            output_path = output_root / row["split"] / label_name / row["video_id"] / f"{frame_id}.jpg"
            save_frame(frame, output_path)
            rows.append(
                {
                    "video_id": row["video_id"],
                    "frame_id": frame_id,
                    "frame_path": str(output_path.resolve()),
                    "label": int(row["label"]),
                    "split": row["split"],
                    "dataset": row["dataset"],
                }
            )
            extracted += 1
        if extracted == 0:
            append_log(f"[extract_frames] extracted 0 frames for {video_path}", log_path)
        capture.release()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniformly extract frames from videos in split CSV.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--log-name", default="extract_frames.log")
    args = parser.parse_args()

    config = load_config(args.config)
    split_df = pd.read_csv(resolve_path(args.split_csv))
    output_root = resolve_path(args.output_root)
    log_path = resolve_path(config["paths"]["log_dir"]) / args.log_name
    manifest = extract_frames_for_manifest(
        split_df=split_df,
        output_root=output_root,
        frames_per_video=config["data"]["frames_per_video"],
        log_path=log_path,
    )
    manifest_out = resolve_path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_out, index=False)
    print(f"Saved frame manifest to {manifest_out}")


if __name__ == "__main__":
    main()
