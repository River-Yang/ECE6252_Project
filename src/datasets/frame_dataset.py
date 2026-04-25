from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FaceFrameDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable | None = None,
        split: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        # Filter inside the dataset so one manifest can back multiple runs.
        df = pd.read_csv(manifest_path)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].copy()
        if dataset_name is not None and "dataset" in df.columns:
            df = df[df["dataset"] == dataset_name].copy()
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.df.iloc[index]
        image = Image.open(row["face_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": float(row["label"]),
            "video_id": row["video_id"],
            "frame_id": row["frame_id"],
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
        }
