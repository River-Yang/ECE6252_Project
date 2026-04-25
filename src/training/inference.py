from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets.frame_dataset import FaceFrameDataset
from src.evaluation.metrics import (
    aggregate_video_predictions,
    compute_binary_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from src.models.resnet50_binary import build_resnet50_binary
from src.utils.config import load_config, resolve_path
from src.utils.corruptions import get_corruption
from src.utils.runtime import get_device, set_seed, write_json


class OptionalCorruption:
    def __init__(self, corruption: Callable | None) -> None:
        self.corruption = corruption

    def __call__(self, image):
        if self.corruption is None:
            return image
        return self.corruption(image)


def build_eval_transform(image_size: int, condition: str) -> transforms.Compose:
    no_corruption_conditions = {"clean", "external_shift", "pre_shifted", "combo_comp_blur_noise"}
    # Pre-shifted data should not be shifted twice.
    corruption = get_corruption(condition, image_size=image_size) if condition not in no_corruption_conditions else None
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            OptionalCorruption(corruption),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(checkpoint_path: Path, pretrained: bool, device: torch.device) -> torch.nn.Module:
    model = build_resnet50_binary(pretrained=pretrained)
    # Checkpoints include extra metadata.
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_loader(dataset: FaceFrameDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_manifest(
    manifest_path: Path,
    checkpoint_path: Path,
    image_size: int,
    threshold: float,
    condition: str,
    dataset_name: str | None,
    split: str | None,
    batch_size: int,
    num_workers: int,
    pretrained: bool,
    transform_condition: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    device = get_device()
    model = load_model(checkpoint_path, pretrained=pretrained, device=device)
    effective_condition = transform_condition or condition
    dataset = FaceFrameDataset(
        manifest_path=manifest_path,
        split=split,
        dataset_name=dataset_name,
        transform=build_eval_transform(image_size=image_size, condition=effective_condition),
    )
    records: list[dict[str, object]] = []
    for attempt_workers in (num_workers, 0):
        if records:
            break
        loader = build_loader(dataset, batch_size=batch_size, num_workers=attempt_workers)
        try:
            # Retry with 0 workers if shared storage complains.
            for batch in tqdm(loader, desc=f"infer:{dataset_name or 'all'}:{condition}", leave=False):
                logits = model(batch["image"].to(device))
                scores = torch.sigmoid(logits).cpu().numpy().ravel().tolist()
                for idx, score in enumerate(scores):
                    records.append(
                        {
                            "video_id": batch["video_id"][idx],
                            "frame_id": batch["frame_id"][idx],
                            "label": int(batch["label"][idx]),
                            "score": float(score),
                            "dataset": batch["dataset"][idx] or (dataset_name or ""),
                            "condition": condition,
                        }
                    )
        except PermissionError:
            records.clear()
            if attempt_workers == 0:
                raise

    frame_df = pd.DataFrame(records)
    video_df = aggregate_video_predictions(frame_df, threshold=threshold)
    metrics = compute_binary_metrics(video_df["label"], video_df["score"], threshold=threshold)
    metrics["auc"] = metrics["roc_auc"]
    return frame_df, video_df, metrics


def save_artifacts(
    frame_df: pd.DataFrame,
    video_df: pd.DataFrame,
    metrics: dict[str, float],
    prefix: str,
    prediction_dir: Path,
    figure_dir: Path,
    table_dir: Path,
) -> None:
    # Keep outputs consistent across experiments.
    frame_path = prediction_dir / f"{prefix}_frame_predictions.csv"
    video_path = prediction_dir / f"{prefix}_video_predictions.csv"
    metrics_path = table_dir / f"{prefix}_metrics.json"
    frame_df.to_csv(frame_path, index=False)
    video_df.to_csv(video_path, index=False)
    write_json(metrics, metrics_path)

    plot_confusion_matrix(video_df["label"], video_df["score"], figure_dir / f"{prefix}_confusion_matrix.png")
    plot_roc_curve(video_df["label"], video_df["score"], figure_dir / f"{prefix}_roc_curve.png")
    plot_pr_curve(video_df["label"], video_df["score"], figure_dir / f"{prefix}_pr_curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frame/video-level inference and save evaluation artifacts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--face-manifest", required=True)
    parser.add_argument("--checkpoint", default="results/checkpoints/resnet50_best_auc.pt")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--transform-condition", default=None)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    frame_df, video_df, metrics = predict_manifest(
        manifest_path=resolve_path(args.face_manifest),
        checkpoint_path=resolve_path(args.checkpoint),
        image_size=config["data"]["image_size"],
        threshold=config["eval"]["threshold"],
        condition=args.condition,
        dataset_name=args.dataset,
        split=args.split,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        pretrained=False,
        transform_condition=args.transform_condition,
    )
    save_artifacts(
        frame_df=frame_df,
        video_df=video_df,
        metrics=metrics,
        prefix=args.prefix,
        prediction_dir=resolve_path(config["paths"]["prediction_dir"]),
        figure_dir=resolve_path(config["paths"]["figure_dir"]),
        table_dir=resolve_path(config["paths"]["table_dir"]),
    )
    print(metrics)


if __name__ == "__main__":
    main()
