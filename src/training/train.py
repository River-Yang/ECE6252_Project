from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets.frame_dataset import FaceFrameDataset
from src.models.resnet50_binary import build_resnet50_binary
from src.utils.config import load_config, resolve_path
from src.utils.runtime import get_device, set_seed


def build_train_transform(image_size: int) -> transforms.Compose:
    # Add light augmentation for the training split.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    # Keep validation preprocessing deterministic.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loader(
    manifest_path: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train_mode: bool,
) -> DataLoader:
    # Read the requested split from the shared manifest.
    dataset = FaceFrameDataset(
        manifest_path=manifest_path,
        split=split,
        transform=build_train_transform(image_size) if train_mode else build_eval_transform(image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_mode,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Adam | None, device: torch.device) -> tuple[float, list[float], list[int]]:
    is_train = optimizer is not None
    # Toggle train/eval mode from whether an optimizer is present.
    model.train(is_train)
    losses: list[float] = []
    scores: list[float] = []
    labels: list[int] = []

    iterator = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for batch in iterator:
        images = batch["image"].to(device)
        targets = batch["label"].float().to(device).unsqueeze(1)

        # Only backprop during the training pass.
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist()
        scores.extend(probs)
        labels.extend(batch["label"].int().tolist())
        losses.append(float(loss.item()))
        iterator.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

    return sum(losses) / max(len(losses), 1), scores, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ResNet50 deepfake baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--face-manifest", default=None)
    parser.add_argument(
        "--checkpoint-name",
        default="resnet50_best_auc.pt",
        help="Checkpoint file name written under results/checkpoints.",
    )
    parser.add_argument(
        "--history-name",
        default="train_history.csv",
        help="Training history CSV file name written under results/logs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device()

    manifest_path = resolve_path(args.face_manifest or config["paths"]["ffpp_face_manifest"])
    train_cfg = config["train"]
    image_size = config["data"]["image_size"]

    # Build separate loaders for train and validation splits.
    train_loader = build_loader(
        manifest_path=manifest_path,
        split="train",
        image_size=image_size,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        train_mode=True,
    )
    val_loader = build_loader(
        manifest_path=manifest_path,
        split="val",
        image_size=image_size,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        train_mode=False,
    )

    model = build_resnet50_binary(pretrained=config["model"]["pretrained"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    checkpoint_dir = resolve_path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Let the caller override artifact names without changing defaults.
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    history: list[dict[str, float | int]] = []
    best_auc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_scores, val_labels = run_epoch(model, val_loader, criterion, None, device)
        val_auc = roc_auc_score(val_labels, val_scores) if len(set(val_labels)) > 1 else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": float(val_auc),
            }
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            # Save the checkpoint with the best validation AUC so far.
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= train_cfg["early_stop_patience"]:
            print("Early stopping triggered.")
            break

    # Save one CSV row per epoch for later plotting.
    history_path = resolve_path(config["paths"]["log_dir"]) / args.history_name
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved best checkpoint to {checkpoint_path}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
