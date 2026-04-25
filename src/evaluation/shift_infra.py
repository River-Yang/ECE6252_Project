from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.metrics import compute_binary_metrics
from src.training.inference import predict_manifest
from src.utils.config import load_config, resolve_path
from src.utils.runtime import set_seed, write_json


def evaluate_manifest(
    manifest: Path,
    checkpoint: Path,
    config_path: str,
    eval_name: str,
    shift_name: str,
    is_baseline: bool,
    frame_level: bool,
) -> dict[str, object]:
    config = load_config(config_path)
    frame_df, _, metrics = predict_manifest(
        manifest_path=resolve_path(manifest),
        checkpoint_path=resolve_path(checkpoint),
        image_size=config["data"]["image_size"],
        threshold=config["eval"]["threshold"],
        condition="clean",
        dataset_name=None,
        split="test",
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        pretrained=False,
    )
    if frame_level:
        metrics = compute_binary_metrics(
            frame_df["label"],
            frame_df["score"],
            threshold=config["eval"]["threshold"],
        )
        metrics["auc"] = metrics["roc_auc"]
    return {
        "eval_name": eval_name,
        "shift_name": shift_name,
        "is_baseline": is_baseline,
        "eval_level": "frame" if frame_level else "video",
        "manifest_path": str(resolve_path(manifest)),
        **metrics,
    }


def infer_process_manifest(config: dict, shift_name: str) -> Path:
    shift_root = resolve_path(config["paths"]["ffpp_shift_root"])
    candidates = [
        shift_root / f"test_shift_{shift_name}" / f"test_shift_{shift_name}_manifest.csv",
        shift_root / shift_name / f"{shift_name}_manifest.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a process manifest for shift '{shift_name}' under {shift_root}"
    )


def build_shift_eval_manifest(
    baseline_manifest: Path,
    process_manifest: Path,
    output_path: Path,
) -> Path:
    # Build eval CSVs for pre-shifted image folders.
    base_df = pd.read_csv(resolve_path(baseline_manifest))
    process_df = pd.read_csv(resolve_path(process_manifest))

    required_base_cols = {"face_path", "label", "split", "video_id", "frame_id"}
    required_process_cols = {"relative_path", "absolute_path"}
    missing_base = required_base_cols - set(base_df.columns)
    missing_process = required_process_cols - set(process_df.columns)
    if missing_base:
        raise ValueError(f"Baseline manifest is missing columns: {sorted(missing_base)}")
    if missing_process:
        raise ValueError(f"Process manifest is missing columns: {sorted(missing_process)}")

    test_mask = base_df["split"] == "test"
    relative_paths = (
        base_df.loc[test_mask, "face_path"]
        .astype(str)
        .str.extract(r"/test/(.+)$", expand=False)
    )
    if relative_paths.isna().any():
        raise ValueError("Could not derive relative test paths from baseline manifest face_path values.")

    process_map = (
        process_df[["relative_path", "absolute_path"]]
        .drop_duplicates(subset=["relative_path"])
        .set_index("relative_path")["absolute_path"]
    )
    resolved_paths = relative_paths.map(process_map)
    if resolved_paths.isna().any():
        missing = relative_paths[resolved_paths.isna()].head(5).tolist()
        raise ValueError(f"Missing shifted image paths for test samples, examples: {missing}")

    merged_df = base_df.copy()
    merged_df.loc[test_mask, "face_path"] = resolved_paths.values
    if "applied_shifts" in process_df.columns:
        shift_map = (
            process_df[["relative_path", "applied_shifts"]]
            .drop_duplicates(subset=["relative_path"])
            .set_index("relative_path")["applied_shifts"]
        )
        merged_df.loc[test_mask, "applied_shifts"] = relative_paths.map(shift_map).values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    return output_path


def add_delta_from_baseline(df: pd.DataFrame) -> pd.DataFrame:
    baseline_df = df[df["is_baseline"]].copy()
    if baseline_df.empty:
        df["delta_auc"] = 0.0
        df["delta_fnr"] = 0.0
        return df

    baseline_row = baseline_df.iloc[0]
    df = df.copy()
    df["delta_auc"] = float(baseline_row["auc"]) - df["auc"]
    df["delta_fnr"] = df["fnr"] - float(baseline_row["fnr"])
    return df


def plot_bar(df: pd.DataFrame, metric: str, outpath: Path, title: str | None = None) -> None:
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="eval_name", y=metric, color="#3c78d8")
    plt.xticks(rotation=20, ha="right")
    plt.title(title or f"{metric} by evaluation")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def maybe_read_process_file(process_file: str | None) -> str | None:
    if not process_file:
        return None
    path = Path(process_file)
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return " | ".join(line for line in lines if not line.startswith("#")) or None


def save_metric_jsons(df: pd.DataFrame, table_dir: Path, prefix: str) -> None:
    for row in df.to_dict(orient="records"):
        eval_name = str(row["eval_name"])
        payload = {
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "roc_auc": row["roc_auc"],
            "fpr": row["fpr"],
            "fnr": row["fnr"],
            "auc": row["auc"],
        }
        write_json(payload, table_dir / f"{prefix}_{eval_name}_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple baseline vs manual-shift evaluation and save a plot-ready log.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--baseline-manifest", default=None)
    parser.add_argument("--shift-manifest", default=None)
    parser.add_argument("--shift-process-manifest", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shift-name", default="manual_shift")
    parser.add_argument("--frame", action="store_true", help="Compute metrics at frame level instead of aggregated video level.")
    parser.add_argument("--metric", default="auc")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--process-file", default=None, help="Optional path to process notes for logging only.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    baseline_manifest = Path(args.baseline_manifest or config["paths"]["ffpp_face_manifest"])

    shift_manifest = Path(args.shift_manifest) if args.shift_manifest else None
    shift_process_manifest = Path(args.shift_process_manifest) if args.shift_process_manifest else None
    if shift_manifest is None and shift_process_manifest is None and args.shift_name != "manual_shift":
        shift_process_manifest = infer_process_manifest(config, args.shift_name)
    if shift_manifest is None and shift_process_manifest is not None:
        safe_shift_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.shift_name)
        generated_manifest = resolve_path(config["paths"]["table_dir"]) / f"{args.prefix}_{safe_shift_name}_eval_manifest.csv"
        shift_manifest = build_shift_eval_manifest(
            baseline_manifest=baseline_manifest,
            process_manifest=shift_process_manifest,
            output_path=generated_manifest,
        )

    rows = [
        evaluate_manifest(
            manifest=baseline_manifest,
            checkpoint=Path(args.checkpoint),
            config_path=args.config,
            eval_name="baseline",
            shift_name="no_shift",
            is_baseline=True,
            frame_level=args.frame,
        )
    ]
    if shift_manifest:
        rows.append(
            evaluate_manifest(
                manifest=shift_manifest,
                checkpoint=Path(args.checkpoint),
                config_path=args.config,
                eval_name=args.shift_name,
                shift_name=args.shift_name,
                is_baseline=False,
                frame_level=args.frame,
            )
        )

    df = add_delta_from_baseline(pd.DataFrame(rows))
    process_note = maybe_read_process_file(args.process_file)
    if process_note is not None:
        df["process_note"] = process_note

    table_path = resolve_path(config["paths"]["table_dir"]) / f"{args.prefix}_shift_log.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)
    save_metric_jsons(df, table_path.parent, args.prefix)

    plot_path = resolve_path("results/plot") / "shift_type.csv"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    if plot_path.exists():
        df.to_csv(plot_path, index=False, mode="a", header=False)
    else:
        df.to_csv(plot_path, index=False)

    if not args.no_plot and len(df["eval_name"].unique()) > 1:
        figure_path = resolve_path(config["paths"]["figure_dir"]) / f"{args.prefix}_{args.metric}_bar.png"
        plot_bar(df, metric=args.metric, outpath=figure_path, title=f"{args.metric} for {args.shift_name}")
        print(f"Wrote plot to {figure_path}")

    print(f"Wrote log to {table_path}")


if __name__ == "__main__":
    main()
