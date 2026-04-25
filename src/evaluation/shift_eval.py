from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import plot_metric_by_condition
from src.training.inference import predict_manifest, save_artifacts
from src.utils.config import load_config, resolve_path
from src.utils.runtime import set_seed


def evaluate_dataset(
    config: dict,
    manifest_path: str | Path,
    checkpoint: str | Path,
    dataset_name: str,
    summary_dataset_name: str,
    conditions: list[str],
    baseline_auc: float | None = None,
    baseline_fnr: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clean_auc = baseline_auc
    clean_fnr = baseline_fnr

    for condition in conditions:
        prefix = f"{dataset_name}_test_{condition}"
        frame_df, video_df, metrics = predict_manifest(
            manifest_path=resolve_path(manifest_path),
            checkpoint_path=resolve_path(checkpoint),
            image_size=config["data"]["image_size"],
            threshold=config["eval"]["threshold"],
            condition=condition,
            dataset_name=dataset_name,
            split="test",
            batch_size=config["eval"].get("batch_size", config["train"]["batch_size"]),
            num_workers=config["eval"].get("num_workers", config["train"]["num_workers"]),
            pretrained=False,
        )
        save_artifacts(
            frame_df=frame_df,
            video_df=video_df,
            metrics=metrics,
            prefix=prefix,
            prediction_dir=resolve_path(config["paths"]["prediction_dir"]),
            figure_dir=resolve_path(config["paths"]["figure_dir"]),
            table_dir=resolve_path(config["paths"]["table_dir"]),
        )
        if condition == "clean":
            clean_auc = metrics["auc"]
            clean_fnr = metrics["fnr"]
        rows.append(
            {
                "dataset": summary_dataset_name,
                "condition": condition,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "fpr": metrics["fpr"],
                "fnr": metrics["fnr"],
                "delta_auc": 0.0,
                "delta_fnr": 0.0,
            }
        )

    summary = pd.DataFrame(rows)
    if clean_auc is not None and clean_fnr is not None:
        # Deltas use the clean split from the same dataset.
        summary["delta_auc"] = clean_auc - summary["auc"]
        summary["delta_fnr"] = summary["fnr"] - clean_fnr
    return summary


def append_external_shift_summary(
    summary: pd.DataFrame,
    config: dict,
    checkpoint: str | Path,
) -> pd.DataFrame:
    manifest_path = resolve_path(config["paths"]["ffpp_external_shift_manifest"])
    if not manifest_path.exists():
        return summary
    condition_label = config["shifts"].get("ffpp_external_condition_label", "external_shift")

    # Pre-shifted images; do not apply another transform.
    frame_df, video_df, metrics = predict_manifest(
        manifest_path=manifest_path,
        checkpoint_path=resolve_path(checkpoint),
        image_size=config["data"]["image_size"],
        threshold=config["eval"]["threshold"],
        condition=condition_label,
        dataset_name="ffpp_shift_external",
        split="test",
        batch_size=config["eval"].get("batch_size", config["train"]["batch_size"]),
        num_workers=config["eval"].get("num_workers", config["train"]["num_workers"]),
        pretrained=False,
        transform_condition="clean",
    )
    save_artifacts(
        frame_df=frame_df,
        video_df=video_df,
        metrics=metrics,
        prefix="ffpp_test_external_shift",
        prediction_dir=resolve_path(config["paths"]["prediction_dir"]),
        figure_dir=resolve_path(config["paths"]["figure_dir"]),
        table_dir=resolve_path(config["paths"]["table_dir"]),
    )

    clean_row = summary[summary["condition"] == "clean"]
    clean_auc = float(clean_row["auc"].iloc[0]) if not clean_row.empty else metrics["auc"]
    clean_fnr = float(clean_row["fnr"].iloc[0]) if not clean_row.empty else metrics["fnr"]
    external_row = pd.DataFrame(
        [
            {
                "dataset": "ffpp",
                "condition": condition_label,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "fpr": metrics["fpr"],
                "fnr": metrics["fnr"],
                "delta_auc": clean_auc - metrics["auc"],
                "delta_fnr": metrics["fnr"] - clean_fnr,
            }
        ]
    )
    summary = summary[summary["condition"] != condition_label].copy()
    return pd.concat([summary, external_row], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robustness across post-processing shifts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="results/checkpoints/resnet50_best_auc.pt")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ffpp", "dfdc"],
        default=["ffpp", "dfdc"],
        help="Datasets to evaluate. Use this to run FF++ shifts before DFDC artifacts are ready.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Optional subset of conditions to evaluate for the selected dataset(s).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    dataset_specs = {
        "ffpp": {
            "manifest_path": config["paths"]["ffpp_face_manifest"],
            "conditions": config["shifts"]["ffpp_conditions"],
            "auc_plot": "shift_auc_bar_ffpp.png",
            "fnr_plot": "shift_fnr_line_ffpp.png",
            "auc_title": "FF++ AUC vs Shift",
            "fnr_title": "FF++ FNR vs Shift",
        },
        "dfdc": {
            "manifest_path": config["paths"]["dfdc_face_manifest"],
            "conditions": config["shifts"]["dfdc_conditions"],
            "auc_plot": "shift_auc_bar_dfdc.png",
            "fnr_plot": "shift_fnr_line_dfdc.png",
            "auc_title": "DFDC AUC vs Shift",
            "fnr_title": "DFDC FNR vs Shift",
        },
    }

    summaries: list[pd.DataFrame] = []
    figure_dir = resolve_path(config["paths"]["figure_dir"])
    summary_path = resolve_path(config["paths"]["table_dir"]) / "shift_metrics_summary.csv"
    existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    for dataset_name in args.datasets:
        spec = dataset_specs[dataset_name]
        manifest_path = resolve_path(spec["manifest_path"])
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing manifest for dataset '{dataset_name}': {manifest_path}. "
                "Use --datasets to select available datasets."
            )

        selected_conditions = args.conditions or spec["conditions"]
        baseline_auc = None
        baseline_fnr = None
        if not existing_summary.empty:
            clean_row = existing_summary[
                (existing_summary["dataset"] == dataset_name) & (existing_summary["condition"] == "clean")
            ]
            if not clean_row.empty:
                baseline_auc = float(clean_row["auc"].iloc[0])
                baseline_fnr = float(clean_row["fnr"].iloc[0])

        dataset_summary = evaluate_dataset(
            config=config,
            manifest_path=manifest_path,
            checkpoint=args.checkpoint,
            dataset_name=dataset_name,
            summary_dataset_name=dataset_name,
            conditions=selected_conditions,
            baseline_auc=baseline_auc,
            baseline_fnr=baseline_fnr,
        )
        if dataset_name == "ffpp" and (args.conditions is None or "combo_comp_blur_noise" in selected_conditions):
            dataset_summary = append_external_shift_summary(dataset_summary, config=config, checkpoint=args.checkpoint)
        summaries.append(dataset_summary)
        plot_metric_by_condition(dataset_summary, "auc", figure_dir / spec["auc_plot"], spec["auc_title"], kind="bar")
        plot_metric_by_condition(dataset_summary, "fnr", figure_dir / spec["fnr_plot"], spec["fnr_title"], kind="line")

    summary = pd.concat(summaries, ignore_index=True)

    if not existing_summary.empty and args.conditions is not None:
        keep_mask = ~existing_summary["dataset"].isin(args.datasets) | ~existing_summary["condition"].isin(args.conditions)
        existing_summary = existing_summary[keep_mask].copy()
        summary = pd.concat([existing_summary, summary], ignore_index=True)
        summary = summary.drop_duplicates(subset=["dataset", "condition"], keep="last")
        summary = summary.sort_values(["dataset", "condition"]).reset_index(drop=True)

    summary.to_csv(summary_path, index=False)

    print(f"Saved shift summary to {summary_path}")


if __name__ == "__main__":
    main()
