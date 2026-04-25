from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import load_config, resolve_path


REPORT_SHIFT_ORDER = [
    "clean",
    "jpeg_q95",
    "jpeg_q75",
    "jpeg_q50",
    "jpeg_q30",
    "blur_s1",
    "blur_s2",
    "gaussian_noise_s8",
    "speckle_s005",
    "pixelate_b8",
    "brightness_07",
    "brightness_13",
    "contrast_07",
    "crop_90",
    "crop_80",
    "resize_reencode",
    "combo_comp_blur_noise",
]

# Keep plots in the same order as the report table.
FOCUS_SHIFT_CONDITIONS = [
    "clean",
    "jpeg_q50",
    "gaussian_noise_s8",
    "speckle_s005",
    "blur_s2",
    "pixelate_b8",
    "combo_comp_blur_noise",
]
FOCUS_POLICY_CONDITIONS = ["clean", "jpeg_q50", "blur_s2", "combo_comp_blur_noise", "clean_cross_dataset"]
POLICY_ORDER = [
    "detector_only",
    "detector_provenance",
    "detector_watermark",
    "detector_provenance_watermark",
]
POLICY_LABELS = {
    "detector_only": "Detector Only",
    "detector_provenance": "Detector + Provenance",
    "detector_watermark": "Detector + Watermark",
    "detector_provenance_watermark": "Detector + Provenance + Watermark",
}


def load_json(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_plot(output_path: Path) -> None:
    # Shared figure save settings.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_core_metrics(ffpp_metrics: dict[str, float], dfdc_metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "FF++ clean",
                "accuracy": ffpp_metrics["accuracy"],
                "f1": ffpp_metrics["f1"],
                "auc": ffpp_metrics["auc"],
                "fpr": ffpp_metrics["fpr"],
                "fnr": ffpp_metrics["fnr"],
            },
            {
                "dataset": "DFDC cross-dataset",
                "accuracy": dfdc_metrics["accuracy"],
                "f1": dfdc_metrics["f1"],
                "auc": dfdc_metrics["auc"],
                "fpr": dfdc_metrics["fpr"],
                "fnr": dfdc_metrics["fnr"],
            },
        ]
    )


def plot_core_metrics(core_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = core_df.melt(id_vars="dataset", value_vars=["accuracy", "f1", "auc", "fnr"], var_name="metric", value_name="value")
    plt.figure(figsize=(8.5, 4.6))
    sns.barplot(data=plot_df, x="metric", y="value", hue="dataset")
    plt.ylim(0, 1.05)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("FF++ Clean vs DFDC Cross-Dataset")
    save_plot(output_path)


def plot_shift_overview(shift_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = shift_df.copy()
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=REPORT_SHIFT_ORDER, ordered=True)
    plot_df = plot_df.sort_values("condition")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    sns.barplot(data=plot_df, x="condition", y="auc", color="#3c78d8", ax=axes[0])
    axes[0].set_ylim(0.85, 1.01)
    axes[0].set_title("FF++ Shift Robustness Overview")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("AUC")

    sns.barplot(data=plot_df, x="condition", y="fnr", color="#cc4125", ax=axes[1])
    axes[1].set_ylim(0, 0.95)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("FNR")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    save_plot(output_path)


def plot_shift_failure_modes(shift_focus_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = shift_focus_df.melt(
        id_vars="condition",
        value_vars=["fpr", "fnr"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(8.8, 4.6))
    sns.barplot(data=plot_df, x="condition", y="value", hue="metric")
    plt.ylim(0, 0.95)
    plt.xlabel("")
    plt.ylabel("Rate")
    plt.title("Different Shifts Trigger Different Failure Modes")
    save_plot(output_path)


def plot_policy_ffpp(policy_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = policy_df[
        policy_df["condition"].isin(["jpeg_q50", "blur_s2", "combo_comp_blur_noise"])
    ].copy()
    plot_df["policy_label"] = plot_df["policy"].map(POLICY_LABELS)
    plot_df["condition"] = pd.Categorical(
        plot_df["condition"],
        categories=["jpeg_q50", "blur_s2", "combo_comp_blur_noise"],
        ordered=True,
    )
    plot_df["policy_label"] = pd.Categorical(
        plot_df["policy_label"],
        categories=[POLICY_LABELS[name] for name in POLICY_ORDER],
        ordered=True,
    )

    metrics = ["unsafe_pass", "false_alarm_exposure", "review_load"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True)
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=plot_df, x="policy_label", y=metric, hue="condition", ax=ax)
        ax.set_ylim(0, 0.95)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        if metric != "unsafe_pass":
            ax.get_legend().remove()
    axes[0].set_title("FF++ Policy Trade-offs under Representative Shifts")
    axes[-1].tick_params(axis="x", rotation=20)
    save_plot(output_path)


def plot_policy_dfdc(policy_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = policy_df[policy_df["condition"] == "clean_cross_dataset"].copy()
    plot_df["policy_label"] = pd.Categorical(
        plot_df["policy"].map(POLICY_LABELS),
        categories=[POLICY_LABELS[name] for name in POLICY_ORDER],
        ordered=True,
    )
    plot_df = plot_df.melt(
        id_vars="policy_label",
        value_vars=["unsafe_pass", "missed_risky_fake", "false_alarm_exposure", "review_load"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 4.8))
    sns.barplot(data=plot_df, x="metric", y="value", hue="policy_label")
    plt.ylim(0, 0.95)
    plt.xlabel("")
    plt.ylabel("Rate")
    plt.title("Policy Trade-offs on DFDC Cross-Dataset")
    save_plot(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final-report-ready tables and figures from existing results.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    # Uses saved outputs; no model inference here.
    config = load_config(args.config)
    table_dir = resolve_path(config["paths"]["table_dir"])
    figure_dir = resolve_path(config["paths"]["figure_dir"])

    ffpp_metrics = load_json(table_dir / "ffpp_test_metrics.json")
    dfdc_metrics = load_json(table_dir / "dfdc_test_clean_metrics.json")
    cross_drop = load_json(table_dir / "cross_dataset_drop.json")
    shift_df = pd.read_csv(table_dir / "shift_metrics_summary.csv")
    policy_df = pd.read_csv(table_dir / "authenticity_policy_summary.csv")

    # Tables copied into the report.
    core_df = build_core_metrics(ffpp_metrics, dfdc_metrics)
    shift_df = shift_df[shift_df["dataset"] == "ffpp"].copy()
    shift_focus_df = shift_df[shift_df["condition"].isin(FOCUS_SHIFT_CONDITIONS)].copy()
    shift_focus_df["condition"] = pd.Categorical(
        shift_focus_df["condition"],
        categories=FOCUS_SHIFT_CONDITIONS,
        ordered=True,
    )
    shift_focus_df = shift_focus_df.sort_values("condition")
    policy_focus_df = policy_df[policy_df["condition"].isin(FOCUS_POLICY_CONDITIONS)].copy()

    cross_drop_df = pd.DataFrame(
        [
            {
                "comparison": "FF++ clean -> DFDC cross-dataset",
                **cross_drop,
            }
        ]
    )

    core_df.to_csv(table_dir / "report_core_metrics.csv", index=False)
    shift_focus_df.to_csv(table_dir / "report_shift_focus.csv", index=False)
    policy_focus_df.to_csv(table_dir / "report_policy_focus.csv", index=False)
    cross_drop_df.to_csv(table_dir / "report_cross_dataset_drop.csv", index=False)

    plot_core_metrics(core_df, figure_dir / "report_core_metrics.png")
    plot_shift_overview(shift_df, figure_dir / "report_shift_overview.png")
    plot_shift_failure_modes(shift_focus_df, figure_dir / "report_shift_failure_modes.png")
    plot_policy_ffpp(policy_focus_df, figure_dir / "report_policy_ffpp.png")
    plot_policy_dfdc(policy_focus_df, figure_dir / "report_policy_dfdc.png")

    print(f"Saved final report tables to {table_dir}")
    print(f"Saved final report figures to {figure_dir}")


if __name__ == "__main__":
    main()
