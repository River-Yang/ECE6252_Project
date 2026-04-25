from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import load_config, resolve_path


# Fixed knobs for the simulated signal layer.
PROVENANCE_BASE_PROBS = {
    0: {"verified": 0.65, "missing": 0.25, "broken": 0.10},
    1: {"verified": 0.10, "missing": 0.55, "broken": 0.35},
}

WATERMARK_BASE_PROBS = {
    0: {"present": 0.45, "recoverable": 0.20, "absent": 0.35},
    1: {"present": 0.08, "recoverable": 0.12, "absent": 0.80},
}

SHIFT_PROV_DEGRADE = {
    "clean": 0.05,
    "jpeg_mild": 0.12,
    "blur_mild": 0.10,
    "crop_resize": 0.22,
    "mixed_severe": 0.35,
    "cross_dataset": 0.15,
}

SHIFT_WM_DEGRADE = {
    "clean": 0.03,
    "jpeg_mild": 0.08,
    "blur_mild": 0.12,
    "crop_resize": 0.18,
    "mixed_severe": 0.28,
    "cross_dataset": 0.12,
}

PROV_DELTA = {
    "verified": -0.25,
    "recovered": -0.15,
    "missing": 0.10,
    "broken": 0.30,
}

WM_DELTA = {
    "present": -0.10,
    "recoverable": -0.05,
    "absent": 0.05,
}

ACTION_ORDER = ["allow", "label", "review", "block"]


@dataclass(frozen=True)
class PredictionSpec:
    dataset: str
    condition: str
    path: Path
    shift_type: str
    severity: str


def sample_from_probs(rng: np.random.Generator, probs: dict[str, float]) -> str:
    labels = list(probs)
    values = np.array([probs[label] for label in labels], dtype=float)
    values = values / values.sum()
    return str(rng.choice(labels, p=values))


def condition_to_shift(condition: str, dataset: str) -> tuple[str, str]:
    # Map eval conditions into coarser policy buckets.
    if dataset == "dfdc":
        return "cross_dataset", "cross_dataset"
    if condition == "clean":
        return "clean", "none"
    if condition in {"jpeg_q95", "jpeg_q75"}:
        return "jpeg_mild", "mild"
    if condition == "jpeg_q50":
        return "jpeg_mild", "strong"
    if condition == "blur_s1":
        return "blur_mild", "mild"
    if condition == "blur_s2":
        return "blur_mild", "moderate"
    if condition in {"crop_90", "crop_80", "resize_reencode"}:
        return "crop_resize", "moderate"
    if condition == "combo_comp_blur_noise":
        return "mixed_severe", "severe"
    return "mixed_severe", "unknown"


def build_default_specs(prediction_dir: Path) -> list[PredictionSpec]:
    ffpp_conditions = [
        ("clean", "ffpp_test_clean_video_predictions.csv"),
        ("jpeg_q95", "ffpp_test_jpeg_q95_video_predictions.csv"),
        ("jpeg_q75", "ffpp_test_jpeg_q75_video_predictions.csv"),
        ("jpeg_q50", "ffpp_test_jpeg_q50_video_predictions.csv"),
        ("blur_s1", "ffpp_test_blur_s1_video_predictions.csv"),
        ("blur_s2", "ffpp_test_blur_s2_video_predictions.csv"),
        ("crop_90", "ffpp_test_crop_90_video_predictions.csv"),
        ("crop_80", "ffpp_test_crop_80_video_predictions.csv"),
        ("resize_reencode", "ffpp_test_resize_reencode_video_predictions.csv"),
        ("combo_comp_blur_noise", "ffpp_test_external_shift_video_predictions.csv"),
    ]
    specs: list[PredictionSpec] = []
    for condition, filename in ffpp_conditions:
        shift_type, severity = condition_to_shift(condition, "ffpp")
        specs.append(
            PredictionSpec(
                dataset="ffpp",
                condition=condition,
                path=prediction_dir / filename,
                shift_type=shift_type,
                severity=severity,
            )
        )

    shift_type, severity = condition_to_shift("clean", "dfdc")
    specs.append(
        PredictionSpec(
            dataset="dfdc",
            condition="clean_cross_dataset",
            path=prediction_dir / "dfdc_test_clean_video_predictions.csv",
            shift_type=shift_type,
            severity=severity,
        )
    )
    return specs


def degrade_provenance(rng: np.random.Generator, prov: str, shift_type: str) -> str:
    # Heavier post-processing breaks provenance more often.
    if prov == "verified" and rng.random() < SHIFT_PROV_DEGRADE[shift_type]:
        return sample_from_probs(rng, {"missing": 0.55, "broken": 0.45})
    return prov


def degrade_watermark(rng: np.random.Generator, wm: str, shift_type: str) -> str:
    # Recoverable is the middle state between present and absent.
    degrade_prob = SHIFT_WM_DEGRADE[shift_type]
    if wm == "present" and rng.random() < degrade_prob:
        return sample_from_probs(rng, {"recoverable": 0.60, "absent": 0.40})
    if wm == "recoverable" and rng.random() < degrade_prob * 0.5:
        return "absent"
    return wm


def recover_provenance(rng: np.random.Generator, prov: str, wm: str) -> tuple[str, int]:
    if prov != "missing":
        return prov, 0
    if wm == "present" and rng.random() < 0.70:
        return "recovered", 1
    if wm == "recoverable" and rng.random() < 0.40:
        return "recovered", 1
    return prov, 0


def simulate_authenticity_signals(
    predictions: pd.DataFrame,
    spec: PredictionSpec,
    rng: np.random.Generator,
    base_signal_cache: dict[tuple[str, str], tuple[str, str]],
) -> pd.DataFrame:
    # Keep each video's base signal draw fixed across conditions.
    rows: list[dict[str, object]] = []
    for row in predictions.itertuples(index=False):
        label = int(row.label)
        cache_key = (spec.dataset, str(row.video_id))
        if cache_key not in base_signal_cache:
            base_signal_cache[cache_key] = (
                sample_from_probs(rng, PROVENANCE_BASE_PROBS[label]),
                sample_from_probs(rng, WATERMARK_BASE_PROBS[label]),
            )
        prov_base, wm_base = base_signal_cache[cache_key]
        prov_raw = degrade_provenance(rng, prov_base, spec.shift_type)
        wm = degrade_watermark(rng, wm_base, spec.shift_type)
        prov_final, was_recovered = recover_provenance(rng, prov_raw, wm)
        rows.append(
            {
                "sample_id": str(row.video_id),
                "video_id": str(row.video_id),
                "dataset": spec.dataset,
                "condition": spec.condition,
                "shift_type": spec.shift_type,
                "severity": spec.severity,
                "y_true": label,
                "p_fake": float(row.score),
                "pred": int(row.pred) if hasattr(row, "pred") else int(float(row.score) > 0.5),
                "provenance_status_base": prov_base,
                "provenance_status_raw": prov_raw,
                "provenance_status": prov_final,
                "watermark_status_base": wm_base,
                "watermark_status": wm,
                "was_recovered": was_recovered,
            }
        )
    return pd.DataFrame(rows)


def clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def decide_action(risk: float) -> str:
    if risk < 0.35:
        return "allow"
    if risk < 0.60:
        return "label"
    if risk < 0.80:
        return "review"
    return "block"


def apply_policy(row: pd.Series, policy: str) -> tuple[float, str]:
    p_fake = float(row["p_fake"])
    prov_raw = str(row["provenance_status_raw"])
    prov_final = str(row["provenance_status"])
    wm = str(row["watermark_status"])

    if policy == "detector_only":
        risk = p_fake
    elif policy == "detector_provenance":
        risk = p_fake + PROV_DELTA[prov_raw]
    elif policy == "detector_watermark":
        risk = p_fake + WM_DELTA[wm]
    elif policy == "detector_provenance_watermark":
        risk = p_fake + PROV_DELTA[prov_final] + WM_DELTA[wm]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    risk = clip_score(risk)

    if policy == "detector_provenance_watermark":
        if prov_final in {"verified", "recovered"} and wm in {"present", "recoverable"} and p_fake < 0.75:
            return risk, "label"
        if prov_final == "broken" and wm == "absent" and p_fake > 0.70:
            return risk, "block"

    return risk, decide_action(risk)


def evaluate_policy(signals: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Same grouping as the shift summary.
    decisions = signals.copy()
    decisions[["risk_score", "action"]] = decisions.apply(
        lambda row: pd.Series(apply_policy(row, policy)),
        axis=1,
    )
    decisions["policy"] = policy

    rows: list[dict[str, object]] = []
    for (dataset, condition, shift_type, severity), group in decisions.groupby(
        ["dataset", "condition", "shift_type", "severity"],
        sort=False,
    ):
        real_mask = group["y_true"] == 0
        fake_mask = group["y_true"] == 1
        block_mask = group["action"] == "block"
        review_mask = group["action"] == "review"
        label_mask = group["action"] == "label"
        allow_mask = group["action"] == "allow"
        total = len(group)
        real_count = max(int(real_mask.sum()), 1)
        fake_count = max(int(fake_mask.sum()), 1)
        block_count = int(block_mask.sum())

        rows.append(
            {
                "policy": policy,
                "dataset": dataset,
                "condition": condition,
                "shift_type": shift_type,
                "severity": severity,
                "n_samples": total,
                "n_real": int(real_mask.sum()),
                "n_fake": int(fake_mask.sum()),
                "false_alarm_exposure": float(((real_mask) & (review_mask | block_mask)).sum() / real_count),
                "missed_risky_fake": float(((fake_mask) & (allow_mask | label_mask)).sum() / fake_count),
                "unsafe_pass": float(((fake_mask) & allow_mask).sum() / fake_count),
                "review_load": float(review_mask.sum() / max(total, 1)),
                "auto_block_precision": float(((fake_mask) & block_mask).sum() / block_count) if block_count else 0.0,
                "block_rate": float(block_mask.sum() / max(total, 1)),
                "label_rate": float(label_mask.sum() / max(total, 1)),
                "allow_rate": float(allow_mask.sum() / max(total, 1)),
                "mean_risk_score": float(group["risk_score"].mean()),
            }
        )
    return decisions, pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    plot_df = summary.copy()
    plot_df["condition_label"] = plot_df["dataset"] + ":" + plot_df["condition"]
    plt.figure(figsize=(13, 4.8))
    sns.barplot(data=plot_df, x="condition_label", y=metric, hue="policy")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.ylabel(metric)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def load_prediction_file(spec: PredictionSpec) -> pd.DataFrame | None:
    if not spec.path.exists():
        print(f"Skipping missing prediction file: {spec.path}")
        return None
    df = pd.read_csv(spec.path)
    required = {"video_id", "label", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{spec.path} is missing required columns: {sorted(missing)}")
    if "pred" not in df.columns:
        df["pred"] = (df["score"] > 0.5).astype(int)
    return df[["video_id", "label", "score", "pred"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate detector, provenance, watermark, and fused layered policy variants."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary-name", default="authenticity_policy_summary.csv")
    parser.add_argument("--signals-name", default="authenticity_signals_all_conditions.csv")
    parser.add_argument("--decisions-name", default="authenticity_policy_decisions.csv")
    parser.add_argument("--figure-prefix", default="authenticity_policy")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    rng = np.random.default_rng(seed)

    prediction_dir = resolve_path(config["paths"]["prediction_dir"])
    table_dir = resolve_path(config["paths"]["table_dir"])
    figure_dir = resolve_path(config["paths"]["figure_dir"])

    specs = build_default_specs(prediction_dir)

    signal_frames: list[pd.DataFrame] = []
    base_signal_cache: dict[tuple[str, str], tuple[str, str]] = {}
    for spec in specs:
        predictions = load_prediction_file(spec)
        if predictions is None:
            continue
        signal_frames.append(simulate_authenticity_signals(predictions, spec, rng, base_signal_cache))

    if not signal_frames:
        raise RuntimeError("No prediction files were found for authenticity policy evaluation.")

    signals = pd.concat(signal_frames, ignore_index=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    signals_path = table_dir.parent / "predictions" / args.signals_name
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(signals_path, index=False)

    policies = [
        "detector_only",
        "detector_provenance",
        "detector_watermark",
        "detector_provenance_watermark",
    ]
    decision_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for policy in policies:
        decisions, summary = evaluate_policy(signals, policy)
        decision_frames.append(decisions)
        summary_frames.append(summary)

    all_decisions = pd.concat(decision_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)

    decisions_path = table_dir.parent / "predictions" / args.decisions_name
    summary_path = table_dir / args.summary_name
    all_decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)

    for metric in [
        "false_alarm_exposure",
        "missed_risky_fake",
        "unsafe_pass",
        "review_load",
        "auto_block_precision",
    ]:
        plot_metric(summary, metric, figure_dir / f"{args.figure_prefix}_{metric}.png")

    print(f"Saved authenticity signals to {signals_path}")
    print(f"Saved policy decisions to {decisions_path}")
    print(f"Saved policy summary to {summary_path}")


if __name__ == "__main__":
    main()
