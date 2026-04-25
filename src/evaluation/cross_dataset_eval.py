from __future__ import annotations

import argparse

from src.training.inference import predict_manifest, save_artifacts
from src.utils.config import load_config, resolve_path
from src.utils.runtime import set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation from FF++ checkpoint to DFDC.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="results/checkpoints/resnet50_best_auc.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    prediction_dir = resolve_path(config["paths"]["prediction_dir"])
    figure_dir = resolve_path(config["paths"]["figure_dir"])
    table_dir = resolve_path(config["paths"]["table_dir"])

    # Same checkpoint, new test domain.
    ffpp_frame_df, ffpp_video_df, ffpp_metrics = predict_manifest(
        manifest_path=resolve_path(config["paths"]["ffpp_face_manifest"]),
        checkpoint_path=resolve_path(args.checkpoint),
        image_size=config["data"]["image_size"],
        threshold=config["eval"]["threshold"],
        condition="clean",
        dataset_name="ffpp",
        split="test",
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        pretrained=False,
    )
    dfdc_frame_df, dfdc_video_df, dfdc_metrics = predict_manifest(
        manifest_path=resolve_path(config["paths"]["dfdc_face_manifest"]),
        checkpoint_path=resolve_path(args.checkpoint),
        image_size=config["data"]["image_size"],
        threshold=config["eval"]["threshold"],
        condition="clean",
        dataset_name="dfdc",
        split="test",
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        pretrained=False,
    )

    save_artifacts(ffpp_frame_df, ffpp_video_df, ffpp_metrics, "ffpp_test", prediction_dir, figure_dir, table_dir)
    save_artifacts(dfdc_frame_df, dfdc_video_df, dfdc_metrics, "dfdc_test_clean", prediction_dir, figure_dir, table_dir)

    drop = {
        "delta_auc": ffpp_metrics["auc"] - dfdc_metrics["auc"],
        "delta_f1": ffpp_metrics["f1"] - dfdc_metrics["f1"],
        "delta_fnr": dfdc_metrics["fnr"] - ffpp_metrics["fnr"],
    }
    write_json(drop, table_dir / "cross_dataset_drop.json")
    print(drop)


if __name__ == "__main__":
    main()
