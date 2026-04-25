#!/usr/bin/env python3
"""Build eval manifests by swapping test paths to shifted images."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def logicalize_path(path: Path) -> Path:
    path = path.resolve()
    home = Path.home()
    user = home.name
    storage_prefix = Path("/storage/ice1/3/1") / user
    if path == storage_prefix or storage_prefix in path.parents:
        relative = path.relative_to(storage_prefix)
        return home / "scratch" / relative
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create eval manifests by replacing the test split face paths in a base "
            "FFPP manifest with shifted test-set image paths."
        )
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("faces/ffpp_faces.csv"),
        help="Base FFPP manifest CSV. Default: faces/ffpp_faces.csv",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("faces/images"),
        help="Root directory containing test_shift_* folders. Default: faces/images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("faces/images"),
        help="Directory for generated eval manifests. Default: faces/images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_manifest_path = args.base_manifest.resolve()
    images_root = args.images_root.resolve()
    output_dir = args.output_dir.resolve()

    base_manifest = pd.read_csv(base_manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    shift_manifest_paths = sorted(images_root.glob("test_shift_*/test_shift_*_manifest.csv"))
    if not shift_manifest_paths:
        raise RuntimeError(f"No shift manifests found under: {images_root}")

    base_test = base_manifest[base_manifest["split"] == "test"].copy()
    base_test["rel_from_test"] = base_test["face_path"].astype(str).str.extract(
        r"/test/(.+)$", expand=False
    )

    for proc_manifest_path in shift_manifest_paths:
        proc = pd.read_csv(proc_manifest_path)
        proc["relative_path"] = proc["relative_path"].astype(str)
        proc["shift_face_path"] = proc["relative_path"].map(
            lambda rel: str(logicalize_path(proc_manifest_path.parent / rel))
        )

        test_map = base_test.merge(
            proc[["relative_path", "shift_face_path", "applied_shifts"]],
            left_on="rel_from_test",
            right_on="relative_path",
            how="left",
        )

        merged = base_manifest.copy()
        test_mask = merged["split"] == "test"
        merged.loc[test_mask, "face_path"] = test_map["shift_face_path"].values
        merged.loc[test_mask, "shift_name"] = proc_manifest_path.stem.replace("_manifest", "")
        merged.loc[test_mask, "applied_shifts"] = test_map["applied_shifts"].values

        out_path = output_dir / f"{proc_manifest_path.stem}_eval.csv"
        merged.drop(columns=["rel_from_test"], errors="ignore").to_csv(out_path, index=False)
        print(logicalize_path(out_path))


if __name__ == "__main__":
    main()
