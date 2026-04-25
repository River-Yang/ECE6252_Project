#!/usr/bin/env python3
"""Apply synthetic shifts to an FFPP image folder in place."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SHIFT_ORDER = (
    "dct",
    "cosine4",
    "compression",
    "noise",
    "speckle",
    "blur",
    "resize",
    "pixelate",
    "lighter",
    "dimmer",
    "contrast_down",
)


def logicalize_path(path: Path) -> Path:
    path = path.resolve()
    home = Path.home()
    user = home.name
    storage_prefix = Path("/storage/ice1/3/1") / user
    if path == storage_prefix or storage_prefix in path.parents:
        # Keep manifests readable when the same scratch data is mounted under /storage.
        relative = path.relative_to(storage_prefix)
        return home / "scratch" / relative
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply image shifts to a copied directory such as "
            "ffpp/faces/images/test_shift while keeping the same filenames."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory to process in place, e.g. ffpp/faces/images/test_shift",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional path for the output manifest CSV. Defaults to "
            "<input-dir>/<input-dir-name>_manifest.csv"
        ),
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=None,
        help=(
            "Optional markdown file that describes the applied shift setup. "
            "Defaults to <input-dir>/SHIFT_INFO.md"
        ),
    )
    parser.add_argument(
        "--assignment",
        choices=("single", "combo", "fixed"),
        default="combo",
        help=(
            "'single' gives each image one shift, 'combo' assigns 1..N shifts, "
            "and 'fixed' applies the same selected shifts to every image."
        ),
    )
    parser.add_argument(
        "--shifts",
        nargs="+",
        choices=SHIFT_ORDER,
        default=list(SHIFT_ORDER),
        help="Allowed shifts to sample from.",
    )
    parser.add_argument(
        "--type",
        nargs="+",
        choices=SHIFT_ORDER,
        default=None,
        help=(
            "Shortcut to explicitly choose the shift type(s) to apply. "
            "Example: --type dct or --type compression noise"
        ),
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Fraction of images to modify. Remaining images are left unchanged.",
    )
    parser.add_argument(
        "--mode",
        choices=("default", "media"),
        default="default",
        help="Shift selection mode. 'media' simulates a social-media style pipeline.",
    )
    parser.add_argument(
        "--prob",
        nargs="+",
        default=None,
        help="Per-shift probabilities like: --prob compression=0.9 blur=0.4 noise=0.2",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="Optional fixed JPEG quality for the compression shift, e.g. 30 or 75.",
    )
    parser.add_argument(
        "--max-shifts-per-image",
        type=int,
        default=2,
        help="Maximum number of shifts per image when --assignment=combo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for deterministic shift assignment.",
    )
    args = parser.parse_args()
    if args.type:
        args.shifts = list(args.type)
    return args


def list_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def stable_seed(base_seed: int, relative_path: str) -> int:
    # Path-based seeds make reruns stable even if only part of the folder is processed.
    digest = hashlib.sha256(f"{base_seed}:{relative_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_dct_matrix(size: int) -> np.ndarray:
    # Build the standard DCT-II basis once for block filtering.
    matrix = np.zeros((size, size), dtype=np.float32)
    scale0 = np.sqrt(1.0 / size)
    scale = np.sqrt(2.0 / size)
    for k in range(size):
        alpha = scale0 if k == 0 else scale
        for n in range(size):
            matrix[k, n] = alpha * np.cos(np.pi * (2 * n + 1) * k / (2 * size))
    return matrix


def build_dct4_matrix(size: int) -> np.ndarray:
    # Build the cosine-IV basis used by the smooth global low-pass shift.
    matrix = np.zeros((size, size), dtype=np.float32)
    scale = np.sqrt(2.0 / size)
    for k in range(size):
        for n in range(size):
            matrix[k, n] = scale * np.cos(np.pi * (n + 0.5) * (k + 0.5) / size)
    return matrix


def apply_block_dct_lowpass(
    image: np.ndarray,
    rng: np.random.Generator,
    block_size: int = 8,
) -> tuple[np.ndarray, dict[str, float]]:
    keep_ratio = float(rng.uniform(0.30, 0.55))
    dct_matrix = build_dct_matrix(block_size)
    cutoff = max(1, int(round(block_size * keep_ratio)))

    padded_h = ((image.shape[0] + block_size - 1) // block_size) * block_size
    padded_w = ((image.shape[1] + block_size - 1) // block_size) * block_size
    padded = np.pad(
        image,
        ((0, padded_h - image.shape[0]), (0, padded_w - image.shape[1]), (0, 0)),
        mode="edge",
    ).astype(np.float32)

    output = np.empty_like(padded)
    mask = np.zeros((block_size, block_size), dtype=np.float32)
    mask[:cutoff, :cutoff] = 1.0

    for y in range(0, padded_h, block_size):
        for x in range(0, padded_w, block_size):
            block = padded[y : y + block_size, x : x + block_size, :]
            for channel in range(block.shape[2]):
                channel_block = block[:, :, channel]
                freq = dct_matrix @ channel_block @ dct_matrix.T
                freq *= mask
                restored = dct_matrix.T @ freq @ dct_matrix
                output[y : y + block_size, x : x + block_size, channel] = restored

    cropped = np.clip(output[: image.shape[0], : image.shape[1], :], 0, 255)
    return cropped.astype(np.uint8), {"keep_ratio": round(keep_ratio, 4), "block_size": block_size}


def apply_global_cosine4_lowpass(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width = image.shape[:2]
    keep_ratio = float(rng.uniform(0.28, 0.50))
    rolloff = float(rng.uniform(0.08, 0.18))
    max_work_size = 96

    # The global cosine transform is expensive, so use a smaller work image.
    work_h = min(height, max_work_size)
    work_w = min(width, max_work_size)
    resized = Image.fromarray(image).resize((work_w, work_h), Image.Resampling.BICUBIC)
    work_image = np.array(resized)

    row_matrix = build_dct4_matrix(work_h)
    col_matrix = build_dct4_matrix(work_w)

    row_coords = np.linspace(0.0, 1.0, work_h, endpoint=False, dtype=np.float32)
    col_coords = np.linspace(0.0, 1.0, work_w, endpoint=False, dtype=np.float32)
    yy, xx = np.meshgrid(row_coords, col_coords, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)

    cutoff = keep_ratio * np.sqrt(2.0)
    transition = max(rolloff, 1e-6)
    mask = np.clip((cutoff - radius) / transition, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)

    output = np.empty_like(work_image, dtype=np.float32)
    for channel in range(work_image.shape[2]):
        channel_data = work_image[:, :, channel].astype(np.float32)
        freq = row_matrix @ channel_data @ col_matrix.T
        freq *= mask
        restored = row_matrix.T @ freq @ col_matrix
        output[:, :, channel] = restored

    filtered = np.clip(output, 0, 255).astype(np.uint8)
    restored = Image.fromarray(filtered).resize((width, height), Image.Resampling.BICUBIC)
    return np.array(restored), {
        "keep_ratio": round(keep_ratio, 4),
        "rolloff": round(rolloff, 4),
        "mode": "global_soft_lowpass",
        "work_size": f"{work_h}x{work_w}",
    }


def apply_jpeg_compression(
    image: np.ndarray,
    rng: np.random.Generator,
    quality_override: int | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    quality = quality_override if quality_override is not None else int(rng.integers(18, 45))
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer).convert("RGB"))
    return compressed, {"quality": quality}


def apply_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    sigma = float(rng.uniform(6.0, 18.0))
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8), {"sigma": round(sigma, 4)}


def apply_speckle_noise(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    sigma = float(rng.uniform(0.06, 0.16))
    noise = rng.normal(0.0, sigma, size=image.shape)
    noisy = image.astype(np.float32) + image.astype(np.float32) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8), {"sigma": round(sigma, 4)}


def apply_blur(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    radius = float(rng.uniform(0.8, 2.2))
    blurred = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred), {"radius": round(radius, 4)}


def apply_resize_degradation(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    pil_image = Image.fromarray(image)
    scale = float(rng.uniform(0.35, 0.75))
    down_w = max(1, int(round(pil_image.width * scale)))
    down_h = max(1, int(round(pil_image.height * scale)))
    resized = pil_image.resize((down_w, down_h), Image.Resampling.BILINEAR)
    restored = resized.resize(pil_image.size, Image.Resampling.BICUBIC)
    return np.array(restored), {"scale": round(scale, 4)}


def apply_pixelate(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    pil_image = Image.fromarray(image)
    patch = int(rng.integers(3, 8))
    down_w = max(1, pil_image.width // patch)
    down_h = max(1, pil_image.height // patch)
    small = pil_image.resize((down_w, down_h), Image.Resampling.BOX)
    restored = small.resize(pil_image.size, Image.Resampling.NEAREST)
    return np.array(restored), {"patch": patch}


def apply_brightness(
    image: np.ndarray,
    rng: np.random.Generator,
    mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    if mode == "lighter":
        factor = float(rng.uniform(1.15, 1.40))
    else:
        factor = float(rng.uniform(0.60, 0.85))
    adjusted = image.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8), {"factor": round(factor, 4)}


def apply_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    factor = float(rng.uniform(0.55, 0.85))
    adjusted = ImageEnhance.Contrast(Image.fromarray(image)).enhance(factor)
    return np.array(adjusted), {"factor": round(factor, 4)}


def parse_prob(prob_list: list[str] | None) -> dict[str, float] | None:
    if prob_list is None:
        return None

    prob_dict: dict[str, float] = {}
    for item in prob_list:
        if "=" not in item:
            raise ValueError(f"Invalid probability entry '{item}'. Use shift=value format.")
        key, value = item.split("=", 1)
        if key not in SHIFT_ORDER:
            raise ValueError(f"Unknown shift in --prob: {key}")
        probability = float(value)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability for {key} must be between 0 and 1.")
        prob_dict[key] = probability
    return prob_dict


def media_pipeline_shifts(
    rng: np.random.Generator,
    allowed_shifts: list[str],
) -> list[str]:
    # Preset for a rough social-media redistribution path.
    probabilities = {
        "compression": 0.9,
        "resize": 0.7,
        "blur": 0.5,
        "noise": 0.4,
        "lighter": 0.3,
    }
    shifts = []
    for shift in SHIFT_ORDER:
        if shift not in allowed_shifts:
            continue
        probability = probabilities.get(shift)
        if probability is not None and rng.random() < probability:
            shifts.append(shift)
    return shifts


def choose_shifts(
    assignment: str,
    selected_shifts: list[str],
    max_shifts_per_image: int,
    rng: np.random.Generator,
    mode: str = "default",
    prob_dict: dict[str, float] | None = None,
) -> list[str]:
    if mode == "media":
        media_shifts = media_pipeline_shifts(rng, selected_shifts)
        if prob_dict:
            return [shift for shift in media_shifts if rng.random() < prob_dict.get(shift, 0.5)]
        return media_shifts

    if prob_dict:
        selected = []
        for shift in selected_shifts:
            probability = prob_dict.get(shift, 0.5)
            if rng.random() < probability:
                selected.append(shift)
        return [shift for shift in SHIFT_ORDER if shift in selected]

    if assignment == "fixed":
        return list(selected_shifts)
    if assignment == "single":
        return [str(rng.choice(selected_shifts))]

    max_count = min(max_shifts_per_image, len(selected_shifts))
    count = int(rng.integers(1, max_count + 1))
    chosen = rng.choice(selected_shifts, size=count, replace=False)
    ordered = [shift for shift in SHIFT_ORDER if shift in chosen.tolist()]
    return ordered


def get_shift_functions(
    jpeg_quality: int | None = None,
) -> dict[str, Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, dict]]]:
    return {
        "dct": apply_block_dct_lowpass,
        "cosine4": apply_global_cosine4_lowpass,
        "compression": lambda image, rng: apply_jpeg_compression(
            image, rng, quality_override=jpeg_quality
        ),
        "noise": apply_gaussian_noise,
        "speckle": apply_speckle_noise,
        "blur": apply_blur,
        "resize": apply_resize_degradation,
        "pixelate": apply_pixelate,
        "lighter": lambda image, rng: apply_brightness(image, rng, "lighter"),
        "dimmer": lambda image, rng: apply_brightness(image, rng, "dimmer"),
        "contrast_down": apply_contrast,
    }


def process_image(
    image_path: Path,
    root_dir: Path,
    base_seed: int,
    assignment: str,
    selected_shifts: list[str],
    max_shifts_per_image: int,
    shift_functions: dict[str, Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, dict]]],
    mode: str,
    prob_dict: dict[str, float] | None,
) -> dict[str, str]:
    relative_path = image_path.relative_to(root_dir).as_posix()
    rng = np.random.default_rng(stable_seed(base_seed, relative_path))
    shifts = choose_shifts(
        assignment,
        selected_shifts,
        max_shifts_per_image,
        rng,
        mode=mode,
        prob_dict=prob_dict,
    )

    if not shifts:
        # Unchanged images are still included so the manifest covers the full folder.
        return {
            "relative_path": relative_path,
            "absolute_path": str(logicalize_path(image_path)),
            "applied_shifts": "none",
            "parameters": "{}",
        }

    image = np.array(Image.open(image_path).convert("RGB"))
    parameters: dict[str, dict] = {}
    for shift_name in shifts:
        # Save sampled parameters so the shifted copy can be audited later.
        image, shift_parameters = shift_functions[shift_name](image, rng)
        parameters[shift_name] = shift_parameters

    # This intentionally modifies the copied test folder in place.
    Image.fromarray(image).save(image_path)
    return {
        "relative_path": relative_path,
        "absolute_path": str(logicalize_path(image_path)),
        "applied_shifts": "|".join(shifts),
        "parameters": json.dumps(parameters, sort_keys=True),
    }


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "absolute_path", "applied_shifts", "parameters"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_label_file(
    label_path: Path,
    input_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    image_count: int,
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    selected_types = ", ".join(args.shifts)
    content = "\n".join(
        [
            "# Shift Information",
            "",
            f"- input_dir: `{logicalize_path(input_dir)}`",
            f"- manifest: `{logicalize_path(manifest_path)}`",
            f"- image_count: `{image_count}`",
            f"- assignment: `{args.assignment}`",
            f"- mode: `{args.mode}`",
            f"- shift_types: `{selected_types}`",
            f"- ratio: `{args.ratio}`",
            f"- seed: `{args.seed}`",
        ]
    )
    if args.assignment == "combo":
        content += f"\n- max_shifts_per_image: `{args.max_shifts_per_image}`"
    if args.prob:
        content += f"\n- prob: `{', '.join(args.prob)}`"
    if args.jpeg_quality is not None:
        content += f"\n- jpeg_quality: `{args.jpeg_quality}`"
    content += "\n"
    label_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    # Validate before touching images; this script overwrites files in --input-dir.
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if args.max_shifts_per_image < 1:
        raise ValueError("--max-shifts-per-image must be at least 1")
    if args.jpeg_quality is not None and not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    args.ratio = max(0.0, min(1.0, args.ratio))
    prob_dict = parse_prob(args.prob)

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = input_dir / f"{input_dir.name}_manifest.csv"
    else:
        manifest_path = manifest_path.resolve()

    label_path = args.label_file
    if label_path is None:
        label_path = input_dir / "SHIFT_INFO.md"
    else:
        label_path = label_path.resolve()

    images = list_images(input_dir)
    if not images:
        raise RuntimeError(f"No image files found under: {input_dir}")

    shift_functions = get_shift_functions(jpeg_quality=args.jpeg_quality)
    rows = []
    rng_global = np.random.default_rng(args.seed)
    for image_path in images:
        if rng_global.random() > args.ratio:
            # Ratio < 1 creates mixed folders with some clean images left in place.
            rows.append(
                {
                    "relative_path": image_path.relative_to(input_dir).as_posix(),
                    "absolute_path": str(logicalize_path(image_path)),
                    "applied_shifts": "none",
                    "parameters": "{}",
                }
            )
            continue
        rows.append(
            process_image(
                image_path=image_path,
                root_dir=input_dir,
                base_seed=args.seed,
                assignment=args.assignment,
                selected_shifts=args.shifts,
                max_shifts_per_image=args.max_shifts_per_image,
                shift_functions=shift_functions,
                mode=args.mode,
                prob_dict=prob_dict,
            )
        )

    write_manifest(manifest_path, rows)
    write_label_file(label_path, input_dir, manifest_path, args, len(rows))
    print(f"Processed {len(rows)} images in place under: {logicalize_path(input_dir)}")
    print(f"Manifest written to: {logicalize_path(manifest_path)}")
    print(f"Label file written to: {logicalize_path(label_path)}")


if __name__ == "__main__":
    main()
