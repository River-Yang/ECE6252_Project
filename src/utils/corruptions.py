from __future__ import annotations

from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _center_crop_resize(image: Image.Image, ratio: float, size: int = 224) -> Image.Image:
    width, height = image.size
    crop_w = int(width * ratio)
    crop_h = int(height * ratio)
    left = max((width - crop_w) // 2, 0)
    top = max((height - crop_h) // 2, 0)
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((size, size))


def _resize_reencode(image: Image.Image, size: int = 224) -> Image.Image:
    resized = image.resize((128, 128)).resize((size, size))
    return _jpeg(resized, quality=75)


def _array_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    base = np.asarray(image).astype(np.float32)
    noise = np.random.normal(0.0, sigma, size=base.shape)
    return _array_to_pil(base + noise)


def _speckle_noise(image: Image.Image, sigma: float) -> Image.Image:
    base = np.asarray(image).astype(np.float32)
    noise = np.random.normal(0.0, sigma, size=base.shape)
    return _array_to_pil(base + base * noise)


def _pixelate(image: Image.Image, block_size: int) -> Image.Image:
    width, height = image.size
    small = image.resize((max(1, width // block_size), max(1, height // block_size)))
    return small.resize((width, height), Image.Resampling.NEAREST)


def get_corruption(condition: str, image_size: int = 224) -> Callable[[Image.Image], Image.Image]:
    # Keep this mapping aligned with configs/default.yaml so every reported
    # shift condition has exactly one implementation here.
    if condition == "clean":
        return lambda image: image
    if condition == "jpeg_q95":
        return lambda image: _jpeg(image, 95)
    if condition == "jpeg_q75":
        return lambda image: _jpeg(image, 75)
    if condition == "jpeg_q50":
        return lambda image: _jpeg(image, 50)
    if condition == "jpeg_q30":
        return lambda image: _jpeg(image, 30)
    if condition == "blur_s1":
        return lambda image: image.filter(ImageFilter.GaussianBlur(radius=1))
    if condition == "blur_s2":
        return lambda image: image.filter(ImageFilter.GaussianBlur(radius=2))
    if condition == "gaussian_noise_s8":
        return lambda image: _gaussian_noise(image, 8.0)
    if condition == "speckle_s005":
        return lambda image: _speckle_noise(image, 0.05)
    if condition == "pixelate_b8":
        return lambda image: _pixelate(image, 8)
    if condition == "brightness_07":
        return lambda image: ImageEnhance.Brightness(image).enhance(0.7)
    if condition == "brightness_13":
        return lambda image: ImageEnhance.Brightness(image).enhance(1.3)
    if condition == "contrast_07":
        return lambda image: ImageEnhance.Contrast(image).enhance(0.7)
    if condition == "crop_90":
        return lambda image: _center_crop_resize(image, 0.9, size=image_size)
    if condition == "crop_80":
        return lambda image: _center_crop_resize(image, 0.8, size=image_size)
    if condition == "resize_reencode":
        return lambda image: _resize_reencode(image, size=image_size)
    raise ValueError(f"Unknown corruption condition: {condition}")
