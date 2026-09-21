"""
Low-level image utility functions.

All functions are pure (no side effects) and operate on NumPy arrays or
file paths, making them easy to unit-test in isolation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def read_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR NumPy array.

    Raises ``ValueError`` if the bytes cannot be decoded as an image.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes. Ensure the file is a valid image.")
    return img


def read_image_from_path(path: str | Path) -> np.ndarray:
    """Load an image from disk into a BGR NumPy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path, quality: int = 95) -> None:
    """Save a BGR NumPy array to disk.

    Supports JPEG, PNG, BMP, and TIFF based on file extension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success = cv2.imwrite(str(path), img, params)
    if not success:
        raise IOError(f"Failed to write image to {path}")


def crop_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Return a cropped sub-image, clamped to image bounds."""
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop region: ({x1},{y1})-({x2},{y2}) on image size ({w},{h})"
        )
    return img[y1:y2, x1:x2]


def resize_image(
    img: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    inter: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """Resize while preserving aspect ratio if only one dimension is given."""
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        assert height is not None
        scale = height / h
        new_w = int(w * scale)
        new_h = height
    elif height is None:
        scale = width / w
        new_w = width
        new_h = int(h * scale)
    else:
        new_w, new_h = width, height
    return cv2.resize(img, (new_w, new_h), interpolation=inter)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale. Idempotent for already-gray images."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert grayscale to BGR. Idempotent for already-colour images."""
    if len(img.shape) == 3 and img.shape[2] == 3:
        return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def ndarray_to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR NumPy array to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_ndarray(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to BGR NumPy array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_image_dimensions(img: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) of image."""
    h, w = img.shape[:2]
    return w, h
