"""
File storage service.

Responsible solely for persisting cropped plate images to the local
filesystem under ``uploads/plates/``.  Filenames are UUID-based to
avoid collisions and exposure of internal data in URLs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.core.config import settings
from app.utils.image_utils import save_image

logger = logging.getLogger(__name__)


class StorageService:
    """Manages filesystem persistence of detected plate images."""

    def __init__(self) -> None:
        self._plates_dir = Path(settings.PLATES_DIR)
        self._vehicles_dir = Path(settings.VEHICLES_DIR)

    def ensure_dirs(self) -> None:
        """Create upload directories if they do not exist."""
        self._plates_dir.mkdir(parents=True, exist_ok=True)
        self._vehicles_dir.mkdir(parents=True, exist_ok=True)

    def save_plate_image(self, plate_crop: np.ndarray) -> str:
        """Save a cropped plate image and return its relative file path.

        The returned path is relative to the project root and suitable
        for storage in MongoDB and returning via the API.

        Args:
            plate_crop: BGR NumPy array of the plate region.

        Returns:
            Relative path string, e.g. ``"uploads/plates/<uuid>.jpg"``.

        Raises:
            IOError: If the file cannot be written.
        """
        self.ensure_dirs()
        filename = f"{uuid4().hex}.jpg"
        full_path = self._plates_dir / filename
        save_image(plate_crop, full_path, quality=95)
        relative_path = str(full_path)
        logger.debug("Plate image saved: %s", relative_path)
        return relative_path

    def save_vehicle_image(self, vehicle_crop: np.ndarray) -> str:
        """Save a cropped vehicle image and return its relative file path."""
        self.ensure_dirs()
        filename = f"{uuid4().hex}.jpg"
        full_path = self._vehicles_dir / filename
        save_image(vehicle_crop, full_path, quality=90)
        relative_path = str(full_path)
        logger.debug("Vehicle image saved: %s", relative_path)
        return relative_path
