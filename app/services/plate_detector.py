"""
License plate detection service.

Uses a dedicated YOLO model (license_plate_detector.pt) to detect plate
regions within an already-cropped vehicle image.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings
from app.core.constants import MIN_PLATE_HEIGHT, MIN_PLATE_WIDTH

logger = logging.getLogger(__name__)


@dataclass
class PlateDetection:
    """Result of a single license plate detection."""

    detected: bool
    confidence: Optional[float] = None
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None

    def has_region(self) -> bool:
        return all(v is not None for v in [self.x1, self.y1, self.x2, self.y2])

    def width(self) -> int:
        return (self.x2 or 0) - (self.x1 or 0)

    def height(self) -> int:
        return (self.y2 or 0) - (self.y1 or 0)


class PlateDetectorService:
    """Loads a dedicated plate-detection YOLO model and runs inference."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Lazy-load the plate detector model. Called once at startup."""
        if self._loaded:
            return

        model_path = Path(settings.PLATE_MODEL_PATH)
        if not model_path.exists():
            logger.warning(
                "Plate detector model not found at '%s'. "
                "Will fall back to full-image OCR. Place the model and restart.",
                model_path,
            )
            self._loaded = True
            return

        try:
            from ultralytics import YOLO  # type: ignore[import]
            self._model = YOLO(str(model_path))
            logger.info("Plate detector model loaded from '%s'", model_path)
        except Exception as exc:
            logger.error("Failed to load plate detector model: %s", exc)
            raise
        self._loaded = True

    def is_available(self) -> bool:
        """Return True if license plate model file exists and is loaded."""
        return Path(settings.PLATE_MODEL_PATH).exists() and self._model is not None

    def detect(self, vehicle_crop: np.ndarray) -> PlateDetection:
        """Detect the license plate region within a vehicle crop.

        Falls back to treating the full crop as the plate region when the
        model is unavailable.
        """
        if self._model is None:
            logger.debug("Plate model unavailable – treating full crop as plate region.")
            h, w = vehicle_crop.shape[:2]
            return PlateDetection(
                detected=True,
                confidence=1.0,
                x1=0, y1=0, x2=w, y2=h,
            )

        results = self._model.predict(
            source=vehicle_crop,
            conf=settings.PLATE_CONF_THRESHOLD,
            iou=settings.PLATE_IOU_THRESHOLD,
            verbose=False,
        )

        best: Optional[PlateDetection] = None
        best_conf: float = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0].item())
                if conf > best_conf:
                    xyxy = box.xyxy[0].tolist()
                    candidate = PlateDetection(
                        detected=True,
                        confidence=round(conf, 4),
                        x1=int(xyxy[0]),
                        y1=int(xyxy[1]),
                        x2=int(xyxy[2]),
                        y2=int(xyxy[3]),
                    )
                    # Discard implausibly tiny detections
                    if (
                        candidate.width() >= MIN_PLATE_WIDTH
                        and candidate.height() >= MIN_PLATE_HEIGHT
                    ):
                        best_conf = conf
                        best = candidate

        if best is None:
            logger.debug("No license plate detected in vehicle crop.")
            return PlateDetection(detected=False)

        logger.debug(
            "Plate detected: conf=%.3f bbox=(%d,%d,%d,%d)",
            best.confidence, best.x1, best.y1, best.x2, best.y2,
        )
        return best
