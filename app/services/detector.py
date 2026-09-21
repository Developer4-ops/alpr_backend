"""
Vehicle detection service.

Wraps Ultralytics YOLO to detect vehicles in a scene image.
Only returns the detection with the highest confidence among valid
vehicle classes (car, motorcycle, bus, truck).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings
from app.core.constants import VEHICLE_CLASS_IDS, VEHICLE_CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class VehicleDetection:
    """Result of a single vehicle detection."""

    detected: bool
    vehicle_type: Optional[str] = None
    confidence: Optional[float] = None
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None

    def has_region(self) -> bool:
        return all(v is not None for v in [self.x1, self.y1, self.x2, self.y2])


class VehicleDetectorService:
    """Loads a YOLO model and runs vehicle detection on BGR images."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Lazy-load the YOLO model. Called once at application startup."""
        if self._loaded:
            return

        model_path = Path(settings.VEHICLE_MODEL_PATH)
        if not model_path.exists():
            logger.warning(
                "Vehicle model not found at '%s'. "
                "Detection will be skipped. Place the model file and restart.",
                model_path,
            )
            self._loaded = True  # Mark as loaded to avoid repeated warnings
            return

        try:
            from ultralytics import YOLO  # type: ignore[import]
            self._model = YOLO(str(model_path))
            logger.info("Vehicle detection model loaded from '%s'", model_path)
        except Exception as exc:
            logger.error("Failed to load vehicle model: %s", exc)
            raise
        self._loaded = True

    def is_available(self) -> bool:
        """Return True if vehicle detection model file exists and is loaded."""
        return Path(settings.VEHICLE_MODEL_PATH).exists() and self._model is not None

    def detect(self, image: np.ndarray) -> VehicleDetection:
        """Run vehicle detection and return the best detection.

        If the model is unavailable, returns a dummy detection indicating
        the whole image is the vehicle region (graceful degradation so
        plate detection can still be attempted).
        """
        if self._model is None:
            logger.debug(
                "Vehicle model not available – treating full image as vehicle region."
            )
            h, w = image.shape[:2]
            return VehicleDetection(
                detected=True,
                vehicle_type="unknown",
                confidence=1.0,
                x1=0, y1=0, x2=w, y2=h,
            )

        results = self._model.predict(
            source=image,
            conf=settings.VEHICLE_CONF_THRESHOLD,
            iou=settings.VEHICLE_IOU_THRESHOLD,
            verbose=False,
        )

        best: Optional[VehicleDetection] = None
        best_conf: float = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue
                conf = float(box.conf[0].item())
                if conf > best_conf:
                    best_conf = conf
                    xyxy = box.xyxy[0].tolist()
                    best = VehicleDetection(
                        detected=True,
                        vehicle_type=VEHICLE_CLASS_NAMES[cls_id],
                        confidence=round(conf, 4),
                        x1=int(xyxy[0]),
                        y1=int(xyxy[1]),
                        x2=int(xyxy[2]),
                        y2=int(xyxy[3]),
                    )

        if best is None:
            logger.debug("No vehicle detected in image.")
            return VehicleDetection(detected=False)

        logger.debug(
            "Vehicle detected: type=%s conf=%.3f bbox=(%d,%d,%d,%d)",
            best.vehicle_type, best.confidence,
            best.x1, best.y1, best.x2, best.y2,
        )
        return best
