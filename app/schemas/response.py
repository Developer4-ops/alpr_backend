"""
API response schemas.

These Pydantic models define the exact JSON structure returned by every
API endpoint.  Keeping them separate from MongoDB document models allows
the API contract to evolve independently of the storage layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.constants import STATUS_SUCCESS


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class VehicleDetectionResult(BaseModel):
    detected: bool
    vehicle_type: Optional[str] = None
    confidence: Optional[float] = None
    bounding_box: Optional[BoundingBox] = None


class PlateDetectionResult(BaseModel):
    detected: bool
    confidence: Optional[float] = None
    bounding_box: Optional[BoundingBox] = None


class OCRResult(BaseModel):
    raw_text: Optional[str] = None
    confidence: Optional[float] = None


class StorageResult(BaseModel):
    plate_image_id: Optional[str] = None
    plate_record_id: Optional[str] = None
    image_path: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Top-level response
# ─────────────────────────────────────────────────────────────────────────────

class DetectionResponse(BaseModel):
    """Unified response for ``POST /detect/image``."""

    status: str = Field(default=STATUS_SUCCESS, description="Pipeline result status code")
    plate_number: Optional[str] = Field(
        default=None, description="Validated and corrected plate number"
    )
    confidence: Optional[float] = Field(
        default=None, description="OCR confidence (0–1)"
    )
    vehicle: VehicleDetectionResult
    plate_detection: PlateDetectionResult
    ocr: OCRResult
    storage: StorageResult
    camera_id: Optional[str] = None
    processing_time_ms: float = Field(description="Total pipeline wall-clock time in ms")
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Generic error envelope."""

    status: str = "ERROR"
    detail: str
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    app_version: str
    mongodb: str
    timestamp: datetime
