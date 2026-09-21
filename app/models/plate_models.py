"""
MongoDB document models (ODM layer).

These Pydantic models represent the exact shape of documents stored in
MongoDB, including the ``_id`` field mapped to ``id``.  They are used
by repositories when serialising/deserialising documents.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# plate_images collection
# ─────────────────────────────────────────────────────────────────────────────

class PlateImageDocument(BaseModel):
    """Represents a document in the ``plate_images`` collection.

    Stores the cropped plate image path alongside the OCR result so that
    operators can visually verify detections.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), alias="_id")
    plate_number: str
    image_path: str
    vehicle_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=_utcnow)
    confidence: float
    camera_id: Optional[str] = None

    model_config = {"populate_by_name": True}

    def to_mongo(self) -> dict:
        """Serialise to a plain dict suitable for Motor insert."""
        data = self.model_dump(by_alias=True)
        # Ensure _id is stored as a string (UUID)
        return data


# ─────────────────────────────────────────────────────────────────────────────
# plate_records collection
# ─────────────────────────────────────────────────────────────────────────────

class PlateRecordDocument(BaseModel):
    """Represents a document in the ``plate_records`` collection.

    Lightweight event log – one record per detection event, without image
    payload, designed for fast querying and analytics.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), alias="_id")
    plate_number: str
    timestamp: datetime = Field(default_factory=_utcnow)
    confidence: float
    camera_id: Optional[str] = None
    raw_ocr_text: Optional[str] = None
    vehicle_type: Optional[str] = None
    processing_time_ms: Optional[float] = None

    model_config = {"populate_by_name": True}

    def to_mongo(self) -> dict:
        """Serialise to a plain dict suitable for Motor insert."""
        return self.model_dump(by_alias=True)
