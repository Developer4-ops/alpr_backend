"""
API request schemas.

These Pydantic models define the validated shape of incoming data.
FastAPI injects them automatically from multipart form data or JSON.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ImageDetectionRequest(BaseModel):
    """Query parameters / metadata that may accompany an image upload.

    The actual image bytes are received as an ``UploadFile`` in the route
    and are NOT part of this schema.
    """

    camera_id: Optional[str] = Field(
        default=None,
        description="Identifier of the CCTV camera that captured the image.",
        max_length=64,
    )
