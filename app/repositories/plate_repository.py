"""
Plate repository.

Encapsulates all MongoDB read/write operations for the
``plate_images`` and ``plate_records`` collections.

Services must NEVER import Motor or call MongoDB directly —
all persistence goes through this repository.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import errors as pymongo_errors

from app.database.collections import PLATE_IMAGES_COLLECTION, PLATE_RECORDS_COLLECTION
from app.models.plate_models import PlateImageDocument, PlateRecordDocument

logger = logging.getLogger(__name__)


class PlateRepository:
    """Data-access layer for plate detection results."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:  # type: ignore[type-arg]
        self._db = db
        self._images_col = db[PLATE_IMAGES_COLLECTION]
        self._records_col = db[PLATE_RECORDS_COLLECTION]

    # ------------------------------------------------------------------ #
    # plate_images
    # ------------------------------------------------------------------ #

    async def insert_plate_image(self, document: PlateImageDocument) -> str:
        """Insert a plate image document and return its ``_id``."""
        try:
            result = await self._images_col.insert_one(document.to_mongo())
            inserted_id = str(result.inserted_id)
            logger.debug("Inserted plate_image: _id=%s", inserted_id)
            return inserted_id
        except pymongo_errors.PyMongoError as exc:
            logger.error("Failed to insert plate_image: %s", exc)
            raise

    async def find_plate_image_by_id(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single plate image document by ``_id``."""
        return await self._images_col.find_one({"_id": doc_id})

    async def find_images_by_plate_number(
        self, plate_number: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return the *limit* most-recent images for a given plate."""
        cursor = (
            self._images_col.find({"plate_number": plate_number})
            .sort("timestamp", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------------------ #
    # plate_records
    # ------------------------------------------------------------------ #

    async def insert_plate_record(self, document: PlateRecordDocument) -> str:
        """Insert a plate record document and return its ``_id``."""
        try:
            result = await self._records_col.insert_one(document.to_mongo())
            inserted_id = str(result.inserted_id)
            logger.debug("Inserted plate_record: _id=%s", inserted_id)
            return inserted_id
        except pymongo_errors.PyMongoError as exc:
            logger.error("Failed to insert plate_record: %s", exc)
            raise

    async def find_record_by_id(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single plate record by ``_id``."""
        return await self._records_col.find_one({"_id": doc_id})

    async def find_records_by_plate_number(
        self, plate_number: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return the *limit* most-recent records for a given plate."""
        cursor = (
            self._records_col.find({"plate_number": plate_number})
            .sort("timestamp", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def find_recent_records(
        self, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return the most-recent *limit* detection records."""
        cursor = self._records_col.find({}).sort("timestamp", -1).limit(limit)
        return await cursor.to_list(length=limit)
