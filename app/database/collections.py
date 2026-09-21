"""
MongoDB collection names and index definitions.

Centralising these here means routes, services, and repositories never
hard-code collection names.  Index bootstrapping is called once at startup.
"""
from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Collection name constants
# ─────────────────────────────────────────────────────────────────────────────
PLATE_IMAGES_COLLECTION: str = "plate_images"
PLATE_RECORDS_COLLECTION: str = "plate_records"


# ─────────────────────────────────────────────────────────────────────────────
# Index definitions
# ─────────────────────────────────────────────────────────────────────────────

_PLATE_IMAGES_INDEXES: list[IndexModel] = [
    IndexModel([("plate_number", ASCENDING)], name="idx_plate_images_plate_number"),
    IndexModel([("timestamp", DESCENDING)], name="idx_plate_images_timestamp"),
    IndexModel([("camera_id", ASCENDING)], name="idx_plate_images_camera_id"),
]

_PLATE_RECORDS_INDEXES: list[IndexModel] = [
    IndexModel([("plate_number", ASCENDING)], name="idx_plate_records_plate_number"),
    IndexModel([("timestamp", DESCENDING)], name="idx_plate_records_timestamp"),
    IndexModel(
        [("plate_number", ASCENDING), ("timestamp", DESCENDING)],
        name="idx_plate_records_plate_ts",
    ),
    IndexModel([("camera_id", ASCENDING)], name="idx_plate_records_camera_id"),
]


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:  # type: ignore[type-arg]
    """Create all required indexes idempotently.

    Motor's ``create_indexes`` is a no-op if the index already exists
    with the same specification, so this is safe to call on every startup.
    """
    logger.info("Ensuring MongoDB indexes …")

    await db[PLATE_IMAGES_COLLECTION].create_indexes(_PLATE_IMAGES_INDEXES)
    logger.debug("Indexes ensured for collection '%s'", PLATE_IMAGES_COLLECTION)

    await db[PLATE_RECORDS_COLLECTION].create_indexes(_PLATE_RECORDS_INDEXES)
    logger.debug("Indexes ensured for collection '%s'", PLATE_RECORDS_COLLECTION)

    logger.info("MongoDB indexes verified.")
