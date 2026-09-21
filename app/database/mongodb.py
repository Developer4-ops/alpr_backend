"""
MongoDB connection management using Motor (async driver).

Provides:
- ``connect_db()``  – open the connection pool (called at startup)
- ``close_db()``    – gracefully close connections (called at shutdown)
- ``get_database()``– returns the active ``AsyncIOMotorDatabase`` instance
"""
from __future__ import annotations

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import errors as pymongo_errors

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level state (intentionally not a singleton class to avoid
# coupling callers to an instance – they use ``get_database()`` instead)
_client: Optional[AsyncIOMotorClient] = None  # type: ignore[type-arg]
_database: Optional[AsyncIOMotorDatabase] = None  # type: ignore[type-arg]


async def connect_db() -> None:
    """Initialise the Motor client and verify connectivity.

    Raises ``ConnectionFailure`` if MongoDB is unreachable so the
    application fails loudly at startup rather than at first request.
    """
    global _client, _database

    logger.info("Connecting to MongoDB at %s …", settings.MONGO_URI)

    _client = AsyncIOMotorClient(
        settings.MONGO_URI,
        maxPoolSize=settings.MONGO_MAX_POOL_SIZE,
        minPoolSize=settings.MONGO_MIN_POOL_SIZE,
        serverSelectionTimeoutMS=settings.MONGO_SERVER_SELECTION_TIMEOUT_MS,
    )

    # Ping to validate the connection before the app starts serving traffic
    try:
        await _client.admin.command("ping")
    except pymongo_errors.ConnectionFailure as exc:
        logger.critical("MongoDB connection failed: %s", exc)
        raise

    _database = _client[settings.MONGO_DB_NAME]
    logger.info("MongoDB connected – database: '%s'", settings.MONGO_DB_NAME)


async def close_db() -> None:
    """Close the Motor client and release all connection pool resources."""
    global _client, _database
    if _client is not None:
        _client.close()
        _client = None
        _database = None
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:  # type: ignore[type-arg]
    """Return the active database instance.

    Raises ``RuntimeError`` if called before ``connect_db()``.
    """
    if _database is None:
        raise RuntimeError(
            "Database is not initialised. "
            "Ensure ``connect_db()`` was awaited at application startup."
        )
    return _database
