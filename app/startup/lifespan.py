"""
Application startup and shutdown lifecycle handlers.

FastAPI's ``lifespan`` context manager is used to:
1. Configure logging.
2. Ensure runtime directories exist.
3. Connect to MongoDB and verify the connection.
4. Bootstrap database indexes.
5. Load ML models (YOLO + PaddleOCR) into ``app.state``.
6. Gracefully shut down connections when the server stops.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging
from app.database.collections import ensure_indexes
from app.database.mongodb import close_db, connect_db, get_database
from app.services.detector import VehicleDetectorService
from app.services.image_preprocessor import ImagePreprocessorService
from app.services.ocr_service import OCRService
from app.services.plate_detector import PlateDetectorService
from app.services.storage_service import StorageService
from app.services.tracking_service import TrackingService
from app.services.validator import ValidatorService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager.

    Everything before ``yield`` runs at startup; everything after runs at
    shutdown.
    """
    # ── Startup ─────────────────────────────────────────────────────────
    setup_logging()
    logger.info("Starting %s v%s [%s]", settings.APP_NAME, settings.APP_VERSION, settings.APP_ENV)

    # Create runtime directories
    settings.ensure_directories()

    # MongoDB
    await connect_db()
    await ensure_indexes(get_database())

    # Initialise services (load models into memory)
    vehicle_detector = VehicleDetectorService()
    vehicle_detector.load()

    plate_detector = PlateDetectorService()
    plate_detector.load()

    ocr_service = OCRService()
    ocr_service.load()

    image_preprocessor = ImagePreprocessorService()
    validator_service = ValidatorService()
    storage_service = StorageService()
    storage_service.ensure_dirs()
    tracking_service = TrackingService()

    # Attach services to app.state for dependency injection
    app.state.vehicle_detector = vehicle_detector
    app.state.plate_detector = plate_detector
    app.state.ocr_service = ocr_service
    app.state.image_preprocessor = image_preprocessor
    app.state.validator_service = validator_service
    app.state.storage_service = storage_service
    app.state.tracking_service = tracking_service

    logger.info("All services initialised. Server is ready.")

    yield  # ← Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down %s …", settings.APP_NAME)
    await close_db()
    logger.info("Shutdown complete.")
