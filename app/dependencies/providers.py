"""
FastAPI dependency providers.

FastAPI's ``Depends()`` system is used to inject pre-initialised
service and repository instances into route handlers.  All singleton
service instances live in the application lifespan state (``app.state``)
to avoid re-creating models on every request.
"""
from __future__ import annotations

from fastapi import Depends, Request

from app.database.mongodb import get_database
from app.repositories.plate_repository import PlateRepository
from app.services.detector import VehicleDetectorService
from app.services.image_preprocessor import ImagePreprocessorService
from app.services.ocr_service import OCRService
from app.services.plate_detector import PlateDetectorService
from app.services.storage_service import StorageService
from app.services.tracking_service import TrackingService
from app.services.validator import ValidatorService


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

def get_plate_repository() -> PlateRepository:
    """Provide a ``PlateRepository`` bound to the active Motor database."""
    db = get_database()
    return PlateRepository(db)


# ─────────────────────────────────────────────────────────────────────────────
# Services (singletons stored on app.state by startup handler)
# ─────────────────────────────────────────────────────────────────────────────

def get_vehicle_detector(request: Request) -> VehicleDetectorService:
    return request.app.state.vehicle_detector


def get_plate_detector(request: Request) -> PlateDetectorService:
    return request.app.state.plate_detector


def get_image_preprocessor(request: Request) -> ImagePreprocessorService:
    return request.app.state.image_preprocessor


def get_ocr_service(request: Request) -> OCRService:
    return request.app.state.ocr_service


def get_validator_service(request: Request) -> ValidatorService:
    return request.app.state.validator_service


def get_storage_service(request: Request) -> StorageService:
    return request.app.state.storage_service


def get_tracking_service(request: Request) -> TrackingService:
    return request.app.state.tracking_service
