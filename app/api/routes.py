"""
ANPR API routes.

This module contains ONLY route definitions and request/response
orchestration.  All business logic lives in the service layer.

Endpoints:
    GET  /health          – Liveness + dependency health check.
    POST /detect/image    – Single-image ANPR pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.constants import (
    ALLOWED_IMAGE_EXTENSIONS,
    STATUS_ERROR,
    STATUS_LOW_CONFIDENCE,
    STATUS_NO_PLATE,
    STATUS_NO_VEHICLE,
    STATUS_SUCCESS,
)
from app.dependencies.providers import (
    get_image_preprocessor,
    get_ocr_service,
    get_plate_detector,
    get_plate_repository,
    get_storage_service,
    get_tracking_service,
    get_validator_service,
    get_vehicle_detector,
)
from app.models.plate_models import PlateImageDocument, PlateRecordDocument
from app.repositories.plate_repository import PlateRepository
from app.schemas.response import (
    BoundingBox,
    DetectionResponse,
    HealthResponse,
    OCRResult,
    PlateDetectionResult,
    StorageResult,
    VehicleDetectionResult,
)
from app.services.detector import VehicleDetectorService
from app.services.image_preprocessor import ImagePreprocessorService
from app.services.ocr_service import OCRService
from app.services.plate_detector import PlateDetectorService
from app.services.storage_service import StorageService
from app.services.tracking_service import TrackingService
from app.services.validator import ValidatorService
from app.utils.image_utils import crop_region, read_image_from_bytes
from app.utils.timer import Timer

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns application liveness and MongoDB connectivity status.",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """Lightweight liveness probe."""
    from app.database.mongodb import get_database
    mongo_status = "unavailable"
    try:
        db = get_database()
        await db.command("ping")
        mongo_status = "connected"
    except Exception:
        mongo_status = "disconnected"

    return HealthResponse(
        status="ok",
        app_version=settings.APP_VERSION,
        mongodb=mongo_status,
        timestamp=datetime.now(tz=timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image Detection
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/detect/image",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="ANPR – Single Image",
    description=(
        "Upload a vehicle image and receive the detected license plate number, "
        "OCR confidence, image path, and MongoDB record IDs."
    ),
    tags=["Detection"],
)
async def detect_image(
    file: UploadFile = File(..., description="Vehicle image (JPEG, PNG, BMP, WEBP, TIFF)"),
    camera_id: Optional[str] = Form(default=None, description="Optional CCTV camera identifier"),
    # ── Injected services ──────────────────────────────────────────────
    vehicle_detector: VehicleDetectorService = Depends(get_vehicle_detector),
    plate_detector: PlateDetectorService = Depends(get_plate_detector),
    preprocessor: ImagePreprocessorService = Depends(get_image_preprocessor),
    ocr_service: OCRService = Depends(get_ocr_service),
    validator: ValidatorService = Depends(get_validator_service),
    storage: StorageService = Depends(get_storage_service),
    tracker: TrackingService = Depends(get_tracking_service),
    repository: PlateRepository = Depends(get_plate_repository),
) -> DetectionResponse:
    """Full ANPR pipeline for a single vehicle image."""

    timer = Timer()
    timer.start()

    # ── 0. Validate service/model availability ─────────────────────────
    if not vehicle_detector.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vehicle detection model missing",
        )
    if not plate_detector.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="License plate detection model missing",
        )
    if not ocr_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR engine unavailable",
        )

    # ── 1. Validate upload ─────────────────────────────────────────────
    _validate_upload(file)
    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    # ── 2. Decode image ────────────────────────────────────────────────
    try:
        image = read_image_from_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # ── 3. Vehicle detection ───────────────────────────────────────────
    vehicle_result = vehicle_detector.detect(image)

    vehicle_response = VehicleDetectionResult(
        detected=vehicle_result.detected,
        vehicle_type=vehicle_result.vehicle_type,
        confidence=vehicle_result.confidence,
        bounding_box=BoundingBox(
            x1=vehicle_result.x1 or 0,
            y1=vehicle_result.y1 or 0,
            x2=vehicle_result.x2 or 0,
            y2=vehicle_result.y2 or 0,
        ) if vehicle_result.has_region() else None,
    )

    if not vehicle_result.detected:
        elapsed = timer.stop()
        return _build_response(
            status=STATUS_NO_VEHICLE,
            vehicle=vehicle_response,
            plate_detection=PlateDetectionResult(detected=False),
            ocr=OCRResult(),
            storage=StorageResult(),
            camera_id=camera_id,
            elapsed_ms=elapsed,
        )

    # ── 4. Crop vehicle region ─────────────────────────────────────────
    if vehicle_result.has_region():
        vehicle_crop = crop_region(
            image,
            vehicle_result.x1, vehicle_result.y1,  # type: ignore[arg-type]
            vehicle_result.x2, vehicle_result.y2,  # type: ignore[arg-type]
        )
    else:
        vehicle_crop = image

    # ── 5. License plate detection ─────────────────────────────────────
    plate_result = plate_detector.detect(vehicle_crop)

    plate_response = PlateDetectionResult(
        detected=plate_result.detected,
        confidence=plate_result.confidence,
        bounding_box=BoundingBox(
            x1=plate_result.x1 or 0,
            y1=plate_result.y1 or 0,
            x2=plate_result.x2 or 0,
            y2=plate_result.y2 or 0,
        ) if plate_result.has_region() else None,
    )

    if not plate_result.detected:
        elapsed = timer.stop()
        return _build_response(
            status=STATUS_NO_PLATE,
            vehicle=vehicle_response,
            plate_detection=plate_response,
            ocr=OCRResult(),
            storage=StorageResult(),
            camera_id=camera_id,
            elapsed_ms=elapsed,
        )

    # ── 6. Crop plate region ───────────────────────────────────────────
    plate_crop = crop_region(
        vehicle_crop,
        plate_result.x1, plate_result.y1,  # type: ignore[arg-type]
        plate_result.x2, plate_result.y2,  # type: ignore[arg-type]
    )

    # ── 7. Image enhancement ───────────────────────────────────────────
    enhanced_plate = preprocessor.preprocess(plate_crop)

    # ── 8. OCR ────────────────────────────────────────────────────────
    ocr_result = ocr_service.run(enhanced_plate)

    ocr_response = OCRResult(
        raw_text=ocr_result.text,
        confidence=ocr_result.confidence,
    )

    if ocr_result.is_empty():
        elapsed = timer.stop()
        return _build_response(
            status=STATUS_NO_PLATE,
            vehicle=vehicle_response,
            plate_detection=plate_response,
            ocr=ocr_response,
            storage=StorageResult(),
            camera_id=camera_id,
            elapsed_ms=elapsed,
        )

    # ── 9. Validate & correct ──────────────────────────────────────────
    validation = validator.validate(ocr_result.text, ocr_result.confidence)

    if not validation.is_valid:
        elapsed = timer.stop()
        return _build_response(
            status=validation.status,
            vehicle=vehicle_response,
            plate_detection=plate_response,
            ocr=ocr_response,
            storage=StorageResult(),
            camera_id=camera_id,
            elapsed_ms=elapsed,
            plate_number=validation.plate_number,
        )

    # ── 10. Tracking (Phase 1: always saves) ──────────────────────────
    tracking = tracker.process_detection(
        frame_id=0,
        plate_number=validation.plate_number,
    )

    # ── 11. Persist ────────────────────────────────────────────────────
    image_path = storage.save_plate_image(plate_crop)

    elapsed = timer.stop()

    plate_image_id: Optional[str] = None
    plate_record_id: Optional[str] = None

    if tracking.should_save:
        image_doc = PlateImageDocument(
            plate_number=validation.plate_number,  # type: ignore[arg-type]
            image_path=image_path,
            vehicle_type=vehicle_result.vehicle_type,
            confidence=validation.confidence,  # type: ignore[arg-type]
            camera_id=camera_id,
        )
        plate_image_id = await repository.insert_plate_image(image_doc)

        record_doc = PlateRecordDocument(
            plate_number=validation.plate_number,  # type: ignore[arg-type]
            confidence=validation.confidence,  # type: ignore[arg-type]
            camera_id=camera_id,
            raw_ocr_text=ocr_result.text,
            vehicle_type=vehicle_result.vehicle_type,
            processing_time_ms=elapsed,
        )
        plate_record_id = await repository.insert_plate_record(record_doc)

    logger.info(
        "Detection complete | plate=%s conf=%.3f time=%.1fms camera=%s",
        validation.plate_number, validation.confidence, elapsed, camera_id,
    )

    return _build_response(
        status=STATUS_SUCCESS,
        vehicle=vehicle_response,
        plate_detection=plate_response,
        ocr=ocr_response,
        storage=StorageResult(
            plate_image_id=plate_image_id,
            plate_record_id=plate_record_id,
            image_path=image_path,
        ),
        camera_id=camera_id,
        elapsed_ms=elapsed,
        plate_number=validation.plate_number,
        confidence=validation.confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    """Raise HTTP 400 if the uploaded file has an unsupported extension."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
            ),
        )


def _build_response(
    *,
    status: str,
    vehicle: VehicleDetectionResult,
    plate_detection: PlateDetectionResult,
    ocr: OCRResult,
    storage: StorageResult,
    elapsed_ms: float,
    camera_id: Optional[str] = None,
    plate_number: Optional[str] = None,
    confidence: Optional[float] = None,
) -> DetectionResponse:
    return DetectionResponse(
        status=status,
        plate_number=plate_number,
        confidence=confidence,
        vehicle=vehicle,
        plate_detection=plate_detection,
        ocr=ocr,
        storage=storage,
        camera_id=camera_id,
        processing_time_ms=round(elapsed_ms, 2),
        timestamp=datetime.now(tz=timezone.utc),
    )
