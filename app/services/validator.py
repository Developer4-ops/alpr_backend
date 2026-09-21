"""
License plate text validation and correction service.

Orchestrates the regex validation and position-aware OCR correction
utilities.  Returns a structured result that the pipeline uses to
decide whether to persist the detection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.core.config import settings
from app.core.constants import (
    STATUS_INVALID_PLATE,
    STATUS_LOW_CONFIDENCE,
    STATUS_SUCCESS,
)
from app.utils.correction_utils import correct_ocr_text
from app.utils.regex_utils import is_valid_indian_plate, normalise_plate_number

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of the validate-and-correct step."""

    is_valid: bool
    status: str
    plate_number: Optional[str] = None  # Corrected & normalised
    raw_text: Optional[str] = None
    confidence: Optional[float] = None


class ValidatorService:
    """Validates and corrects OCR text for Indian registration numbers."""

    def validate(self, raw_text: str, confidence: float) -> ValidationResult:
        """Run the full validate + correct pipeline.

        Steps:
        1. Check confidence threshold.
        2. Apply position-aware OCR corrections.
        3. Normalise (strip spaces/hyphens, upper-case).
        4. Validate against Indian plate regex.

        Args:
            raw_text:   Raw OCR string from OCR service.
            confidence: OCR confidence score (0–1).

        Returns:
            ``ValidationResult`` with status and corrected plate number.
        """
        if confidence < settings.OCR_MIN_CONFIDENCE:
            logger.debug(
                "OCR confidence %.3f below threshold %.3f – LOW_CONFIDENCE",
                confidence, settings.OCR_MIN_CONFIDENCE,
            )
            return ValidationResult(
                is_valid=False,
                status=STATUS_LOW_CONFIDENCE,
                raw_text=raw_text,
                confidence=confidence,
            )

        # Step 1: apply positional OCR corrections
        corrected = correct_ocr_text(raw_text)

        # Step 2: normalise
        normalised = normalise_plate_number(corrected)

        # Step 3: regex validation
        if not is_valid_indian_plate(normalised):
            logger.debug(
                "Plate '%s' (raw: '%s') failed regex validation.",
                normalised, raw_text,
            )
            return ValidationResult(
                is_valid=False,
                status=STATUS_INVALID_PLATE,
                plate_number=normalised,
                raw_text=raw_text,
                confidence=confidence,
            )

        logger.debug("Plate validated: '%s' (confidence=%.3f)", normalised, confidence)
        return ValidationResult(
            is_valid=True,
            status=STATUS_SUCCESS,
            plate_number=normalised,
            raw_text=raw_text,
            confidence=confidence,
        )
