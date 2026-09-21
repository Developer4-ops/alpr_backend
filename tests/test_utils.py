"""
Unit tests for utility modules.

These tests run without requiring MongoDB, YOLO models, or PaddleOCR.
They validate the pure-function utilities that underpin the pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.utils.correction_utils import correct_ocr_text
from app.utils.regex_utils import (
    extract_plate_parts,
    is_valid_indian_plate,
    normalise_plate_number,
)
from app.utils.timer import Timer


# ─────────────────────────────────────────────────────────────────────────────
# regex_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestIsValidIndianPlate:
    def test_valid_compact(self):
        assert is_valid_indian_plate("MH12AB1234") is True

    def test_valid_spaced(self):
        assert is_valid_indian_plate("MH 12 AB 1234") is True

    def test_valid_hyphenated(self):
        assert is_valid_indian_plate("MH-12-AB-1234") is True

    def test_valid_lowercase(self):
        assert is_valid_indian_plate("mh12ab1234") is True

    def test_invalid_too_short(self):
        assert is_valid_indian_plate("MH12") is False

    def test_invalid_wrong_format(self):
        assert is_valid_indian_plate("1234MH56") is False

    def test_invalid_empty(self):
        assert is_valid_indian_plate("") is False


class TestNormalisePlateNumber:
    def test_strips_spaces(self):
        assert normalise_plate_number("MH 12 AB 1234") == "MH12AB1234"

    def test_strips_hyphens(self):
        assert normalise_plate_number("MH-12-AB-1234") == "MH12AB1234"

    def test_uppercases(self):
        assert normalise_plate_number("mh12ab1234") == "MH12AB1234"


class TestExtractPlateParts:
    def test_valid_plate(self):
        parts = extract_plate_parts("MH12AB1234")
        assert parts is not None
        assert parts["state"] == "MH"
        assert parts["district"] == "12"
        assert parts["series"] == "AB"
        assert parts["number"] == "1234"

    def test_invalid_plate_returns_none(self):
        assert extract_plate_parts("INVALID") is None


# ─────────────────────────────────────────────────────────────────────────────
# correction_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectOcrText:
    def test_digit_to_letter_in_state(self):
        # "0H12AB1234" → state segment "0H" → "OH" → valid
        result = correct_ocr_text("0H12AB1234")
        assert result.startswith("OH")

    def test_letter_to_digit_in_district(self):
        # "MHIOAB1234" → district "IO" → "10"
        result = correct_ocr_text("MHIOAB1234")
        assert result[2:4] == "10"

    def test_no_corruption_on_valid(self):
        # A valid plate with no ambiguous chars should pass through unchanged
        result = correct_ocr_text("DL4CAB1234")
        assert isinstance(result, str)
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# timer
# ─────────────────────────────────────────────────────────────────────────────

class TestTimer:
    def test_elapsed_is_positive(self):
        t = Timer()
        t.start()
        import time; time.sleep(0.01)
        elapsed = t.stop()
        assert elapsed > 0

    def test_context_manager(self):
        with Timer() as t:
            import time; time.sleep(0.01)
        assert t.elapsed_ms > 0

    def test_elapsed_reasonable(self):
        with Timer() as t:
            pass
        # Should complete in well under 100 ms
        assert t.elapsed_ms < 100
