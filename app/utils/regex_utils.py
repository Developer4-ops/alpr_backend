"""
Regex utilities for license plate validation.

Separated from the validator service so the patterns can be tested
independently and reused in other parts of the application.
"""
from __future__ import annotations

import re
from typing import Optional

from app.core.constants import INDIAN_PLATE_REGEX

# Compile once at import time for performance
_PLATE_PATTERN: re.Pattern[str] = re.compile(INDIAN_PLATE_REGEX, re.IGNORECASE)

# Normalisation: strip whitespace/hyphens for storage
_NORMALISE_PATTERN: re.Pattern[str] = re.compile(r"[\s\-]+")


def is_valid_indian_plate(text: str) -> bool:
    """Return True if *text* matches the Indian registration number format.

    Accepts compact (``MH12AB1234``) and spaced (``MH 12 AB 1234``) forms.
    Matching is case-insensitive.
    """
    cleaned = text.strip()
    return bool(_PLATE_PATTERN.fullmatch(cleaned))


def normalise_plate_number(text: str) -> str:
    """Return a canonical uppercase plate string with no spaces or hyphens.

    Example: ``"mh 12 ab 1234"`` → ``"MH12AB1234"``
    """
    return _NORMALISE_PATTERN.sub("", text.strip().upper())


def extract_plate_parts(text: str) -> Optional[dict[str, str]]:
    """Parse a validated plate number into its constituent parts.

    Returns a dict with keys: ``state``, ``district``, ``series``, ``number``.
    Returns ``None`` if the text does not match the expected format.
    """
    normalised = normalise_plate_number(text)
    # Match: 2 letters (state) + 1-2 digits (district) + 1-3 letters (series) + 1-4 digits
    match = re.fullmatch(
        r"([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})",
        normalised,
        re.IGNORECASE,
    )
    if not match:
        return None
    state, district, series, number = match.groups()
    return {
        "state": state.upper(),
        "district": district,
        "series": series.upper(),
        "number": number,
    }
