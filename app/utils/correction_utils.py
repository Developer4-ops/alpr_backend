"""
OCR character-correction utilities.

Indian number plates follow the format:
    [STATE:2L] [DISTRICT:1-2D] [SERIES:1-3L] [NUMBER:1-4D]
    e.g. MH 12 AB 1234

The correction logic is *position-aware*:
- In letter positions (state, series) we map digit-like chars → letters.
- In digit positions (district, number) we map letter-like chars → digits.

This prevents blindly converting every "0" to "O" or every "O" to "0".
"""
from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# Correction tables
# ─────────────────────────────────────────────────────────────────────────────

# Used when a LETTER is expected but a digit-like char was detected
_DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "2": "Z",
}

# Used when a DIGIT is expected but a letter-like char was detected
_LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
    "D": "0",
    "Q": "0",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _correct_letter_segment(segment: str) -> str:
    """Apply digit→letter substitutions for expected-letter positions."""
    return "".join(_DIGIT_TO_LETTER.get(c.upper(), c.upper()) for c in segment)


def _correct_digit_segment(segment: str) -> str:
    """Apply letter→digit substitutions for expected-digit positions."""
    return "".join(_LETTER_TO_DIGIT.get(c.upper(), c.upper()) for c in segment)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def correct_ocr_text(raw: str) -> str:
    """Apply position-aware corrections to a raw OCR string based on Indian Registration format:
        [STATE: 2 Letters] [DISTRICT: 1-2 Digits] [SERIES: 1-3 Letters] [NUMBER: 1-4 Digits]
    """
    if not raw or not raw.strip():
        return ""

    raw_clean = raw.strip().upper()
    tokens = [t for t in re.split(r"[\s\-]+", raw_clean) if t]

    # Case 1: Tokens separated by whitespace or hyphens (e.g. "MH 12 DE 1433", "HR 26 DQ 5555")
    if len(tokens) == 4:
        state_tok, dist_tok, series_tok, num_tok = tokens
        if len(state_tok) == 2 and 1 <= len(dist_tok) <= 2 and 1 <= len(series_tok) <= 3 and 1 <= len(num_tok) <= 4:
            st = _correct_letter_segment(state_tok)
            di = _correct_digit_segment(dist_tok)
            se = _correct_letter_segment(series_tok)
            nu = _correct_digit_segment(num_tok)
            return f"{st}{di}{se}{nu}"

    # Case 2: Unspaced or concatenated plate string (e.g. "MH12DE1433")
    cleaned = re.sub(r"[\s\-]", "", raw_clean)

    if len(cleaned) >= 7:
        st_raw = cleaned[:2]
        st = _correct_letter_segment(st_raw)
        rem = cleaned[2:]

        # Match District (1-2 digits), Series (1-3 letters), Number (1-4 digits)
        m = re.match(r"^([0-9OISBZGQ]{2}|[0-9OISBZGQ]{1})([A-Z]{1,3})([0-9]{1,4})$", rem)
        if not m:
            m = re.match(r"^([0-9OISBZGQ]{2}|[0-9OISBZGQ]{1})([A-ZOISBZGQ]{1,3})([0-9OISBZGQ]{1,4})$", rem)

        if m:
            di_raw, se_raw, nu_raw = m.groups()
            di = _correct_digit_segment(di_raw)
            se = _correct_letter_segment(se_raw)
            nu = _correct_digit_segment(nu_raw)
            return f"{st}{di}{se}{nu}"

    return cleaned


def _digit_to_letter_char(c: str) -> str:
    return _DIGIT_TO_LETTER.get(c.upper(), c.upper())


def _letter_to_digit_char(c: str) -> str:
    return _LETTER_TO_DIGIT.get(c.upper(), c.upper())
