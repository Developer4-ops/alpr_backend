"""
Application-wide constants.

Values here are truly constant and never loaded from environment.
For configurable thresholds, use ``app.core.config.settings``.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# YOLO class labels (COCO subset relevant to vehicles)
# ─────────────────────────────────────────────────────────────────────────────
VEHICLE_CLASS_NAMES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASS_IDS: frozenset[int] = frozenset(VEHICLE_CLASS_NAMES.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────
# Target short-side dimension for plate crops before OCR
PLATE_TARGET_HEIGHT: int = 64
PLATE_TARGET_WIDTH: int = 256

# Minimum acceptable plate crop dimensions (pixels)
MIN_PLATE_WIDTH: int = 60
MIN_PLATE_HEIGHT: int = 20

# CLAHE clip limit and grid size
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)

# Gaussian blur kernel (must be odd)
GAUSSIAN_KERNEL_SIZE: tuple[int, int] = (3, 3)

# Median blur kernel
MEDIAN_KERNEL_SIZE: int = 3

# Sharpening kernel
SHARPEN_KERNEL = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
]

# ─────────────────────────────────────────────────────────────────────────────
# OCR / Validation
# ─────────────────────────────────────────────────────────────────────────────
# Indian registration number regex
# Format: XX 00 XX 0000  (state code + district + series + number)
# Supports both spaced and compact forms.
INDIAN_PLATE_REGEX: str = (
    r"^[A-Z]{2}[\s-]?[0-9]{1,2}[\s-]?[A-Z]{1,3}[\s-]?[0-9]{1,4}$"
)

# OCR character confusion map used in position-aware correction
# Maps misread char → correct char for specific positions
OCR_CHAR_CORRECTIONS: dict[str, str] = {
    # Letter positions: digit-like glyphs read as letters
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "2": "Z",
    # Digit positions: letter-like glyphs read as digits
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
}

# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
API_V1_PREFIX: str = "/api/v1"

ALLOWED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
)

ALLOWED_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mkv", ".mov"}
)

# ─────────────────────────────────────────────────────────────────────────────
# Result status codes (internal)
# ─────────────────────────────────────────────────────────────────────────────
STATUS_SUCCESS: str = "SUCCESS"
STATUS_LOW_CONFIDENCE: str = "LOW_CONFIDENCE"
STATUS_NO_VEHICLE: str = "NO_VEHICLE_DETECTED"
STATUS_NO_PLATE: str = "NO_PLATE_DETECTED"
STATUS_INVALID_PLATE: str = "INVALID_PLATE_FORMAT"
STATUS_ERROR: str = "ERROR"
