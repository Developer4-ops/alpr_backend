"""
Image preprocessing pipeline for license plate OCR enhancement.

The pipeline applies a sequence of OpenCV operations designed to maximise
OCR accuracy on plate crops.  Each operation is applied conditionally
based on image characteristics rather than blindly running every step.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from app.core.constants import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    GAUSSIAN_KERNEL_SIZE,
    MEDIAN_KERNEL_SIZE,
    PLATE_TARGET_HEIGHT,
    PLATE_TARGET_WIDTH,
    SHARPEN_KERNEL,
)
from app.utils.image_utils import resize_image, to_grayscale

logger = logging.getLogger(__name__)


class ImagePreprocessorService:
    """Stateless service that applies an ordered preprocessing pipeline."""

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def preprocess(self, plate_crop: np.ndarray) -> np.ndarray:
        """Run the full plate-enhancement pipeline.

        Args:
            plate_crop: BGR NumPy array of the detected plate region.

        Returns:
            Preprocessed BGR array ready for PaddleOCR.
        """
        img = plate_crop.copy()

        img = self._upscale_if_needed(img)
        img = self._correct_perspective(img)
        img = self._deskew(img)
        img = self._to_grayscale(img)
        img = self._remove_noise(img)
        img = self._apply_clahe(img)
        img = self._sharpen(img)
        img = self._adaptive_threshold(img)
        img = self._morphological_close(img)
        img = self._to_bgr(img)          # PaddleOCR expects BGR/RGB

        return img

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #

    @staticmethod
    def _upscale_if_needed(img: np.ndarray) -> np.ndarray:
        """Upscale small crops to the target resolution."""
        h, w = img.shape[:2]
        if w < PLATE_TARGET_WIDTH or h < PLATE_TARGET_HEIGHT:
            img = resize_image(img, width=PLATE_TARGET_WIDTH, height=PLATE_TARGET_HEIGHT)
            logger.debug("Plate upscaled to %dx%d", PLATE_TARGET_WIDTH, PLATE_TARGET_HEIGHT)
        return img

    @staticmethod
    def _correct_perspective(img: np.ndarray) -> np.ndarray:
        """Attempt perspective correction using contour-based approach.

        Falls back to the original image if a valid quadrilateral cannot
        be found (avoids degrading already-straight crops).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return img

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                # Warp perspective to a standard rectangle
                dst = np.array(
                    [[0, 0], [PLATE_TARGET_WIDTH, 0],
                     [PLATE_TARGET_WIDTH, PLATE_TARGET_HEIGHT], [0, PLATE_TARGET_HEIGHT]],
                    dtype=np.float32,
                )
                M = cv2.getPerspectiveTransform(pts, dst)
                warped = cv2.warpPerspective(
                    img, M, (PLATE_TARGET_WIDTH, PLATE_TARGET_HEIGHT)
                )
                logger.debug("Perspective correction applied.")
                return warped
        return img

    @staticmethod
    def _deskew(img: np.ndarray) -> np.ndarray:
        """Correct small rotation angles using Hough lines."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
        if lines is None:
            return img

        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if abs(angle) < 15:  # Only correct small skews
                angles.append(angle)

        if not angles:
            return img

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("Deskew applied: %.2f degrees", median_angle)
        return rotated

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        return to_grayscale(img)

    @staticmethod
    def _remove_noise(img: np.ndarray) -> np.ndarray:
        """Apply median + Gaussian blur for noise removal."""
        img = cv2.medianBlur(img, MEDIAN_KERNEL_SIZE)
        img = cv2.GaussianBlur(img, GAUSSIAN_KERNEL_SIZE, 0)
        return img

    @staticmethod
    def _apply_clahe(img: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalisation."""
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE,
        )
        return clahe.apply(img)

    @staticmethod
    def _sharpen(img: np.ndarray) -> np.ndarray:
        """Enhance edge contrast with an unsharp mask kernel."""
        kernel = np.array(SHARPEN_KERNEL, dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def _adaptive_threshold(img: np.ndarray) -> np.ndarray:
        """Binarise using adaptive Gaussian thresholding."""
        return cv2.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

    @staticmethod
    def _morphological_close(img: np.ndarray) -> np.ndarray:
        """Close small gaps in characters with a morphological kernel."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _to_bgr(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
