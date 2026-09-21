"""
OCR service using PaddleOCR.

Wraps the PaddleOCR engine with a clean interface that returns structured
results.  The model is loaded once at startup (lazy-singleton pattern).
"""
from __future__ import annotations

import os
os.environ["FLAGS_use_mkldnn"] = "0"

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Structured OCR output for a single image."""

    text: str = ""
    confidence: float = 0.0
    bounding_boxes: list = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.text.strip()


class OCRService:
    """PaddleOCR wrapper with lazy initialisation."""

    def __init__(self) -> None:
        self._ocr = None
        self._loaded = False

    def load(self) -> None:
        """Initialise PaddleOCR. Called once at application startup."""
        if self._loaded:
            return

        try:
            import os
            os.environ["FLAGS_use_mkldnn"] = "0"
            from paddleocr import PaddleOCR  # type: ignore[import]
            self._ocr = PaddleOCR(
                lang=settings.OCR_LANG,
                enable_mkldnn=False,
            )
            logger.info("PaddleOCR initialised (lang=%s)", settings.OCR_LANG)
        except ImportError:
            logger.error(
                "OCR engine unavailable: paddleocr is not installed. "
                "Run: pip install paddleocr paddlepaddle"
            )
            self._ocr = None
        except Exception as exc:
            logger.error("OCR engine unavailable: Failed to initialise PaddleOCR: %s", exc)
            self._ocr = None
        finally:
            self._loaded = True

    def is_available(self) -> bool:
        """Return True if PaddleOCR is loaded and available for inference."""
        return self._ocr is not None

    def run(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a preprocessed plate image.

        Args:
            image: BGR NumPy array (preprocessed plate crop).

        Returns:
            ``OCRResult`` with the best aggregated text and mean confidence.
        """
        if self._ocr is None:
            logger.warning("OCR engine not loaded – returning empty result.")
            return OCRResult()

        try:
            raw_results = self._ocr.ocr(image)
        except Exception as exc:
            logger.error("PaddleOCR inference failed: %s", exc)
            return OCRResult()

        if not raw_results or not raw_results[0]:
            logger.debug("PaddleOCR returned no results.")
            return OCRResult()

        texts: list[str] = []
        confidences: list[float] = []
        bounding_boxes: list = []

        if isinstance(raw_results, list) and len(raw_results) > 0:
            res_item = raw_results[0]
            if isinstance(res_item, dict):
                rec_texts = res_item.get("rec_texts", [])
                rec_scores = res_item.get("rec_scores", [])
                rec_polys = res_item.get("rec_polys", [])
                for idx, t in enumerate(rec_texts):
                    if t and str(t).strip():
                        texts.append(str(t).strip())
                        conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
                        confidences.append(conf)
                        if idx < len(rec_polys):
                            bounding_boxes.append(rec_polys[idx])
            elif isinstance(res_item, list):
                for line in res_item:
                    if not line or len(line) < 2:
                        continue
                    bbox = line[0]
                    text_conf = line[1]
                    if not text_conf or len(text_conf) < 2:
                        continue
                    text, conf = text_conf[0], float(text_conf[1])
                    if text.strip():
                        texts.append(text.strip())
                        confidences.append(conf)
                        bounding_boxes.append(bbox)

        if not texts:
            return OCRResult()

        # Concatenate all text segments (plate may span multiple OCR words)
        combined_text = "".join(texts)
        mean_confidence = float(np.mean(confidences))

        logger.debug(
            "OCR result: text='%s' confidence=%.3f",
            combined_text, mean_confidence,
        )

        return OCRResult(
            text=combined_text,
            confidence=round(mean_confidence, 4),
            bounding_boxes=bounding_boxes,
        )
