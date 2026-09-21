"""
Tracking service (Phase 2 stub).

In Phase 1 this module is a no-op stub.  When Phase 2 is implemented,
this service will integrate ByteTrack or DeepSORT to assign persistent
track IDs to vehicles across video frames and prevent duplicate MongoDB
entries for the same vehicle in consecutive frames.

The interface is defined here so callers can be written against it now
without needing to change when the implementation is filled in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Result from the tracking service for a single detection."""

    track_id: Optional[str] = None
    is_new_track: bool = True
    should_save: bool = True  # False when this is a duplicate in same track


class TrackingService:
    """Vehicle tracking across frames (Phase 2).

    Phase 1: Always returns ``should_save=True`` (no deduplication).
    Phase 2: Use ByteTrack/DeepSORT to maintain track state.
    """

    def __init__(self) -> None:
        # Phase 2: initialise tracker here
        logger.debug("TrackingService initialised (Phase 1 stub)")

    def process_detection(
        self,
        frame_id: int,
        plate_number: Optional[str],
        bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> TrackingResult:
        """Evaluate whether a detection should be persisted.

        Phase 1: Always saves.
        Phase 2: Check track history and dedup within cooldown window.

        Args:
            frame_id:     Sequential frame index (0 for single images).
            plate_number: Validated plate number or None.
            bbox:         (x1, y1, x2, y2) bounding box in frame coords.

        Returns:
            ``TrackingResult`` indicating whether to persist this event.
        """
        # Phase 1: trivial pass-through
        return TrackingResult(
            track_id=None,
            is_new_track=True,
            should_save=True,
        )

    def reset(self) -> None:
        """Reset tracker state (e.g. on new video/stream)."""
        logger.debug("TrackingService state reset.")
