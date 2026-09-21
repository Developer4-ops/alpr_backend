"""
High-resolution wall-clock timer utility.

Wraps ``time.perf_counter`` for accurate sub-millisecond timing and
provides a context-manager interface for concise usage in service layers.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


class Timer:
    """Simple elapsed-time tracker.

    Usage::

        t = Timer()
        t.start()
        # … work …
        elapsed_ms = t.stop()

    Or as a context manager::

        with Timer() as t:
            # … work …
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0
        self.elapsed_ms: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds."""
        self._end = time.perf_counter()
        self.elapsed_ms = (self._end - self._start) * 1_000
        return self.elapsed_ms

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


@contextmanager
def timed_block(label: str = "") -> Generator[Timer, None, None]:
    """Context manager that yields a Timer and logs elapsed time.

    Example::

        with timed_block("OCR") as t:
            result = ocr_service.run(img)
        # t.elapsed_ms is available here
    """
    t = Timer()
    t.start()
    try:
        yield t
    finally:
        t.stop()
