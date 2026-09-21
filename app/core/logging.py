"""
Centralised logging configuration.

Sets up structured, colourised console logging and rotating file logging
using the standard `logging` module.  Call `setup_logging()` once at
application startup before any other import that might log.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from app.core.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Custom Formatter
# ─────────────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",     # Cyan
    logging.INFO: "\033[32m",      # Green
    logging.WARNING: "\033[33m",   # Yellow
    logging.ERROR: "\033[31m",     # Red
    logging.CRITICAL: "\033[35m",  # Magenta
}


class _ColourFormatter(logging.Formatter):
    """Console formatter that adds ANSI colour codes per log level."""

    _FMT = (
        "{color}{bold}%(levelname)-8s{reset}  "
        "%(asctime)s  "
        "{bold}%(name)s{reset}  "
        "%(message)s"
    )

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, RESET)
        fmt = self._FMT.format(color=color, bold=BOLD, reset=RESET)
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes)."""

    _FMT = "%(levelname)-8s  %(asctime)s  %(name)s  %(message)s"

    def __init__(self) -> None:
        super().__init__(self._FMT, datefmt="%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

_configured = False


def setup_logging() -> None:
    """Configure root logger with console + rotating-file handlers.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return
    _configured = True

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Ensure log directory exists
    Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(settings.LOG_DIR) / settings.LOG_FILE

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(_ColourFormatter())

    # ── Rotating file handler ────────────────────────────────────────────────
    # Rotate when file reaches ~10 MB, keep 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(_PlainFormatter())

    # ── Root logger ──────────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid adding duplicate handlers on re-import during hot-reload
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    # ── Silence noisy third-party loggers ────────────────────────────────────
    for noisy in ("uvicorn.access", "motor", "pymongo", "ppocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Always call ``setup_logging()`` first."""
    return logging.getLogger(name)
