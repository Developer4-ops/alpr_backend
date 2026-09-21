"""
Production entry point.

Usage:
    python run.py

Starts the Uvicorn ASGI server with settings loaded from the .env file.
For production use with multiple workers, prefer running via gunicorn:

    gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4
"""
from __future__ import annotations

import uvicorn

from app.core.config import settings


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
