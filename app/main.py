"""
FastAPI application factory.

Constructs and configures the FastAPI application instance.  The actual
server is started by ``run.py`` (production) or ``uvicorn`` directly.

Design decisions:
- ``lifespan`` context manager handles startup/shutdown (preferred over
  deprecated ``on_event`` decorators).
- Exception handlers are registered here to keep routes clean.
- The API router is included under the ``/api/v1`` prefix.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.constants import API_V1_PREFIX
from app.middleware.timing import TimingMiddleware
from app.startup.lifespan import lifespan

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory. Returns a fully configured FastAPI instance."""

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-grade Automatic Number Plate Recognition (ANPR) backend. "
            "Phase 1: Single-image detection. Phase 2: Video & RTSP streams."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ───────────────────────────────────────────────────────
    app.add_middleware(TimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ───────────────────────────────────────────────
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        logger.warning("HTTP %s: %s | path=%s", exc.status_code, exc.detail, request.url.path)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "ERROR",
                "detail": exc.detail,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Validation error: %s | path=%s", exc.errors(), request.url.path)
        return JSONResponse(
            status_code=422,
            content={
                "status": "ERROR",
                "detail": "Request validation failed.",
                "errors": exc.errors(),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception: %s | path=%s", exc, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "detail": "An internal server error occurred.",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    # ── Routers ──────────────────────────────────────────────────────────
    from app.api.routes import router as detection_router  # noqa: PLC0415
    app.include_router(detection_router, prefix=API_V1_PREFIX)

    logger.debug("FastAPI application created.")
    return app


# Module-level application instance (used by uvicorn / gunicorn)
app = create_app()
