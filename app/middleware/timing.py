"""
Request/Response timing middleware.

Adds an ``X-Process-Time-Ms`` response header to every HTTP response
containing the wall-clock time (in milliseconds) spent processing the
request.  This is useful for performance monitoring without needing to
instrument individual route handlers.
"""
from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp


class TimingMiddleware(BaseHTTPMiddleware):
    """Inject ``X-Process-Time-Ms`` header into every response."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response
