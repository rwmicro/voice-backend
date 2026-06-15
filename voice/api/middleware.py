"""
Middleware for FastAPI application
Provides request tracking, error handling, and logging
"""

import asyncio
import time
import uuid
from collections import defaultdict
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from voice.utils.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track requests with unique IDs and timing
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"request_id": request_id},
        )

        # Track timing
        start_time = time.time()

        # Exception-to-response conversion is owned by ErrorHandlingMiddleware
        # (a separate, inner middleware), so we don't duplicate a 500 handler
        # here — we only measure timing and attach tracking headers.
        response = await call_next(request)

        duration = time.time() - start_time

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.3f}s"

        logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {duration:.3f}s",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration": duration,
            },
        )

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle errors consistently
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except ValueError as e:
            # Validation errors
            logger.warning(f"Validation error: {e}", exc_info=True)
            return JSONResponse(
                status_code=422,
                content={"error": "Validation Error", "message": str(e)},
            )
        except PermissionError as e:
            # Permission errors — don't expose internal paths
            logger.warning(f"Permission error: {e}")
            return JSONResponse(
                status_code=403, content={"error": "Forbidden", "message": "Access denied"}
            )
        except FileNotFoundError as e:
            # Not found errors — don't expose internal paths
            logger.warning(f"Not found: {e}")
            return JSONResponse(
                status_code=404, content={"error": "Not Found", "message": "Resource not found"}
            )
        except Exception as e:
            # Generic server errors
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                },
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting by client IP."""

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._counts: dict = defaultdict(list)
        self._lock = asyncio.Lock()

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/", "/api/voice/health"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = asyncio.get_event_loop().time()

        async with self._lock:
            # Clean old entries
            self._counts[client_ip] = [
                t for t in self._counts[client_ip]
                if now - t < self.window_seconds
            ]

            if len(self._counts[client_ip]) >= self.max_requests:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too Many Requests",
                        "message": f"Rate limit: {self.max_requests} requests per {self.window_seconds}s",
                        "retry_after": self.window_seconds,
                    },
                    headers={"Retry-After": str(self.window_seconds)},
                )

            self._counts[client_ip].append(now)

        return await call_next(request)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Cancel requests that take too long."""

    def __init__(self, app, timeout_seconds: int = 120):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip timeout for WebSocket upgrades
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout ({self.timeout_seconds}s): "
                f"{request.method} {request.url.path}"
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Gateway Timeout",
                    "message": f"Request processing exceeded {self.timeout_seconds}s limit",
                },
            )
