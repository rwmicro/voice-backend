"""
Middleware for FastAPI application
Provides request tracking, error handling, and logging
"""

import time
import uuid
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

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{duration:.3f}s"

            # Log response
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

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                f"Request failed: {request.method} {request.url.path} - "
                f"Error: {str(e)} - Duration: {duration:.3f}s",
                exc_info=True,
                extra={"request_id": request_id, "duration": duration},
            )

            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": str(e),
                    "request_id": request_id,
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": f"{duration:.3f}s",
                },
            )


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
            # Permission errors
            logger.warning(f"Permission error: {e}")
            return JSONResponse(
                status_code=403, content={"error": "Forbidden", "message": str(e)}
            )
        except FileNotFoundError as e:
            # Not found errors
            logger.warning(f"Not found: {e}")
            return JSONResponse(
                status_code=404, content={"error": "Not Found", "message": str(e)}
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
