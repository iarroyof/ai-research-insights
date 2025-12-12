# services/api/app/middleware/security.py
from __future__ import annotations
import os
from typing import Callable, Awaitable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.config import settings


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Simple API key security middleware.
    - Reads X-API-Key from the request headers.
    - Also accepts ?api_key=... as a fallback (useful for browser-only flows).
    - Compares it against settings.security.api_key (with env var override).
    - If security.require_api_key is False, the check is bypassed.
    - On failure, returns 403 with a JSON error body (no unhandled exceptions).
    """

    # Public endpoints that don't require authentication
    PUBLIC_PATHS = [
        "/health",
        "/triplets/graph/view",
    ]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[JSONResponse]],
    ):
        path = request.url.path
        cfg = settings.security

        # 🔓 Public endpoints: no API key required
        if path == "/health" or path.startswith("/triplets/graph/view"):
            request.state.api_key_valid = True  # Mark as valid for downstream
            return await call_next(request)

        # If API key is not required globally, just pass through
        if not cfg.require_api_key:
            return await call_next(request)

        # 1) Primary: header
        header_key = request.headers.get("X-API-Key")
        # 2) Fallback: query param (for browser-accessible URLs)
        query_key = request.query_params.get("api_key")
        effective_key = header_key or query_key

        # Get expected key: env var takes precedence over config file
        expected_key = os.getenv("API_KEY") or cfg.api_key

        # Missing or mismatching key → 403
        if not effective_key or effective_key != expected_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid X-API-Key"},
            )

        # Expose auth status to downstream handlers
        request.state.api_key_valid = True
        return await call_next(request)

