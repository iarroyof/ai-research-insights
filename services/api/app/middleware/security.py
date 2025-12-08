# services/api/app/middleware/security.py
from __future__ import annotations

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 🔓 Public graph viewer: read-only, no API key required
        if path.startswith("/triplets/graph/view"):
            return await call_next(request)

        # 🔐 Normal API key checks for the rest
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing X-API-Key")

        expected = getattr(settings, "api_key", None)
        if expected and api_key != expected:
            raise HTTPException(status_code=403, detail="Invalid X-API-Key")

        return await call_next(request)
