# services/api/app/middleware/security.py
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, HTTPException, status
from app.config import settings

class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip checks for /health or public endpoints if needed
        if request.url.path.startswith("/health"):
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        expected = settings.security.api_key

        if not api_key or api_key != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
            )

        # Continue request chain
        return await call_next(request)
