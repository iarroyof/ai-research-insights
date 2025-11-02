# services/api/app/middleware/tenant.py
from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow health without tenant header
        if request.url.path == "/health":
            return await call_next(request)

        tenant_id = request.headers.get("x-tenant-id")  # case-insensitive
        if not tenant_id or not tenant_id.strip():
            return JSONResponse(
                status_code=400,
                content={"detail": "Missing X-Tenant-Id"},
            )

        request.state.tenant_id = tenant_id.strip()
        return await call_next(request)

