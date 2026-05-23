# services/api/app/middleware/tenant.py
from __future__ import annotations
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


PUBLIC_EXACT_PATHS = {
    "/",
    "/health",
    "/favicon.ico",
    "/docs",
    "/openapi.json",
    "/redoc",
}

PUBLIC_PREFIXES = (
    "/triplets/graph/view",
)


def _is_public_path(path: str) -> bool:
    return path in PUBLIC_EXACT_PATHS or any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)


class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if _is_public_path(path):
            tenant_id = (
                request.headers.get("X-Tenant-Id")
                or request.query_params.get("tenant")
                or "default"
            )
            request.state.tenant_id = tenant_id
            return await call_next(request)

        tenant_id = request.headers.get("X-Tenant-Id")
        if not tenant_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "Missing X-Tenant-Id"},
            )

        request.state.tenant_id = tenant_id
        return await call_next(request)
