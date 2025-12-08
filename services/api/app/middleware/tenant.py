# services/api/app/middleware/tenant.py
from __future__ import annotations

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 🔓 Public graph viewer:
        # Allow tenant from header, query param, or fallback to "default"
        if path.startswith("/triplets/graph/view"):
            tenant_id = (
                request.headers.get("X-Tenant-Id")
                or request.query_params.get("tenant")
                or "default"
            )
            request.state.tenant_id = tenant_id
            return await call_next(request)

        # 🔐 Normal behavior for all other endpoints
        tenant_id = request.headers.get("X-Tenant-Id")
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Missing X-Tenant-Id")

        request.state.tenant_id = tenant_id
        return await call_next(request)
