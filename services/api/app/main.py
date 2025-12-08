# services/api/app/main.py
from __future__ import annotations
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .middleware.security import SecurityMiddleware  # Fixed: was ApiKeyMiddleware
from .middleware.tenant import TenantMiddleware

# Routers
from .routers import search as search_router
from .routers import chat as chat_router
from .routers import triplets as triplets_router
from .routers import papers as papers_router
from .routers import annotations as annotations_router
from .routers import ingest

log = logging.getLogger("uvicorn")

app = FastAPI(title="AI Research Insights API", version="0.1.0")

# CORS
allow_origins = settings.app.allow_cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or [],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security & Tenant middlewares
# Note: Order matters! Security checks first, then tenant resolution
app.add_middleware(SecurityMiddleware)  # Fixed: was ApiKeyMiddleware
app.add_middleware(TenantMiddleware)

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Mount routers
app.include_router(search_router.router)
app.include_router(chat_router.router)
app.include_router(triplets_router.router)
app.include_router(papers_router.router)
app.include_router(annotations_router.router)
app.include_router(ingest.router)
