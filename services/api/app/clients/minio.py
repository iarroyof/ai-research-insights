# services/api/app/clients/minio.py

from __future__ import annotations

from minio import Minio
from urllib.parse import urlparse

# the rest of your file's imports...
try:
    # most of your code uses app.config.settings elsewhere
    from app.config import settings  # existing config loader
except Exception:
    # fallback to the new typed settings if needed
    from app.core.settings import settings  # typed pydantic settings

def _endpoint_parts(endpoint: str) -> tuple[str, bool]:
    """
    Return (host:port, secure) given an endpoint string like http://minio:9000.
    """
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        u = urlparse(endpoint)
        hostport = u.netloc or u.path  # tolerate bare host in odd configs
        return hostport, (u.scheme == "https")
    # bare host:port
    return endpoint, False

def get_minio() -> Minio:
    hostport, secure = _endpoint_parts(settings.minio.endpoint)
    return Minio(
        endpoint=hostport,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=secure,
    )

