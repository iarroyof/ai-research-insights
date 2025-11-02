from opensearchpy import OpenSearch
from app.config import settings

_os = OpenSearch(settings.os_url)

def idx(name: str) -> str:
    return f"{settings.os_index_prefix}{name}"

PAPERS  = lambda t: idx(f"{t}_papers")
CHUNKS  = lambda t: idx(f"{t}_chunks")
TRIPLETS= lambda t: idx(f"{t}_triplets")
CHATS   = lambda t: idx(f"{t}_chats")
