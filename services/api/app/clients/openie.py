import httpx
from app.config import settings

class OpenIEClient:
    def __init__(self):
        self.base = settings.openie6_adapter_url

    async def extract(self, sentences: list[str], num_extractions: int = 5):
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self.base}/extract", json={
                "sentences": sentences,
                "num_extractions": num_extractions
            })
            r.raise_for_status()
            return r.json()

    async def health(self):
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{self.base}/health")
            r.raise_for_status()
            return r.json()
