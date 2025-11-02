from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=400)
    target: str = Field(default="all")
    filters: Optional[dict] = None

class SearchItem(BaseModel):
    paper_id: str
    title: str
    pmid: str | None = None
    pmcid: str | None = None
    page: int | None = None
    sent_id: int | None = None
    score: float | None = None

class SearchResponse(BaseModel):
    items: List[SearchItem]
