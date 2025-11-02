from pydantic import BaseModel, Field
from typing import List, Optional

class ChatItem(BaseModel):
    paper_id: str
    sent_id: int | None = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1, max_length=4000)
    items: List[ChatItem] = []
    params: dict = {}

class ChatChunk(BaseModel):
    type: str  # token|final|error
    data: dict
