# app/schemas/search_schemas.py
"""
기억 검색 관련 스키마
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .commons_schemas import BaseResponse

class SearchRequest(BaseModel):
    query: str
    authKeyId: str  
    limit: Optional[int] = 5

class MemoryResult(BaseModel):
    collection: str
    content: str
    score: float
    date_text: Optional[str] = None
    emotion_tone: Optional[str] = None
    tags: Optional[List[str]] = None
    relevance_score: Optional[float] = None

class SearchResponse(BaseResponse):
    memories: List[MemoryResult]
    search_strategy: str
    total_found: int