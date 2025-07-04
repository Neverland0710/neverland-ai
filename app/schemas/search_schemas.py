# app/schemas/search_schemas.py
"""
지능형 검색 관련 스키마
"""

from pydantic import BaseModel
from typing import List, Optional, Dict
from .commons_schemas import BaseResponse


class SearchRequest(BaseModel):
    user_id: str
    query: str
    context: Optional[str] = ""
    emotion_filter: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None
    limit: Optional[int] = 5


class SearchMemory(BaseModel):
    content: str
    collection: str
    score: float
    date: Optional[str] = None
    item_category: Optional[str] = None


class SearchResponse(BaseResponse):
    memories: List[SearchMemory]
    total_found: int
    search_strategy: Optional[str] = None
