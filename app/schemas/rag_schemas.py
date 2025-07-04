# app/schemas/rag_schemas.py (새로 추가)
"""
RAG 서비스 관련 스키마
"""
from app.schemas.commons_schemas import BaseResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class RAGSearchRequest(BaseModel):
    query: str
    user_id: str
    collection_name: Optional[str] = None
    metadata_filter: Optional[Dict] = None
    limit: Optional[int] = 5

class RAGSearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class RAGStoreRequest(BaseModel):
    content: str
    user_id: str
    memory_type: str
    metadata: Dict[str, Any]
    collection_name: Optional[str] = None

class RAGStoreResponse(BaseResponse):
    stored_id: Optional[str] = None
    collection: str
    metadata: Dict[str, Any]