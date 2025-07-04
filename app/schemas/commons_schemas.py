# app/schemas/commons_schemas.py
"""
공통 스키마 - 여러 API에서 공유하는 스키마들
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# 기본 응답 스키마
class BaseResponse(BaseModel):
    status: str
    message: Optional[str] = None
    timestamp: Optional[str] = None

# 고인 정보 스키마 (여러 곳에서 사용)
class DeceasedInfo(BaseModel):
    deceased_id: str
    name: str
    nickname: str
    speaking_style: Optional[str] = None
    personality: Optional[str] = None
    hobbies: Optional[str] = None
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    age: Optional[int] = None
    days_since_death: Optional[int] = None
    profile_image_path: Optional[str] = None
    user_name: Optional[str] = None
    relation_to_user: Optional[str] = None

# 메타데이터 스키마 (Qdrant용)
class MemoryMetadata(BaseModel):
    user_id: str
    item_id: str
    item_category: str  # keepsake/photo/letter/chat/summary
    type: str
    date: str
    created_at: str
    source: str

# 삭제 관련 스키마
class DeleteRequest(BaseModel):
    item_id: str
    item_type: str  # keepsake/photo/letter
    user_id: str

class DeleteResponse(BaseModel):
    status: str
    deleted_from_collections: List[str]
    deleted_count: int
    message: str