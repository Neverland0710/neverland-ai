# app/schemas/memory_schemas.py
"""
기억 생성 관련 스키마 (유품/사진)
"""

from pydantic import BaseModel
from typing import Optional, Dict
from .commons_schemas import BaseResponse, MemoryMetadata

class ProcessKeepsakeRequest(BaseModel):
    keepsake_id: str
    authKeyId: str

class ProcessPhotoRequest(BaseModel):
    photo_id: str
    authKeyId: str

class MemoryProcessResponse(BaseResponse):
    memory_content: Optional[str] = None
    item_id: Optional[str] = None
    item_category: Optional[str] = None
    metadata: Optional[Dict] = None

class BackgroundProcessResponse(BaseResponse):
    item_id: str
    task_id: Optional[str] = None