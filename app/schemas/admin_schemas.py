# app/schemas/admin_schemas.py
"""
관리자/삭제 관련 스키마
"""

from pydantic import BaseModel
from typing import List, Optional
from .commons_schemas import BaseResponse

class DeleteRequest(BaseModel):
    item_id: str
    item_type: str  # keepsake/photo/letter
    user_id: str

class DeleteResponse(BaseResponse):
    deleted_from_collections: List[str]
    deleted_count: int
    item_id: Optional[str] = None
    item_category: Optional[str] = None

class IndividualDeleteRequest(BaseModel):
    user_id: str

class IndividualDeleteResponse(BaseResponse):
    item_id: str
    deleted_count: int