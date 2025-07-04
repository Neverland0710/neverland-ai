# app/schemas/database_schemas.py (새로 추가)
"""
데이터베이스 관련 스키마
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class KeepsakeData(BaseModel):
    keepsake_id: str
    authKeyId: str
    item_name: str
    acquisition_period: Optional[str] = None
    description: Optional[str] = None
    special_story: Optional[str] = None
    estimated_value: Optional[int] = None
    image_path: Optional[str] = None
    created_at: Optional[str] = None

class PhotoData(BaseModel):
    photo_id: str
    authKeyId: str
    title: Optional[str] = None
    photo_date: Optional[str] = None
    description: Optional[str] = None
    image_path: Optional[str] = None
    file_size: Optional[int] = None
    file_format: Optional[str] = None
    uploaded_at: Optional[str] = None

class ConversationData(BaseModel):
    conversation_id: str
    authKeyId: str
    sender: str  # USER/CHATBOT
    message: str
    sent_at: str