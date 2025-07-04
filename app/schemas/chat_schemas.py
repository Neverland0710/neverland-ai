# app/schemas/chat_schemas.py
"""
실시간 대화 관련 스키마
"""

from pydantic import BaseModel
from typing import List, Optional, Dict
from .commons_schemas import BaseResponse, DeceasedInfo

class ChatRequest(BaseModel):
    authKeyId: str
    user_input: str
    user_id: str

class UsedMemory(BaseModel):
    collection: str
    content: str
    emotion_tone: Optional[str] = None
    tags: Optional[List[str]] = None
    relevance_score: Optional[float] = None

class ChatResponse(BaseResponse):
    response: str
    emotion_analysis: Optional[str] = None
    used_memories: Optional[List[UsedMemory]] = None
    conversation_id: Optional[str] = None