# app/schemas/letter_schemas.py

from typing import List, Optional
from pydantic import BaseModel


# 요청 스키마
class LetterRequest(BaseModel):
    letter_id: str                
    user_id: str
    authKeyId: str
    letter_text: str


# 내부 응답 (LLM 처리 및 저장 포함)
class LetterProcessInternalResult(BaseModel):
    response: str
    summary_stored: Optional[str] = None
    emotion_tone: Optional[str] = None
    tags: Optional[List[str]] = None
    processing_time: Optional[float] = None


# 외부로 나가는 최종 응답 (요약/감정 제외)
class LetterProcessResponse(BaseModel):
    status: str = "success"
    response: str
