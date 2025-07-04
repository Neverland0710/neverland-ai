# app/models/voice.py
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class VoiceTextRequest(BaseModel):
    """텍스트 입력 요청"""
    authKeyId: str
    user_text: str
    session_id: Optional[str] = None

class VoiceTextResponse(BaseModel):
    """텍스트 응답 (TTS 없이)"""
    status: str
    response: str
    voice_analysis: Optional[str] = None
    emotion_risk: Optional[str] = None
    used_memories: Optional[List[Dict]] = None
    timestamp: Optional[datetime] = None

class VoiceConversation(BaseModel):
    """음성 대화 기록"""
    conversation_id: str
    authKeyId: str
    user_text: str
    ai_text: str
    created_at: datetime
    processing_time: float

class DeceasedVoiceProfile(BaseModel):
    """고인 음성 프로필"""
    deceased_id: str
    name: str
    speaking_style: Optional[str] = None
    nickname: str
    personality: Optional[str] = None
    hobbies: Optional[str] = None
    voice_id: Optional[str] = None  # ElevenLabs voice ID