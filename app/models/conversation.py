"""
대화 모델
"""

import uuid
from typing import Dict, Optional
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy import Enum as SQLEnum
from datetime import datetime, timezone, timedelta
from dateutil import parser

from .base import Base, AsyncSessionLocal
from app.utils.logger import logger

# 한국 시간대 정의
KST = timezone(timedelta(hours=9))

class TextConversation(Base):
    __tablename__ = "text_conversation_TB"
    
    CONVERSATION_ID = Column(String(36), primary_key=True)
    AUTH_KEY_ID = Column(String(36), ForeignKey('auth_key_TB.AUTH_KEY_ID'), nullable=False)
    SENDER = Column(SQLEnum('USER', 'CHATBOT'), nullable=False)
    MESSAGE = Column(Text, nullable=False)
    SENT_AT = Column(DateTime, default=lambda: datetime.now(KST), nullable=False)  # 기본값도 KST

    @classmethod
    async def save_message(
        cls, 
        authKeyId: str, 
        sender: str, 
        message: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """대화 저장 (RunnableWithMessageHistory 보완용)"""
        try:
            # metadata에 sent_at이 있으면 파싱해서 사용, 없으면 현재 시각 (KST)
            if metadata and "sent_at" in metadata:
                sent_at = parser.isoparse(metadata["sent_at"]).astimezone(KST)
            else:
                sent_at = datetime.now(KST)

            async with AsyncSessionLocal() as session:
                conversation = cls(
                    CONVERSATION_ID=str(uuid.uuid4()),
                    AUTH_KEY_ID=authKeyId,
                    SENDER=sender,
                    MESSAGE=message,
                    SENT_AT=sent_at
                )
                session.add(conversation)
                await session.commit()
                logger.info(f" 대화 저장 완료: sender={sender}, sent_at={sent_at.isoformat()}")
                return True
                
        except Exception as e:
            logger.error(f" 대화 저장 실패: {e}")
            return False
