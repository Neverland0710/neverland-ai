# app/models/keepsake.py
"""
유품 모델 - DB 조회 메서드 추가
"""

from sqlalchemy import Column, String, Text, DateTime, BigInteger, ForeignKey
from sqlalchemy.future import select
from datetime import datetime
from typing import Optional, Dict
from .base import Base, get_async_session
from app.utils.logger import logger

class Keepsake(Base):
    __tablename__ = "keepsake_TB"
    
    KEEPSAKE_ID = Column(String(36), primary_key=True)
    AUTH_KEY_ID = Column(String(36), ForeignKey('auth_key_TB.AUTH_KEY_ID'), nullable=False)
    ITEM_NAME = Column(String(200), nullable=False)
    ACQUISITION_PERIOD = Column(String(100))
    DESCRIPTION = Column(Text)
    SPECIAL_STORY = Column(Text)
    ESTIMATED_VALUE = Column(BigInteger)
    IMAGE_PATH = Column(Text)
    CREATED_AT = Column(DateTime, default=datetime.utcnow, nullable=False)

    @classmethod
    async def get_by_id(cls, keepsake_id: str) -> Optional[Dict]:
        """유품 ID로 조회"""
        try:
            async for session in get_async_session():
                query = select(cls).where(cls.KEEPSAKE_ID == keepsake_id)
                result = await session.execute(query)
                keepsake = result.scalar_one_or_none()
                
                if not keepsake:
                    return None
                
                return {
                    "keepsake_id": keepsake.KEEPSAKE_ID,
                    "authKeyId": keepsake.AUTH_KEY_ID,
                    "item_name": keepsake.ITEM_NAME,
                    "acquisition_period": keepsake.ACQUISITION_PERIOD,
                    "description": keepsake.DESCRIPTION,
                    "special_story": keepsake.SPECIAL_STORY,
                    "estimated_value": keepsake.ESTIMATED_VALUE,
                    "image_path": keepsake.IMAGE_PATH,
                    "created_at": keepsake.CREATED_AT.strftime("%Y-%m-%d") if keepsake.CREATED_AT else None
                }
                break
        except Exception as e:
            logger.error(f" 유품 조회 실패: {e}")
            return None