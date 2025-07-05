"""
사진 모델 - DB 조회 메서드 추가
"""

from sqlalchemy import Column, String, Text, DateTime, Date, BigInteger, ForeignKey
from sqlalchemy.future import select
from datetime import datetime
from typing import Optional, Dict
from .base import Base, get_async_session
from app.utils.logger import logger

class PhotoAlbum(Base):
    __tablename__ = "photo_album_TB"
    
    PHOTO_ID = Column(String(36), primary_key=True)
    AUTH_KEY_ID = Column(String(36), ForeignKey('auth_key_TB.AUTH_KEY_ID'), nullable=False)
    TITLE = Column(String(200))
    PHOTO_DATE = Column(Date)
    DESCRIPTION = Column(Text)
    IMAGE_PATH = Column(Text, nullable=False)
    FILE_SIZE = Column(BigInteger)
    FILE_FORMAT = Column(String(10))
    UPLOADED_AT = Column(DateTime, default=datetime.utcnow, nullable=False)

    @classmethod
    async def get_by_id(cls, photo_id: str) -> Optional[Dict]:
        """사진 ID로 조회"""
        try:
            async for session in get_async_session():
                query = select(cls).where(cls.PHOTO_ID == photo_id)
                result = await session.execute(query)
                photo = result.scalar_one_or_none()
                
                if not photo:
                    return None
                
                return {
                    "photo_id": photo.PHOTO_ID,
                    "authKeyId": photo.AUTH_KEY_ID,
                    "title": photo.TITLE,
                    "photo_date": photo.PHOTO_DATE.strftime("%Y-%m-%d") if photo.PHOTO_DATE else None,
                    "description": photo.DESCRIPTION,
                    "image_path": photo.IMAGE_PATH,
                    "file_size": photo.FILE_SIZE,
                    "file_format": photo.FILE_FORMAT,
                    "uploaded_at": photo.UPLOADED_AT.strftime("%Y-%m-%d") if photo.UPLOADED_AT else None
                }
                break
        except Exception as e:
            logger.error(f" 사진 조회 실패: {e}")
            return None