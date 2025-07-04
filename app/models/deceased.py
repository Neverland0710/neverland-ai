# app/models/deceased.py
"""
고인 정보 모델
"""

from typing import Dict, Optional
from sqlalchemy import Column, String, Text, DateTime, Date, ForeignKey
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, date
from .base import Base, AsyncSessionLocal
from app.utils.logger import logger

class Deceased(Base):
    __tablename__ = "deceased_TB"
    
    DECEASED_ID = Column(String(36), primary_key=True)
    NAME = Column(String(100), nullable=False)
    BIRTH_DATE = Column(Date)
    DEATH_DATE = Column(Date)
    PROFILE_IMAGE_PATH = Column(Text)
    SPEAKING_STYLE = Column(Text)
    NICKNAME = Column(String(50), nullable=False)
    PERSONALITY = Column(Text)
    HOBBIES = Column(Text)
    REGISTERED_AT = Column(DateTime, default=datetime.utcnow)
    CREATOR_USER_ID = Column(String(36), ForeignKey('user_TB.USER_ID'), nullable=False)

    @classmethod
    async def get_by_auth_key(cls, authKeyId: str) -> Optional[Dict]:
        """인증키로 고인 정보 + 사용자 관계 정보 조회"""
        try:
            # 순환 import 방지를 위해 함수 내에서 import
            from .auth import AuthKey
            from .user import User
            
            async with AsyncSessionLocal() as session:
                # AuthKey -> Deceased + User 3-way 조인 쿼리
                query = select(cls, User.NAME, User.RELATION_TO_DECEASED).join(
                    AuthKey, cls.DECEASED_ID == AuthKey.DECEASED_ID
                ).join(
                    User, AuthKey.USER_ID == User.USER_ID
                ).where(
                    AuthKey.AUTH_KEY_ID == authKeyId,
                    AuthKey.IS_VALID == True
                )
                result = await session.execute(query)
                row = result.first()
                
                if not row:
                    logger.warning(f"⚠️ 고인 정보를 찾을 수 없음: authKeyId='{authKeyId}'")
                    return None
                
                deceased, user_name, relation = row
                
                # 나이 계산
                age = None
                if deceased.BIRTH_DATE:
                    today = date.today()
                    age = today.year - deceased.BIRTH_DATE.year
                
                # 사망 후 경과 시간 계산
                days_since_death = None
                if deceased.DEATH_DATE:
                    today = date.today()
                    days_since_death = (today - deceased.DEATH_DATE).days
                
                return {
                    "deceased_id": deceased.DECEASED_ID,
                    "name": deceased.NAME,
                    "nickname": deceased.NICKNAME,
                    "speaking_style": deceased.SPEAKING_STYLE,
                    "personality": deceased.PERSONALITY,
                    "hobbies": deceased.HOBBIES,
                    "birth_date": deceased.BIRTH_DATE.isoformat() if deceased.BIRTH_DATE else None,
                    "death_date": deceased.DEATH_DATE.isoformat() if deceased.DEATH_DATE else None,
                    "age": age,
                    "days_since_death": days_since_death,
                    "profile_image_path": deceased.PROFILE_IMAGE_PATH,
                    # 사용자 관계 정보 추가
                    "user_name": user_name,
                    "relation_to_user": relation
                }
                
        except Exception as e:
            logger.error(f"❌ 고인 정보 조회 실패: {e}")
            return None