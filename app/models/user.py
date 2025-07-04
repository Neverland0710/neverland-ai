# app/models/user.py
"""
사용자 모델
"""

from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy import Enum as SQLEnum
from datetime import datetime
from .base import Base

class User(Base):
    __tablename__ = "user_TB"
    
    USER_ID = Column(String(36), primary_key=True)
    NAME = Column(String(100), nullable=False)
    EMAIL = Column(String(100), unique=True, nullable=False)
    SOCIAL_PROVIDER = Column(SQLEnum('GOOGLE', 'KAKAO', 'NAVER', 'APPLE'), nullable=False)
    SOCIAL_ID = Column(String(100), nullable=False)
    PROFILE_IMAGE_URL = Column(Text)
    JOINED_AT = Column(DateTime, default=datetime.utcnow)
    RELATION_TO_DECEASED = Column(String(100))