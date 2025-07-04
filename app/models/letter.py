# app/models/letter.py
"""
편지 모델 (AI 서버에서는 읽기 전용)
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy import Enum as SQLEnum
from datetime import datetime
from .base import Base

class Letter(Base):
    __tablename__ = "letter_TB"
    
    LETTER_ID = Column(String(36), primary_key=True)
    AUTH_KEY_ID = Column(String(36), ForeignKey('auth_key_TB.AUTH_KEY_ID'), nullable=False)
    TITLE = Column(String(200))
    CONTENT = Column(Text, nullable=False)
    CREATED_AT = Column(DateTime, default=datetime.utcnow, nullable=False)
    DELIVERY_STATUS = Column(SQLEnum('DRAFT', 'SENT', 'DELIVERED'), default='DRAFT', nullable=False)