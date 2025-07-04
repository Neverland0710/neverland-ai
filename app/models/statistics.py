# app/models/statistics.py
"""
통계 모델 (AI 서버에서는 읽기 전용)
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from datetime import datetime
from .base import Base

class Statistics(Base):
    __tablename__ = "statistics_TB"
    
    STAT_ID = Column(String(36), primary_key=True)
    USER_ID = Column(String(36), ForeignKey('user_TB.USER_ID'), unique=True, nullable=False)
    PHOTO_COUNT = Column(Integer, default=0, nullable=False)
    SENT_LETTER_COUNT = Column(Integer, default=0, nullable=False)
    KEEPSAKE_COUNT = Column(Integer, default=0, nullable=False)
    TOTAL_CONVERSATIONS = Column(Integer, default=0, nullable=False)
    LAST_UPDATED = Column(DateTime, default=datetime.utcnow, nullable=False)