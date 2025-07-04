# app/models/auth.py
"""
인증키 모델
"""

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey
from datetime import datetime
from .base import Base

class AuthKey(Base):
    __tablename__ = "auth_key_TB"
    
    AUTH_KEY_ID = Column(String(36), primary_key=True)
    USER_ID = Column(String(36), ForeignKey('user_TB.USER_ID'), nullable=False)
    DECEASED_ID = Column(String(36), ForeignKey('deceased_TB.DECEASED_ID'), nullable=False)
    AUTH_CODE = Column(String(20), unique=True, nullable=False)
    IS_VALID = Column(Boolean, default=True, nullable=False)
    ISSUED_AT = Column(DateTime, default=datetime.utcnow, nullable=False)
    EXPIRED_AT = Column(DateTime)