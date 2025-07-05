# app/models/base.py
"""
SQLAlchemy Base 설정 및 DB 연결 관리
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import settings
from app.utils.logger import logger

# Base 모델
Base = declarative_base()

# 비동기 엔진 생성
engine = create_async_engine(
    f"mysql+aiomysql://{settings.mysql_user}:{settings.mysql_password}@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}",
    echo=settings.debug,
    pool_pre_ping=True,
    pool_recycle=3600
)

# 비동기 세션 팩토리
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """비동기 세션 제공"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """테이블 생성 (초기 설정용)"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(" 데이터베이스 테이블 생성 완료")
    except Exception as e:
        logger.error(f" 테이블 생성 실패: {e}")