# app/services/database_service.py
"""
MySQL 데이터베이스 서비스
SQLAlchemy 기반 비동기 DB 연결
"""

from typing import Dict, List, Optional
from datetime import datetime, date, timezone, timedelta
import uuid

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Text, DateTime, Enum, Date, BigInteger, Boolean, ForeignKey, Integer
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.utils.logger import logger

# 공통 시간대 (한국 시간)
KST = timezone(timedelta(hours=9))
def now_kst():
    return datetime.now(KST)

Base = declarative_base()

class User(Base):
    __tablename__ = "user_TB"
    USER_ID = Column(String(36), primary_key=True)
    NAME = Column(String(100), nullable=False)
    EMAIL = Column(String(100), unique=True, nullable=False)
    SOCIAL_PROVIDER = Column(Enum('GOOGLE', 'KAKAO', 'NAVER', 'APPLE'), nullable=False)
    SOCIAL_ID = Column(String(100), nullable=False)
    PROFILE_IMAGE_URL = Column(Text)
    JOINED_AT = Column(DateTime, default=now_kst)
    RELATION_TO_DECEASED = Column(String(100))

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
    VOICE_ID = Column(String(100), nullable=True)  # 🎵 ElevenLabs voice ID 추가
    REGISTERED_AT = Column(DateTime, default=now_kst)
    CREATOR_USER_ID = Column(String(36), ForeignKey('user_TB.USER_ID'), nullable=False)

class AuthKey(Base):
    __tablename__ = "auth_key_TB"
    AUTH_KEY_ID = Column(String(36), primary_key=True)
    USER_ID = Column(String(36), ForeignKey('user_TB.USER_ID'), nullable=False)
    DECEASED_ID = Column(String(36), ForeignKey('deceased_TB.DECEASED_ID'), nullable=False)
    AUTH_CODE = Column(String(20), unique=True, nullable=False)
    IS_VALID = Column(Boolean, default=True, nullable=False)
    ISSUED_AT = Column(DateTime, default=now_kst, nullable=False)
    EXPIRED_AT = Column(DateTime)

class TextConversation(Base):
    __tablename__ = "text_conversation_TB"
    CONVERSATION_ID = Column(String(36), primary_key=True)
    AUTH_KEY_ID = Column(String(36), ForeignKey('auth_key_TB.AUTH_KEY_ID'), nullable=False)
    SENDER = Column(Enum('USER', 'CHATBOT'), nullable=False)
    MESSAGE = Column(Text, nullable=False)
    SENT_AT = Column(DateTime, default=now_kst, nullable=False)

class Letter(Base):
    __tablename__ = "letter_TB"
    LETTER_ID = Column(String(36), primary_key=True)
    AUTH_KEY_ID = Column(String(36), ForeignKey('auth_key_TB.AUTH_KEY_ID'), nullable=False)
    TITLE = Column(String(200))
    CONTENT = Column(Text, nullable=False)
    CREATED_AT = Column(DateTime, default=now_kst, nullable=False)
    DELIVERY_STATUS = Column(Enum('DRAFT', 'SENT', 'DELIVERED'), default='DRAFT', nullable=False)

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
    UPLOADED_AT = Column(DateTime, default=now_kst, nullable=False)

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
    CREATED_AT = Column(DateTime, default=now_kst, nullable=False)

class Statistics(Base):
    __tablename__ = "statistics_TB"
    STAT_ID = Column(String(36), primary_key=True)
    USER_ID = Column(String(36), ForeignKey('user_TB.USER_ID'), unique=True, nullable=False)
    PHOTO_COUNT = Column(Integer, default=0, nullable=False)
    SENT_LETTER_COUNT = Column(Integer, default=0, nullable=False)
    KEEPSAKE_COUNT = Column(Integer, default=0, nullable=False)
    TOTAL_CONVERSATIONS = Column(Integer, default=0, nullable=False)
    LAST_UPDATED = Column(DateTime, default=now_kst, nullable=False)

class DatabaseService:
    def __init__(self):
        self.database_url = f"mysql+aiomysql://{settings.mysql_user}:{settings.mysql_password}@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
        self.engine = create_async_engine(self.database_url, echo=settings.debug, pool_pre_ping=True, pool_recycle=3600)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        logger.info(" DatabaseService 초기화 완료")

    async def create_tables(self):
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info(" 데이터베이스 테이블 생성 완료")
        except Exception as e:
            logger.error(f" 테이블 생성 실패: {e}")

    async def get_user_by_auth_key(self, authKeyId: str) -> Optional[Dict]:
        """기존 방식 유지 - 인증키로 사용자 정보 조회"""
        try:
            async with self.async_session() as session:
                query = select(User, Deceased, AuthKey).join(
                    AuthKey, User.USER_ID == AuthKey.USER_ID
                ).join(
                    Deceased, Deceased.DECEASED_ID == AuthKey.DECEASED_ID
                ).where(
                    AuthKey.AUTH_KEY_ID == authKeyId,
                    AuthKey.IS_VALID == True
                )
                result = await session.execute(query)
                row = result.first()
                if not row:
                    return None
                user, deceased, auth = row
                return {
                    "user_id": user.USER_ID,
                    "user_name": user.NAME,
                    "relation_to_deceased": user.RELATION_TO_DECEASED,
                    "deceased_name": deceased.NAME,
                    "deceased_nickname": deceased.NICKNAME,
                    "speaking_style": deceased.SPEAKING_STYLE,
                    "personality": deceased.PERSONALITY
                }

        except SQLAlchemyError as e:
            logger.error(f" 사용자 조회 실패: {e}")
            return None

    async def get_deceased_by_auth_key(self, authKeyId: str) -> Dict:
        """기존 방식 유지하면서 voice_id 추가"""
        try:
            async with self.async_session() as session:
                query = select(Deceased, User.NAME, User.RELATION_TO_DECEASED).join(
                    AuthKey, Deceased.DECEASED_ID == AuthKey.DECEASED_ID
                ).join(
                    User, AuthKey.USER_ID == User.USER_ID
                ).where(
                    AuthKey.AUTH_KEY_ID == authKeyId,
                    AuthKey.IS_VALID == True
                )
                result = await session.execute(query)
                row = result.first()
                if not row:
                    logger.warning(f" 고인 정보를 찾을 수 없음: authKeyId='{authKeyId}'")
                    return {}
                deceased, user_name, relation = row
                age = date.today().year - deceased.BIRTH_DATE.year if deceased.BIRTH_DATE else None
                days_since_death = (date.today() - deceased.DEATH_DATE).days if deceased.DEATH_DATE else None
                
                deceased_info = {
                    "deceased_id": deceased.DECEASED_ID,
                    "name": deceased.NAME,
                    "nickname": deceased.NICKNAME,
                    "speaking_style": deceased.SPEAKING_STYLE,
                    "personality": deceased.PERSONALITY,
                    "hobbies": deceased.HOBBIES,
                    "voice_id": deceased.VOICE_ID,  # 🎵 voice_id 추가
                    "birth_date": deceased.BIRTH_DATE.isoformat() if deceased.BIRTH_DATE else None,
                    "death_date": deceased.DEATH_DATE.isoformat() if deceased.DEATH_DATE else None,
                    "age": age,
                    "days_since_death": days_since_death,
                    "profile_image_path": deceased.PROFILE_IMAGE_PATH,
                    "user_name": user_name,
                    "relation_to_user": relation
                }
                
                # voice_id 로깅
                if deceased.VOICE_ID:
                    logger.info(f" 고인 정보 조회 성공: {deceased_info['name']} (voice_id: {deceased.VOICE_ID})")
                else:
                    logger.warning(f" 고인 {deceased_info['name']}의 voice_id가 설정되지 않음")
                
                return deceased_info
                
        except SQLAlchemyError as e:
            logger.error(f" 고인 정보 조회 실패: {e}")
            return {}

    async def update_deceased_voice_id(self, deceased_id: str, voice_id: str) -> bool:
        """고인의 voice_id 업데이트"""
        try:
            async with self.async_session() as session:
                # 고인 정보 조회
                query = select(Deceased).where(Deceased.DECEASED_ID == deceased_id)
                result = await session.execute(query)
                deceased = result.scalar_one_or_none()
                
                if deceased:
                    # voice_id 업데이트
                    deceased.VOICE_ID = voice_id
                    await session.commit()
                    
                    logger.info(f" voice_id 업데이트 성공: {deceased.NAME} -> {voice_id}")
                    return True
                else:
                    logger.warning(f" 고인을 찾을 수 없음: {deceased_id}")
                    return False
                    
        except SQLAlchemyError as e:
            logger.error(f" voice_id 업데이트 실패: {e}")
            await session.rollback()
            return False

    async def get_recent_conversations(self, authKeyId: str, limit: int = 5) -> List[Dict]:
        try:
            async with self.async_session() as session:
                query = select(TextConversation).where(
                    TextConversation.AUTH_KEY_ID == authKeyId
                ).order_by(TextConversation.SENT_AT.desc()).limit(limit)
                result = await session.execute(query)
                conversations = result.scalars().all()
                return [
                    {
                        "conversation_id": conv.CONVERSATION_ID,
                        "sender": conv.SENDER,
                        "message": conv.MESSAGE,
                        "sent_at": conv.SENT_AT.isoformat(),
                        "is_recent": True
                    }
                    for conv in conversations
                ]
        except SQLAlchemyError as e:
            logger.error(f" 대화 조회 실패: {e}")
            return []

    async def save_conversation(self, authKeyId: str, sender: str, message: str, metadata: Dict = None):
        try:
            from dateutil import parser
            from datetime import timezone, timedelta

            #  한국 시간으로 설정
            KST = timezone(timedelta(hours=9))

            sent_at = parser.isoparse(metadata["sent_at"]) if metadata and "sent_at" in metadata else datetime.now(KST)

            async with self.async_session() as session:
                conversation = TextConversation(
                    CONVERSATION_ID=str(uuid.uuid4()),
                    AUTH_KEY_ID=authKeyId,
                    SENDER=sender,
                    MESSAGE=message,
                    SENT_AT=sent_at
                )
                session.add(conversation)
                await session.commit()
                logger.info(f" 대화 저장 완료: sender={sender}, sent_at={sent_at.isoformat()}")
        except SQLAlchemyError as e:
            logger.error(f" 대화 저장 실패: {e}")

    async def save_letter(self, letter_id: str, authKeyId: str, title: str, content: str, status: str = "SENT"):
        try:
            async with self.async_session() as session:
                letter = Letter(
                    LETTER_ID=letter_id,
                    AUTH_KEY_ID=authKeyId,
                    TITLE=title,
                    CONTENT=content,
                    DELIVERY_STATUS=status
                )
                session.add(letter)
                await session.commit()
                logger.info(f" 편지 저장 완료: {letter_id}")
        except SQLAlchemyError as e:
            logger.error(f" 편지 저장 실패: {e}")

    async def get_letter_by_id(self, letter_id: str) -> Optional[Dict]:
        try:
            async with self.async_session() as session:
                query = select(Letter).where(Letter.LETTER_ID == letter_id)
                result = await session.execute(query)
                letter = result.scalar_one_or_none()
                if not letter:
                    return None
                return {
                    "letter_id": letter.LETTER_ID,
                    "authKeyId": letter.AUTH_KEY_ID,
                    "title": letter.TITLE,
                    "content": letter.CONTENT,
                    "created_at": letter.CREATED_AT.isoformat(),
                    "delivery_status": letter.DELIVERY_STATUS
                }
        except SQLAlchemyError as e:
            logger.error(f" 편지 조회 실패: {e}")
            return None

    async def get_keepsake_by_id(self, keepsake_id: str) -> Optional[Dict]:
        try:
            async with self.async_session() as session:
                query = select(Keepsake).where(Keepsake.KEEPSAKE_ID == keepsake_id)
                result = await session.execute(query)
                keepsake = result.scalar_one_or_none()
                if not keepsake:
                    return None
                return {
                    "keepsake_id": keepsake.KEEPSAKE_ID,
                    "authKeyId": keepsake.AUTH_KEY_ID,
                    "item_name": keepsake.ITEM_NAME,
                    "description": keepsake.DESCRIPTION,
                    "special_story": keepsake.SPECIAL_STORY,
                    "acquisition_period": keepsake.ACQUISITION_PERIOD,
                    "image_path": keepsake.IMAGE_PATH,
                    "created_at": keepsake.CREATED_AT.isoformat()
                }
        except SQLAlchemyError as e:
            logger.error(f" 유품 조회 실패: {e}")
            return None

    async def get_photo_by_id(self, photo_id: str) -> Optional[Dict]:
        try:
            async with self.async_session() as session:
                query = select(PhotoAlbum).where(PhotoAlbum.PHOTO_ID == photo_id)
                result = await session.execute(query)
                photo = result.scalar_one_or_none()
                if not photo:
                    return None
                return {
                    "photo_id": photo.PHOTO_ID,
                    "authKeyId": photo.AUTH_KEY_ID,
                    "title": photo.TITLE,
                    "description": photo.DESCRIPTION,
                    "photo_date": photo.PHOTO_DATE.isoformat() if photo.PHOTO_DATE else None,
                    "image_path": photo.IMAGE_PATH,
                    "file_size": photo.FILE_SIZE,
                    "file_format": photo.FILE_FORMAT,
                    "uploaded_at": photo.UPLOADED_AT.isoformat()
                }
        except SQLAlchemyError as e:
            logger.error(f" 사진 조회 실패: {e}")
            return None

    async def close(self):
        await self.engine.dispose()
        logger.info("🔌 데이터베이스 연결 종료")

# 전역 인스턴스화
database_service = DatabaseService()