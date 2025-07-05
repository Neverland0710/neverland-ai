# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import chat, letter, memory, admin, search, voice
from app.services.scheduler_service import scheduler_service
from app.services.database_service import database_service
from app.utils.logger import setup_logger
from app.config import settings
from app.api.voice import router as voice_router
import uvicorn
import os

# 로거 설정
logger = setup_logger()

app = FastAPI(
    title="AI Memorial Chat Service",
    description="고정밀 RAG 기반 AI 추모 대화 서비스 + 실시간 음성 통화",
    version="2.1.0",
    debug=settings.debug
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포시에는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """앱 시작시 초기화"""
    logger.info(" AI Memorial Chat Service 시작")
    logger.info(f" Debug 모드: {settings.debug}")
    logger.info(f" Collections: {settings.daily_conversation_collection}, {settings.letter_memory_collection}, {settings.object_memory_collection}")
    
    # 정적 파일 디렉토리 생성
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/audio", exist_ok=True)
    logger.info(" 음성 서비스 디렉토리 생성 완료")
    
    # 데이터베이스 테이블 생성 (필요시)
    try:
        await database_service.create_tables()
        logger.info("🗄️ 데이터베이스 초기화 완료")
    except Exception as e:
        logger.warning(f" 데이터베이스 초기화 실패: {e}")
    
    # 스케줄러 시작 (일일 요약만)
    scheduler_service.start()
    logger.info(" 스케줄러 시작 완료")

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료시 정리"""
    logger.info(" AI Memorial Chat Service 종료")
    scheduler_service.stop()
    await database_service.close()

# 디렉토리 생성 후 정적 파일 마운트
os.makedirs("static", exist_ok=True)
os.makedirs("static/audio", exist_ok=True)

# 라우터 등록
app.include_router(chat.router, prefix="/api")
app.include_router(letter.router, prefix="/api")
app.include_router(memory.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(voice_router, prefix="/api")   # 음성 API 추가

# 정적 파일 서빙 (음성 파일용)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {
        "service": "AI Memorial Chat Service",
        "version": "2.1.0",
        "status": "running",
        "features": [
            "고정밀 RAG 검색",
            "편지 즉시 처리",
            "지능형 의도 분석",
            "감정 기반 회상",
            "실시간 음성 통화"  # 새 기능 추가
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "collections": {
            "daily_conversation": settings.daily_conversation_collection,
            "letter_memory": settings.letter_memory_collection,
            "object_memory": settings.object_memory_collection
        },
        "voice_service": {
            "elevenlabs_configured": bool(settings.elevenlabs_api_key),
            "audio_directory": "static/audio"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )