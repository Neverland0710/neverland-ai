from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import Response
import asyncio
from urllib.parse import quote

from app.services.voice_service import VoiceService
from app.services.database_service import database_service
from app.utils.logger import logger

router = APIRouter(prefix="/voice", tags=["voice"])
voice_service = VoiceService()

@router.post("/process")
async def process_voice_text(
    authKeyId: str = Form(...),
    user_text: str = Form(...)
):
    """텍스트를 받아서 하나의 MP3 파일로 통합 응답 및 인코딩된 응답 텍스트 반환"""
    try:
        logger.info(f" 음성 처리 시작: authKeyId={authKeyId}, text={user_text}")

        # 사용자 인증
        user_info = await database_service.get_user_by_auth_key(authKeyId)
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid auth key")

        # 텍스트 응답 + 오디오 동시 생성
        result = await voice_service.generate_response_and_voice(
            user_text=user_text,
            user_info=user_info,
            authKeyId=authKeyId
        )

        # 한글 텍스트 → Header용 인코딩
        encoded_text_response = quote(result["response_text"])

        # MP3 + 인코딩된 텍스트 헤더 응답
        return Response(
            content=result["audio_mp3"],
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=voice_response.mp3",
                "X-AI-Response": encoded_text_response,
                "Access-Control-Expose-Headers": "Content-Disposition, X-AI-Response",
                "Cache-Control": "no-cache",
                "Connection": "close"
            }
        )

    except Exception as e:
        logger.error(f" 음성 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

@router.get("/health")
async def voice_health_check():
    """음성 서비스 건강 상태 체크"""
    try:
        return {
            "status": "healthy",
            "service": "voice_api",
            "timestamp": asyncio.get_event_loop().time(),
            "features": [
                "HTTP TTS 통합 스트리밍",
                "AI 감성 대화 응답",
                "사용자별 음성 생성"
            ]
        }
    except Exception as e:
        logger.error(f" 건강 상태 체크 실패: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")