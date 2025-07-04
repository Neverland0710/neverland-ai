# app/services/voice_service.py
import asyncio
import aiohttp
from typing import AsyncGenerator, Dict, Any
from datetime import datetime

from app.services.database_service import database_service
from app.chains.voice_chain import voice_chain
from app.config import settings
from app.utils.logger import logger

class VoiceService:
    def __init__(self):
        self.voice_chain = voice_chain
        self.db_service = database_service
        self.elevenlabs_api_key = settings.elevenlabs_api_key

    def _detect_user_emotion(self, user_text: str) -> str:
        """사용자 입력에서 감정 감지"""
        user_text_lower = user_text.lower()
        
        # 슬픈 감정 키워드
        sad_keywords = ["힘들어", "슬퍼", "우울해", "아파", "그리워", "보고싶어", "눈물", "울고", "외로워", "고민", "걱정"]
        if any(keyword in user_text_lower for keyword in sad_keywords):
            return "sad"
        
        # 기쁜 감정 키워드  
        happy_keywords = ["기뻐", "좋아", "행복해", "즐거워", "웃음", "재미있어", "신나", "축하", "성공", "합격"]
        if any(keyword in user_text_lower for keyword in happy_keywords):
            return "happy"
        
        # 화난 감정 키워드
        angry_keywords = ["화나", "짜증", "답답해", "미쳐", "열받아", "억울해", "빡쳐", "스트레스"]
        if any(keyword in user_text_lower for keyword in angry_keywords):
            return "empathetic"  # 화날 때는 공감적으로 대응
        
        # 차분한/안정 키워드
        calm_keywords = ["평온", "차분", "안정", "조용", "편안", "휴식", "명상"]
        if any(keyword in user_text_lower for keyword in calm_keywords):
            return "calm"
        
        return "neutral"

    async def generate_response_and_voice(
        self,
        user_text: str,
        user_info: dict,
        authKeyId: str,
        voice_emotion: str = None  # None이면 자동 감지
    ) -> Dict[str, Any]:
        """텍스트 응답 생성 + 음성 생성까지 한번에"""

        logger.info(f"🎤 [통합 처리] 텍스트+음성 생성 시작")

        # ✅ 1. 고인 정보 조회
        deceased_info = await self.db_service.get_deceased_by_auth_key(authKeyId)
        if not deceased_info:
            raise Exception("고인 정보를 찾을 수 없습니다")

        voice_id = deceased_info.get("voice_id") or getattr(settings, 'default_voice_id', 'DMkRitQrfpiddSQT5adl')

        # ✅ 2. 감정 감지 (voice_emotion이 None이면 자동 감지)
        if voice_emotion is None:
            detected_emotion = self._detect_user_emotion(user_text)
            logger.info(f"🎭 감정 자동 감지: '{user_text}' -> {detected_emotion}")
        else:
            detected_emotion = voice_emotion
            logger.info(f"🎭 감정 수동 설정: {detected_emotion}")

        # ✅ 3. GPT 응답 생성 (감정 반영)
        voice_result = await self.voice_chain.generate_voice_response(
            user_speech_text=user_text,
            user_id=user_info.get("user_id"),
            authKeyId=authKeyId,
            voice_emotion=detected_emotion
        )

        if voice_result["status"] != "success":
            raise Exception("GPT 응답 생성 실패")

        # ✅ 4. 텍스트 → TTS로 변환 (안정적인 설정으로)
        gpt_text = voice_result["voice_response"]
        
        audio_data = b""
        async for audio_chunk in self._stream_elevenlabs_tts_http(gpt_text, voice_id):
            if audio_chunk:
                audio_data += audio_chunk

        logger.info(f"✅ [통합 완료] 텍스트 + TTS 변환 완료 ({len(audio_data)} bytes, emotion: {detected_emotion})")

        return {
            "response_text": gpt_text,
            "audio_mp3": audio_data,
            "voice_analysis": voice_result.get("voice_analysis"),
            "emotion_risk": voice_result.get("emotion_risk", "LOW"),
            "used_memories": voice_result.get("used_memories", []),
            "detected_emotion": detected_emotion,  # 감지된 감정 추가
            "status": "success"
        }

    async def _stream_elevenlabs_tts_http(self, text: str, voice_id: str) -> AsyncGenerator[bytes, None]:
        """HTTP API 방식 TTS - 안정적인 기본 설정만 사용"""
        try:
            logger.info(f"🔊 HTTP TTS 시작: voice_id={voice_id}")

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }

            # 안정적인 기본 설정만 사용
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.8,
                    "similarity_boost": 0.9,
                    "style": 0.5,  # 고정값
                    "use_speaker_boost": True
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        async for chunk in response.content.iter_chunked(8192):
                            if chunk:
                                yield chunk
                        logger.info("✅ HTTP TTS 완료")
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ HTTP TTS 실패: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"❌ HTTP TTS 실패: {str(e)}")