# app/api/letter.py

from fastapi import APIRouter, HTTPException
from app.schemas.letter_schemas import LetterRequest, LetterProcessResponse
from app.chains.letter_chain import letter_chain
from app.utils.logger import logger

router = APIRouter(tags=["letter"])


@router.post("/letter/process", response_model=LetterProcessResponse)
async def process_letter(request: LetterRequest):
    """AI 편지 답장 생성 + 요약 저장"""
    try:
        result = await letter_chain.process_letter(
            letter_id=request.letter_id,
            user_id=request.user_id,
            authKeyId=request.authKeyId,
            letter_text=request.letter_text
        )

        return LetterProcessResponse(
            status="success",
            response=result.response
        )

    except Exception as e:
        logger.error(f"❌ 편지 처리 API 오류: {e}")
        raise HTTPException(status_code=500, detail="편지 처리 중 오류가 발생했습니다.")