# app/api/chat.py
"""
실시간 대화 API
"""

from fastapi import APIRouter, HTTPException

from app.chains.chat_chain import chat_chain
from app.schemas.chat_schemas import ChatRequest, ChatResponse
from app.utils.logger import logger

router = APIRouter(tags=["chat"])


@router.post("/chat/generate", response_model=ChatResponse)
async def generate_chat_response(request: ChatRequest):
    """실시간 대화 응답 생성"""
    try:
        logger.info(f" 채팅 요청: user_id={request.user_id}, authKeyId={request.authKeyId}")
        
        # 응답 생성 (고인 정보는 체인 내부에서 조회)
        result = await chat_chain.generate_response(
            user_input=request.user_input,
            user_id=request.user_id,
            authKeyId=request.authKeyId,
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return ChatResponse(
            status="success",
            response=result["response"],
            used_memories=result["used_memories"],
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        logger.error(f" 채팅 API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))