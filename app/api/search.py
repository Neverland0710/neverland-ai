# app/api/search.py
"""
지능형 기억 검색 API
"""

from fastapi import APIRouter, HTTPException

from app.services.advanced_rag_service import advanced_rag_service
from app.schemas.search_schemas import SearchRequest, SearchResponse
from app.utils.logger import logger

router = APIRouter(tags=["search"])


@router.post("/search/memories", response_model=SearchResponse)
async def search_memories(request: SearchRequest):
    """지능형 기억 검색"""
    try:
        logger.info(f" 기억 검색 요청: query='{request.query}'")

        # user_id를 authKeyId로 변환하거나, 스키마에 authKeyId가 있다면 직접 사용
        auth_key_id = getattr(request, 'authKeyId', getattr(request, 'user_id', None))
        
        if not auth_key_id:
            raise HTTPException(status_code=400, detail="authKeyId 또는 user_id가 필요합니다")

        logger.debug(f" 사용할 authKeyId: {auth_key_id}")

        # authKeyId 파라미터로 수정
        memories = await advanced_rag_service.search_memories(
            query=request.query,
            authKeyId=auth_key_id
        )

        logger.info(f" 검색 완료: {len(memories)}개 기억 발견")

        return SearchResponse(
            memories=memories,
            search_strategy="통합 컬렉션 검색 (authKeyId 기반)",
            total_found=len(memories)
        )

    except Exception as e:
        logger.error(f" 기억 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))