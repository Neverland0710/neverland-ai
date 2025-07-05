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

        # memory_type 제거 → 모든 컬렉션에서 검색
        memories = await advanced_rag_service.search_memories(
            query=request.query,
            user_id=request.user_id
        )

        return SearchResponse(
            memories=memories,
            search_strategy="통합 컬렉션 검색",
            total_found=len(memories)
        )

    except Exception as e:
        logger.error(f" 기억 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))