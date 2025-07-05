from fastapi import APIRouter, HTTPException
from app.services.advanced_rag_service import AdvancedRAGService
from app.schemas.commons_schemas import DeleteRequest, DeleteResponse
from app.utils.logger import logger
from app.config import settings

router = APIRouter(tags=["admin"])
rag_service = AdvancedRAGService()

# memory_type → Qdrant 컬렉션 매핑 (chat은 제외)
MEMORY_COLLECTION_MAP = {
    "letter": settings.letter_memory_collection,
    "keepsake": settings.object_memory_collection,
    "photo": settings.object_memory_collection
}

# DELETE → POST로 변경 (WebClient body 인식 오류 대응)
@router.delete("/admin/memory/delete", response_model=DeleteResponse)
async def delete_memory(request: DeleteRequest):
    """편지/유품/사진 기억 삭제 (chat은 제외)"""
    try:
        logger.info(f"🗑️ 기억 삭제 요청: {request.item_type} {request.item_id}")

        # chat은 삭제 금지
        if request.item_type == "chat":
            raise HTTPException(status_code=400, detail="chat 유형의 기억은 삭제할 수 없습니다.")

        # 컬렉션 이름 매핑
        collection_name = MEMORY_COLLECTION_MAP.get(request.item_type)
        if not collection_name:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 item_type: {request.item_type}")

        # 필터 조건 구성 (metadata 기준)
        delete_filter = {
            "must": [
                {"key": "metadata.authKeyId", "match": {"value": request.authKeyId}},
                {"key": "metadata.item_id", "match": {"value": request.item_id}},
                {"key": "metadata.item_type", "match": {"value": request.item_type}}
            ]
        }

        # Qdrant 삭제 실행
        deleted_count = await rag_service.delete_memories_with_filter(
            collection_name=collection_name,
            filter_condition=delete_filter
        )

        logger.info(f" 기억 삭제 완료: {request.item_type} {request.item_id} ({deleted_count}개)")

        return DeleteResponse(
            status="deleted",
            deleted_from_collections=[collection_name],
            deleted_count=deleted_count,
            message=f"{request.item_type} 기억이 삭제되었습니다."
        )

    except Exception as e:
        logger.error(f" 기억 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))