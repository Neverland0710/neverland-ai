from fastapi import APIRouter, HTTPException
from app.services.advanced_rag_service import AdvancedRAGService
from app.schemas.commons_schemas import DeleteRequest, DeleteResponse
from app.utils.logger import logger
from app.config import settings

router = APIRouter(tags=["admin"])
rag_service = AdvancedRAGService()

MEMORY_COLLECTION_MAP = {
    "letter": settings.letter_memory_collection,
    "keepsake": settings.object_memory_collection,
    "photo": settings.object_memory_collection
}

@router.post("/admin/memory/delete", response_model=DeleteResponse)
async def delete_memory(request: DeleteRequest):
    try:
        logger.info(f"🗑️ 기억 삭제 요청: {request.itemCategory} {request.itemId}")

        if request.itemCategory == "chat":
            raise HTTPException(status_code=400, detail="chat 유형의 기억은 삭제할 수 없습니다.")

        collection_name = MEMORY_COLLECTION_MAP.get(request.itemCategory)
        if not collection_name:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 itemCategory: {request.itemCategory}")

        delete_filter = {
            "must": [
                {"key": "metadata.authKeyId", "match": {"value": request.authKeyId}},
                {"key": "metadata.itemId", "match": {"value": request.itemId}},
                {"key": "metadata.itemCategory", "match": {"value": request.itemCategory}}
            ]
        }

        deleted_count = await rag_service.delete_memories_with_filter(
            collection_name=collection_name,
            filter_condition=delete_filter
        )

        logger.info(f" 기억 삭제 완료: {request.itemCategory} {request.itemId} ({deleted_count}개)")

        return DeleteResponse(
            status="deleted",
            deleted_from_collections=[collection_name],
            deleted_count=deleted_count,
            message=f"{request.itemCategory} 기억이 삭제되었습니다.",
            item_id=request.itemId,
            item_category=request.itemCategory
        )

    except Exception as e:
        logger.error(f" 기억 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
