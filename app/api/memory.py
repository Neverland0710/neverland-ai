# app/api/memory.py
"""
기억 생성 API - 전체 플로우 관리
"""

from fastapi import APIRouter, HTTPException
from typing import Dict
from datetime import datetime
import uuid

from app.models.keepsake import Keepsake
from app.models.photo import PhotoAlbum
from app.models.deceased import Deceased
from app.services.memory_processor_service import memory_processor
from app.services.advanced_rag_service import AdvancedRAGService
from app.schemas.memory_schemas import ProcessKeepsakeRequest, ProcessPhotoRequest, MemoryProcessResponse
from app.utils.logger import logger
from app.config import settings

router = APIRouter(tags=["memory"])

# 서비스 인스턴스
rag_service = AdvancedRAGService()

@router.post("/keepsake/process", response_model=MemoryProcessResponse)
async def process_keepsake_memory(request: ProcessKeepsakeRequest):
    """유품을 기억 형태로 변환하여 Qdrant에 저장"""
    try:
        logger.info(f" 유품 기억 처리 요청: {request.keepsake_id}")
        
        # 1. DB에서 유품 정보 조회
        keepsake_data = await Keepsake.get_by_id(request.keepsake_id)
        if not keepsake_data:
            raise HTTPException(status_code=404, detail="유품 정보를 찾을 수 없습니다.")
        
        # 2. 고인 정보 조회
        deceased_info = await Deceased.get_by_auth_key(request.authKeyId)
        if not deceased_info:
            raise HTTPException(status_code=404, detail="고인 정보를 찾을 수 없습니다.")
        
        # 3. 기억 변환
        memory_content = await memory_processor.convert_to_memory(
            item_data=keepsake_data,
            deceased_info=deceased_info,
            item_type="keepsake"
        )
        
        # 4. 고유 ID 및 메타데이터 생성
        unique_id = f"{request.authKeyId}_keepsake_{request.keepsake_id}_{uuid.uuid4()}"
        metadata = {
            "authKeyId": request.authKeyId,
            "item_id": request.keepsake_id,
            "item_category": "keepsake",
            "type": "keepsake",
            "date": keepsake_data.get("created_at", datetime.now().strftime("%Y-%m-%d")),
            "created_at": datetime.now().isoformat(),
            "source": "db_processed"
        }
        
        # 5. Qdrant에 저장
        await rag_service.store_memory_with_metadata(
            id=unique_id,
            content=memory_content,
            page_content=memory_content,
            memory_type="keepsake",
            **metadata
        )
        
        logger.info(f" 유품 기억 처리 완료: {request.keepsake_id}")
        return MemoryProcessResponse(
            status="success",
            memory_content=memory_content,
            item_id=request.keepsake_id,
            item_category="keepsake"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" 유품 기억 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/photo/process", response_model=MemoryProcessResponse)
async def process_photo_memory(request: ProcessPhotoRequest):
    """사진을 기억 형태로 변환하여 Qdrant에 저장"""
    try:
        logger.info(f" 사진 기억 처리 요청: {request.photo_id}")
        
        # 1. DB에서 사진 정보 조회
        photo_data = await PhotoAlbum.get_by_id(request.photo_id)
        if not photo_data:
            raise HTTPException(status_code=404, detail="사진 정보를 찾을 수 없습니다.")
        
        # 2. 고인 정보 조회
        deceased_info = await Deceased.get_by_auth_key(request.authKeyId)
        if not deceased_info:
            raise HTTPException(status_code=404, detail="고인 정보를 찾을 수 없습니다.")
        
        # 3. 기억 변환
        memory_content = await memory_processor.convert_to_memory(
            item_data=photo_data,
            deceased_info=deceased_info,
            item_type="photo"
        )
        
        # 4. 고유 ID 및 메타데이터 생성
        unique_id = f"{request.authKeyId}_photo_{request.photo_id}_{uuid.uuid4()}"
        metadata = {
            "authKeyId": request.authKeyId,
            "item_id": request.photo_id,
            "item_category": "photo",
            "type": "photo",
            "date": photo_data.get("photo_date") or photo_data.get("uploaded_at", datetime.now().strftime("%Y-%m-%d")),
            "created_at": datetime.now().isoformat(),
            "source": "db_processed"
        }
        
        # 5. Qdrant에 저장
        await rag_service.store_memory_with_metadata(
            id=unique_id,
            content=memory_content,
            page_content=memory_content,
            memory_type="photo",
            **metadata
        )
                
        logger.info(f" 사진 기억 처리 완료: {request.photo_id}")
        return MemoryProcessResponse(
            status="success",
            memory_content=memory_content,
            item_id=request.photo_id,
            item_category="photo"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" 사진 기억 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))