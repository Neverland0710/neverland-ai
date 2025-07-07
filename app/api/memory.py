from fastapi import APIRouter, HTTPException
from typing import Dict
from datetime import datetime
import uuid

from app.models.keepsake import Keepsake
from app.models.photo import PhotoAlbum
from app.models.deceased import Deceased
from app.services.memory_processor_service import MemoryProcessorService
from app.schemas.memory_schemas import ProcessKeepsakeRequest, ProcessPhotoRequest, MemoryProcessResponse
from app.utils.logger import logger

router = APIRouter(tags=["memory"])

@router.post("/keepsake/process", response_model=MemoryProcessResponse)
async def process_keepsake_memory(request: ProcessKeepsakeRequest):
    try:
        logger.info(f" 유품 기억 처리 요청: {request.keepsake_id}")

        keepsake_data = await Keepsake.get_by_id(request.keepsake_id)
        if not keepsake_data:
            raise HTTPException(status_code=404, detail="유품 정보를 찾을 수 없습니다.")

        deceased_info = await Deceased.get_by_auth_key(request.authKeyId)
        if not deceased_info:
            raise HTTPException(status_code=404, detail="고인 정보를 찾을 수 없습니다.")

        memory_result = await memory_processor.convert_to_memory(
            item_data=keepsake_data,
            deceased_info=deceased_info,
            item_type="keepsake",
            authKeyId=request.authKeyId
        )

        return MemoryProcessResponse(
            status="success",
            memory_content=memory_result["memoryText"],
            item_id=memory_result["itemId"],
            item_category="keepsake"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" 유품 기억 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/photo/process", response_model=MemoryProcessResponse)
async def process_photo_memory(request: ProcessPhotoRequest):
    try:
        logger.info(f" 사진 기억 처리 요청: {request.photo_id}")

        photo_data = await PhotoAlbum.get_by_id(request.photo_id)
        if not photo_data:
            raise HTTPException(status_code=404, detail="사진 정보를 찾을 수 없습니다.")

        deceased_info = await Deceased.get_by_auth_key(request.authKeyId)
        if not deceased_info:
            raise HTTPException(status_code=404, detail="고인 정보를 찾을 수 없습니다.")

        memory_result = await memory_processor.convert_to_memory(
            item_data=photo_data,
            deceased_info=deceased_info,
            item_type="photo",
            authKeyId=request.authKeyId
        )

        return MemoryProcessResponse(
            status="success",
            memory_content=memory_result["memoryText"],
            item_id=memory_result["itemId"],
            item_category="photo"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" 사진 기억 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))