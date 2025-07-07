# app/services/memory_processor_service.py
"""
기억 변환 서비스 - 유품, 사진 전용 변환 + Qdrant 저장
"""

from typing import Dict
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from langchain_openai import ChatOpenAI
from app.prompts.memory_prompt import get_keepsake_memory_prompt, get_photo_memory_prompt
from app.services.advanced_rag_service import advanced_rag_service
from app.utils.logger import logger
from app.config import settings


# 한국 시간대
KST = timezone(timedelta(hours=9))

def now_kst():
    return datetime.now(KST)

def generate_item_id(item_type: str) -> str:
    """itemId = photo_20250707_a1b2c3 형식 생성"""
    return f"{item_type}_{now_kst().strftime('%Y%m%d')}_{uuid4().hex[:6]}"


class MemoryProcessorService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    async def convert_to_memory(
        self,
        itemData: Dict,
        deceasedInfo: Dict,
        itemType: str,
        authKeyId: str
    ) -> Dict:
        """유품 또는 사진 → GPT 요약 → Qdrant 저장"""
        try:
            prompt = self.build_prompt(itemData, deceasedInfo, itemType)
            logger.debug(f"[ MemoryProcessor ] 프롬프트 생성 완료 - type={itemType}")

            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            memoryText = response.content.strip()

            sourceText = itemData.get("description", "")
            itemId = await self.save_summary_to_qdrant(authKeyId, itemType, memoryText, sourceText)

            return {
                "status": "success",
                "itemId": itemId,
                "memoryText": memoryText
            }

        except Exception as e:
            logger.error(f"[ MemoryProcessor ] 변환 및 저장 실패: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def build_prompt(self, itemData: Dict, deceasedInfo: Dict, itemType: str) -> str:
        """항목 유형에 따라 적절한 프롬프트 반환"""
        if itemType == "keepsake":
            return get_keepsake_memory_prompt(itemData, deceasedInfo)
        elif itemType == "photo":
            return get_photo_memory_prompt(itemData, deceasedInfo)
        else:
            raise ValueError(f"지원하지 않는 itemType입니다: {itemType}")

    async def save_summary_to_qdrant(
        self,
        authKeyId: str,
        itemType: str,
        memoryText: str,
        sourceText: str
    ) -> str:
        """Qdrant에 저장"""
        itemId = generate_item_id(itemType)
        createdAt = now_kst().isoformat()

        metadata = {
            "authKeyId": authKeyId,
            "itemId": itemId,
            "itemCategory": itemType,
            "memoryType": "summary",
            "source": f"{itemType}_TB",
            "date": createdAt[:10],
            "createdAt": createdAt,
            "sourceText": sourceText
        }

        advanced_rag_service.upsert_document(
            collection_name="memories",
            document={
                "page_content": memoryText,
                "metadata": metadata
            }
        )

        return itemId


# 전역 인스턴스
memory_processor = MemoryProcessorService()