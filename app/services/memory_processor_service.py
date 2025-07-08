from typing import Tuple, List, Dict
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
            full_text = response.content.strip()

            memory_text, tags = self._parse_summary_and_tags(full_text)

            source_text = itemData.get("description", "")
            item_id = await self.save_summary_to_qdrant(
                authKeyId, itemType, memory_text, source_text, tags
            )

            return {
                "status": "success",
                "itemId": item_id,
                "memoryText": memory_text,
                "tags": tags
            }

        except Exception as e:
            logger.error(f"[ MemoryProcessor ] 변환 및 저장 실패: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def build_prompt(self, itemData: Dict, deceasedInfo: Dict, itemType: str) -> str:
        if itemType == "keepsake":
            return get_keepsake_memory_prompt(itemData, deceasedInfo)
        elif itemType == "photo":
            return get_photo_memory_prompt(itemData, deceasedInfo)
        else:
            raise ValueError(f"지원하지 않는 itemType입니다: {itemType}")

    def _parse_summary_and_tags(self, response_text: str) -> Tuple[str, List[str]]:
        """응답에서 요약과 태그 추출"""
        lines = response_text.strip().splitlines()
        memory_lines = []
        tags = []

        for line in lines:
            if line.strip().lower().startswith("태그:"):
                tag_line = line.split(":", 1)[-1]
                tags = [t.strip() for t in tag_line.split(",") if t.strip()]
            else:
                memory_lines.append(line.strip())

        return " ".join(memory_lines), tags

    async def save_summary_to_qdrant(
        self,
        authKeyId: str,
        itemType: str,
        memoryText: str,
        sourceText: str,
        tags: list
    ) -> str:
        itemId = generate_item_id(itemType)
        createdAt = now_kst().isoformat()

        #  태그 정제
        clean_tags = []
        if isinstance(tags, dict):
            clean_tags = list(tags.values())
        elif isinstance(tags, str):
            clean_tags = [t.strip() for t in tags.strip("[]").split(",") if t.strip()]
        elif isinstance(tags, list):
            clean_tags = [str(t).strip() for t in tags if isinstance(t, (str, int))]

        #  page_content에는 태그 넣지 않음
        #  태그는 벡터화용 텍스트에만 포함시킴
        vector_text = f"[태그: {', '.join(clean_tags)}]\n{memoryText}" if clean_tags else memoryText

        metadata = {
            "authKeyId": authKeyId,
            "itemId": itemId,
            "itemCategory": itemType,
            "memoryType": "summary",
            "source": f"{itemType}_TB",
            "date": createdAt[:10],
            "createdAt": createdAt,
            "sourceText": sourceText,
            "tags": clean_tags
        }

        await advanced_rag_service.upsert_document(
            collection_name="memories",
            document={
                "page_content": memoryText,  
                "metadata": metadata
            },
            vector_override=vector_text 
        )

        return itemId