# app/services/memory_processor_service.py
"""
기억 변환 서비스 - 순수 변환 로직만 담당
"""

from typing import Dict
from langchain_openai import ChatOpenAI
from app.prompts.memory_prompt import get_keepsake_memory_prompt, get_photo_memory_prompt
from app.utils.logger import logger
from app.config import settings

class MemoryProcessorService:
    """순수 기억 변환만 담당"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=settings.openai_api_key
        )
    
    async def convert_to_memory(
        self, 
        item_data: Dict, 
        deceased_info: Dict, 
        item_type: str
    ) -> str:
        """
        아이템 정보를 자연스러운 기억으로 변환
        
        Args:
            item_data: 유품/사진 데이터
            deceased_info: 고인 정보
            item_type: "keepsake" or "photo"
            
        Returns:
            str: 자연스러운 기억 텍스트
        """
        try:
            # 타입별 프롬프트 선택
            if item_type == "keepsake":
                prompt = get_keepsake_memory_prompt(item_data, deceased_info)
            elif item_type == "photo":
                prompt = get_photo_memory_prompt(item_data, deceased_info)
            else:
                raise ValueError(f"지원하지 않는 아이템 타입: {item_type}")
            
            # LLM 호출
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            
            # 단순 텍스트만 반환
            memory_content = response.content.strip()
            
            logger.info(f" {item_type} 기억 변환 완료")
            return memory_content
            
        except Exception as e:
            logger.error(f" {item_type} 기억 변환 실패: {e}")
            # 기본값 반환
            item_name = item_data.get('item_name') or item_data.get('title', '알 수 없는 아이템')
            return f"{item_name}에 담긴 소중한 기억"

# 전역 인스턴스
memory_processor = MemoryProcessorService()