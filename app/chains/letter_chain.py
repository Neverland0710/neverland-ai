from typing import Dict, List, Tuple
from datetime import datetime
import time
import re

from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.utils.logger import logger
from app.services.advanced_rag_service import advanced_rag_service
from app.services.database_service import database_service
from app.schemas.letter_schemas import LetterProcessInternalResult
from app.prompts.letter_prompt import LetterPrompts


class LetterChain:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            openai_api_key=settings.openai_api_key
        )

    async def process_letter(
        self,
        letter_id: str,
        user_id: str,
        authKeyId: str,
        letter_text: str
    ) -> LetterProcessInternalResult:
        try:
            start_time = time.time()

            # 1. 고인 정보 조회
            deceased_info = await database_service.get_user_by_auth_key(authKeyId)

            # 2. 관련 기억 검색
            rag_memories = await advanced_rag_service.search_memories(
                query=letter_text,
                authKeyId=authKeyId
            )

            memory_context = ""
            date_text = ""
            memory_keywords = ""

            if rag_memories:
                memory_context = rag_memories[0]["content"]
                date_text = rag_memories[0].get("date_text", "")
                memory_keywords = ", ".join(rag_memories[0].get("metadata", {}).get("tags", []))

            # 3. 답장 생성
            response_prompt = PromptTemplate.from_template(LetterPrompts.LETTER_RESPONSE)
            response_input = {
                "title": "",
                "content": letter_text,
                "user_name": deceased_info["user_name"],
                "deceased_name": deceased_info["deceased_name"],
                "relation_to_user": deceased_info["relation_to_deceased"],
                "personality": deceased_info["personality"],
                "speaking_style": deceased_info["speaking_style"],
                "memory_context": memory_context,
                "date_text": date_text,
                "memory_keywords": memory_keywords or "기억 속 따뜻한 장면"
            }
            ai_response = await self.llm.ainvoke(response_prompt.format(**response_input))
            response = ai_response.content if isinstance(ai_response, AIMessage) else str(ai_response)

            # 4. 요약 및 태그 생성
            summary_prompt = PromptTemplate.from_template(LetterPrompts.LETTER_SUMMARY)
            summary_input = {
                "user_letter": letter_text,
                "ai_response": response,
                "user_name": deceased_info["user_name"],
                "deceased_name": deceased_info["deceased_name"],
                "relation_to_user": deceased_info["relation_to_deceased"]
            }
            summary_result = await self.llm.ainvoke(summary_prompt.format(**summary_input))
            summary_text = summary_result.content if isinstance(summary_result, AIMessage) else str(summary_result)

            summary, tags = self._parse_summary_and_tags(summary_text)
            vector_text = self._build_vector_text(summary, tags)

            # 5. Qdrant 저장
            item_id = f"letter_{datetime.utcnow().timestamp()}"
            await advanced_rag_service.store_memory_with_metadata(
                id=item_id,
                content=summary,  
                page_content=summary,
                memory_type="letter",
                authKeyId=authKeyId,
                item_type="letter",
                itemCategory="letter",
                memoryType="summary",
                source="letter",
                tags=tags,
                sourceText=letter_text,
                date=datetime.today().strftime("%Y-%m-%d"),
                createdAt=datetime.utcnow().isoformat(),
                vector_override=vector_text  
            )

            return LetterProcessInternalResult(
                response=response,
                summary_stored=summary,
                emotion_tone=None,
                tags=tags,
                processing_time=round(time.time() - start_time, 2)
            )

        except Exception as e:
            logger.error(f" 편지 처리 실패: {e}")
            return LetterProcessInternalResult(
                response="죄송해요, 답장을 준비하다 문제가 생겼어요."
            )

    def _parse_summary_and_tags(self, response_text: str) -> Tuple[str, List[str]]:
        """
        "요약: ..."과 "태그: ..." 구조에서 요약/태그 분리
        """
        lines = response_text.strip().splitlines()
        memory_lines = []
        tags = []

        for line in lines:
            if line.strip().lower().startswith("태그:"):
                tag_line = line.split(":", 1)[-1]
                tags = [t.strip() for t in tag_line.split(",") if t.strip()]
            elif line.strip().lower().startswith("요약:"):
                content = line.split(":", 1)[-1].strip()
                memory_lines.append(content)
            else:
                memory_lines.append(line.strip())

        summary = " ".join(memory_lines)
        return summary, tags

    def _build_vector_text(self, summary: str, tags: List[str]) -> str:
        """
        벡터 검색용 텍스트: 태그 먼저 → 요약
        """
        return f"[태그: {', '.join(tags)}]\n{summary}" if tags else summary


# 인스턴스
letter_chain = LetterChain()
