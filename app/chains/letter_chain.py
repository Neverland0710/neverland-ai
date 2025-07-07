from datetime import datetime
import time
import re
from typing import Tuple, List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.schemas.letter_schemas import LetterProcessInternalResult
from app.services.advanced_rag_service import advanced_rag_service
from app.services.database_service import database_service
from app.utils.logger import logger
from app.prompts.letter_prompt import LetterPrompts
from app.config import settings


class LetterChain:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=settings.openai_api_key
        )
        self.output_parser = StrOutputParser()

    def _parse_summary_and_tags(self, response_text: str) -> Tuple[str, List[str]]:
        """GPT 응답에서 요약과 태그 추출"""
        summary_match = re.search(r"요약\s*:\s*(.*?)\n", response_text, re.DOTALL)
        tags_match = re.search(r"태그\s*:\s*(.*)", response_text)

        summary = summary_match.group(1).strip() if summary_match else response_text.strip()
        tags_str = tags_match.group(1).strip() if tags_match else ""
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        return summary, tags

    async def process_letter(self, letter_id: str, user_id: str, authKeyId: str, letter_text: str) -> LetterProcessInternalResult:
        try:
            start_time = time.time()

            # 1. 고인 정보 조회
            deceased_info = await database_service.get_deceased_by_auth_key(authKeyId)

            # 2. 프롬프트 템플릿 준비
            summary_prompt = PromptTemplate.from_template(LetterPrompts.LETTER_SUMMARY)
            response_prompt = PromptTemplate.from_template(LetterPrompts.LETTER_RESPONSE)

            summary_chain = summary_prompt | self.llm | self.output_parser
            response_chain = response_prompt | self.llm | self.output_parser

            # 3. 답장 생성
            response_input = {
                "title": "",
                "content": letter_text,
                "user_name": deceased_info["user_name"],
                "deceased_name": deceased_info["name"],
                "relation_to_user": deceased_info["relation_to_user"],
                "personality": deceased_info["personality"],
                "speaking_style": deceased_info["speaking_style"],
                "memory_context": "",
            }
            response = await response_chain.ainvoke(response_input)

            # 4. 요약 + 태그 생성
            summary_input = {
                "user_letter": letter_text,
                "ai_response": response,
                "user_name": deceased_info["user_name"],
                "deceased_name": deceased_info["name"],
                "relation_to_user": deceased_info["relation_to_user"]
            }
            summary_raw = await summary_chain.ainvoke(summary_input)
            summary, tags = self._parse_summary_and_tags(summary_raw)

            # 5. Qdrant 저장
            await advanced_rag_service.store_memory(
                content=summary,
                authKeyId=authKeyId,
                memory_type="letter",
                item_id=f"letter_{datetime.utcnow().timestamp()}",
                item_type="letter",
                source="letter",
                tags=tags,
                date=datetime.today().strftime("%Y-%m-%d")
            )

            elapsed = round(time.time() - start_time, 2)

            return LetterProcessInternalResult(
                response=response,
                summary_stored=summary,
                emotion_tone=None,
                tags=tags,
                processing_time=elapsed
            )

        except Exception as e:
            logger.error(f" 편지 처리 실패: {e}")
            return LetterProcessInternalResult(
                response="죄송해요, 답장을 준비하다 문제가 생겼어요."
            )


letter_chain = LetterChain()
