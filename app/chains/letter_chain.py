# app/chains/letter_chain.py

from datetime import datetime
import time

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

    async def process_letter(self, letter_id: str, user_id: str, authKeyId: str, letter_text: str) -> LetterProcessInternalResult:
        try:
            start_time = time.time()

            deceased_info = await database_service.get_deceased_by_auth_key(authKeyId)

            summary_prompt = PromptTemplate.from_template(LetterPrompts.LETTER_SUMMARY)
            response_prompt = PromptTemplate.from_template(LetterPrompts.LETTER_RESPONSE)

            summary_chain = summary_prompt | self.llm | self.output_parser
            response_chain = response_prompt | self.llm | self.output_parser

            response_input = {
                "title": "",
                "content": letter_text,
                "deceased_name": deceased_info["name"],
                "relation_to_user": deceased_info["relation_to_user"],
                "personality": deceased_info["personality"],
                "speaking_style": deceased_info["speaking_style"],
                "memory_context": ""
            }
            response = await response_chain.ainvoke(response_input)

            summary_input = {
                "user_letter": letter_text,
                "ai_response": response,
                "deceased_name": deceased_info["name"],
                "relation_to_user": deceased_info["relation_to_user"]
            }
            summary = await summary_chain.ainvoke(summary_input)

            store_result = await advanced_rag_service.store_memory(
                content=summary,
                authKeyId=authKeyId,
                memory_type="letter",
                item_id=f"letter_{datetime.utcnow().timestamp()}",
                item_type="letter",
                source="letter",
                tags=["letter", "요약"],
                date=datetime.today().strftime("%Y-%m-%d")
            )

            elapsed = round(time.time() - start_time, 2)

            return LetterProcessInternalResult(
                response=response,
                summary_stored=summary,
                emotion_tone=None,
                tags=["letter"],
                processing_time=elapsed
            )

        except Exception as e:
            logger.error(f" 편지 처리 실패: {e}")
            return LetterProcessInternalResult(
                response="죄송해요, 답장을 준비하다 문제가 생겼어요."
            )

letter_chain = LetterChain()
