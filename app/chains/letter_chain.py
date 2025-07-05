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
            temperature=0.5,
            openai_api_key=settings.openai_api_key
        )
        self.output_parser = StrOutputParser()

    async def process_letter(self,letter_id: str, user_id: str, authKeyId: str, letter_text: str) -> LetterProcessInternalResult:
        try:
            start_time = time.time()

            logger.debug(f"[ process_letter 시작] user_id={user_id}, authKeyId={authKeyId}")
            logger.debug(f"[ 입력 편지 내용] {letter_text[:100]}...")

            #  고인 정보 조회 (DB)
            deceased_info = await database_service.get_deceased_by_auth_key(authKeyId)
            logger.debug(f"[🧑‍💼 고인 정보 조회 완료] {deceased_info}")

            #  프롬프트 준비
            summary_prompt = PromptTemplate.from_template(LetterPrompts.LETTER_SUMMARY)
            response_prompt = PromptTemplate.from_template(LetterPrompts.LETTER_RESPONSE)

            summary_chain = summary_prompt | self.llm | self.output_parser
            response_chain = response_prompt | self.llm | self.output_parser

            #  AI 답장 생성 (저장 안 함)
            response_input = {
                "title": "",
                "content": letter_text,
                "deceased_name": deceased_info["name"],
                "relation_to_user": deceased_info["relation_to_user"],
                "personality": deceased_info["personality"],
                "speaking_style": deceased_info["speaking_style"],
                "memory_context": ""
            }
            logger.debug(f"[ response_chain 입력값] {response_input}")
            response = await response_chain.ainvoke(response_input)

            #  요약 생성 후 RAG에 저장
            summary_input = {
                "user_letter": letter_text,
                "ai_response": response,
                "deceased_name": deceased_info["name"],
                "relation_to_user": deceased_info["relation_to_user"]
            }
            logger.debug(f"[ summary_chain 입력값] {summary_input}")
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
            logger.info(f"[ RAG 저장 결과] {store_result}")

            elapsed = round(time.time() - start_time, 2)

            logger.debug(f"[ 처리 완료] elapsed={elapsed}s")

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


# 전역 인스턴스
letter_chain = LetterChain()