# app/services/scheduler_service.py
"""
스케줄러 서비스 - 일일 요약만 담당
"""
import uuid
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta

from app.utils.logger import logger
from app.services.database_service import database_service
from app.services.advanced_rag_service import advanced_rag_service
from app.prompts.summary_prompt import DAILY_SUMMARY
from langchain_openai import ChatOpenAI


class SchedulerService:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        logger.info(" SchedulerService 초기화 완료")

    def start(self):
        try:
            self.scheduler.add_job(
                func=self.daily_summary_job,
                trigger=CronTrigger(hour=4, minute=0),
                id='daily_summary'
            )
            self.scheduler.start()
            logger.info(" 스케줄러 시작 - 일일 요약 작업 등록")
        except Exception as e:
            logger.error(f" 스케줄러 시작 실패: {e}")

    def stop(self):
        try:
            self.scheduler.shutdown()
            logger.info(" 스케줄러 종료")
        except Exception as e:
            logger.error(f" 스케줄러 종료 실패: {e}")

    async def daily_summary_job(self):
        try:
            logger.info(" 일일 요약 작업 시작")

            # 1. 전체 사용자 고인 리스트 불러오기
            deceased_list = await database_service.get_all_deceased()

            for deceased in deceased_list:
                auth_key_id = deceased["auth_key_id"]
                user_id = deceased["user_id"]
                user_name = deceased["user_name"]
                deceased_name = deceased["name"]

                # 2. 전날 날짜 계산
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                
                # 3. 전날 대화 불러오기
                messages = await database_service.get_conversations_by_date(
                    auth_key_id=auth_key_id,
                    date=yesterday
                )
                if not messages:
                    continue

                # 4. 대화 텍스트 구성
                dialogue = "\n".join([f"{m['sender']}: {m['message']}" for m in messages])

                # 5. 프롬프트 생성 및 LLM 호출
                prompt = DAILY_SUMMARY.format(
                    date=yesterday,
                    user_name=user_name,
                    deceased_name=deceased_name,
                    dialogue=dialogue
                )
                response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
                summary = response.content.strip()

                # 6. Qdrant 저장
                unique_id = f"{user_id}_summary_{yesterday}_{uuid.uuid4()}"
                metadata = {
                    "user_id": user_id,
                    "auth_key_id": auth_key_id,
                    "item_id": f"summary_{yesterday}",
                    "item_category": "daily",
                    "type": "daily",
                    "date": yesterday,
                    "source": "daily_summary",
                    "created_at": datetime.now().isoformat()
                }

                await advanced_rag_service.store_memory_with_metadata(
                    id=unique_id,
                    content=summary,
                    page_content=summary,
                    user_id=user_id,
                    memory_type="daily",
                    **metadata
                )
                logger.info(f" 요약 저장 완료: {user_id} / {yesterday}")

            logger.info(" 모든 사용자 요약 완료")

        except Exception as e:
            logger.error(f" 일일 요약 작업 실패: {e}")


# 전역 인스턴스
scheduler_service = SchedulerService()