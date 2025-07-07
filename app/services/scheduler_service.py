from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
import uuid

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

            deceased_list = await database_service.get_all_deceased()

            for deceased in deceased_list:
                authKeyId = deceased["auth_key_id"]
                deceasedName = deceased["name"]
                userName = deceased["user_name"]

                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                messages = await database_service.get_conversations_by_date(
                    auth_key_id=authKeyId,
                    date=yesterday
                )
                if not messages:
                    continue

                dialogue = "\n".join([f"{m['sender']}: {m['message']}" for m in messages])

                prompt = DAILY_SUMMARY.format(
                    date=yesterday,
                    user_name=userName,
                    deceased_name=deceasedName,
                    dialogue=dialogue
                )
                response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
                summary = response.content.strip()

                itemId = f"summary_{yesterday}"
                uniqueId = f"{authKeyId}_{itemId}_{uuid.uuid4().hex[:6]}"
                createdAt = datetime.now().isoformat()

                metadata = {
                    "authKeyId": authKeyId,
                    "itemId": itemId,
                    "itemCategory": "daily",
                    "memoryType": "summary",
                    "date": yesterday,
                    "createdAt": createdAt,
                    "source": "daily_summary"
                }

                await advanced_rag_service.store_memory_with_metadata(
                    id=uniqueId,
                    content=summary,
                    page_content=summary,
                    memory_type="daily",
                    **metadata
                )
                logger.info(f" 요약 저장 완료: {authKeyId} / {yesterday}")

            logger.info(" 모든 사용자 요약 완료")

        except Exception as e:
            logger.error(f" 일일 요약 작업 실패: {e}")

scheduler_service = SchedulerService()
