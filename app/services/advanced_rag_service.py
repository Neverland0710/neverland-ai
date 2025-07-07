from datetime import datetime, date
from typing import List, Dict
import os
import logging
import asyncio

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient

from app.config import settings
from app.utils.logger import logger

logger = logging.getLogger("memorial_chat")

#  LangSmith 추적 연동
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

def format_date_relative(memory_date: str) -> str:
    try:
        mem_date = datetime.strptime(memory_date, "%Y-%m-%d").date()
        today = date.today()
        delta = (today - mem_date).days

        if delta == 0:
            return "오늘 있었던 일"
        elif delta == 1:
            return "어제 있었던 일"
        elif delta < 7:
            return f"{delta}일 전에 있었던 일"
        elif mem_date.year == today.year:
            return f"{mem_date.month}월 {mem_date.day}일에 있었던 일"
        else:
            return f"{mem_date.year}년 {mem_date.month}월 {mem_date.day}일에 있었던 일"
    except:
        return "날짜 미상"

class AdvancedRAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )

        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

        self.daily_conversation_store = Qdrant(
            client=self.qdrant_client,
            collection_name=settings.daily_conversation_collection,
            embeddings=self.embeddings
        )
        self.letter_memory_store = Qdrant(
            client=self.qdrant_client,
            collection_name=settings.letter_memory_collection,
            embeddings=self.embeddings
        )
        self.object_memory_store = Qdrant(
            client=self.qdrant_client,
            collection_name=settings.object_memory_collection,
            embeddings=self.embeddings
        )

        logger.info(" AdvancedRAGService 초기화 완료")

    async def search_memories(self, query: str, authKeyId: str) -> List[Dict]:
        try:
            logger.info(f"🔍 RAG 검색 시작: query='{query}', authKeyId='{authKeyId}'")
            results = []

            # 병렬 검색 수행
            letter_task = self.letter_memory_store.asimilarity_search_with_score(query, k=3)
            object_task = self.object_memory_store.asimilarity_search_with_score(query, k=3)
            daily_task = self.daily_conversation_store.asimilarity_search_with_score(query, k=3)

            letter_docs, object_docs, daily_docs = await asyncio.gather(
                letter_task, object_task, daily_task
            )
            
            logger.info(f" 검색 결과: letter={len(letter_docs)}, object={len(object_docs)}, daily={len(daily_docs)}")

            for docs, collection in [
                (letter_docs, "letter"),
                (object_docs, "object"),
                (daily_docs, "daily")
            ]:
                for doc, score in docs:
                    meta = doc.metadata or {}
                    results.append({
                        "content": doc.page_content,
                        "metadata": meta,
                        "collection": collection,
                        "score": score,
                        "date_text": format_date_relative(meta.get("date", ""))
                    })

            logger.info(f" 전체 검색 결과: {len(results)}개")
            for r in results:
                logger.info(f"  - {r['collection']} | score: {r['score']:.4f} | authKeyId: {r['metadata'].get('authKeyId', 'NONE')}")

            filtered = [r for r in results if r["metadata"].get("authKeyId") == authKeyId]
            logger.info(f" authKeyId 필터 후: {len(filtered)}개")

            RELEVANCE_THRESHOLD = 0.3  
            relevant = [
                r for r in filtered
                if r["score"] is not None and r["score"] >= RELEVANCE_THRESHOLD
            ]

            logger.info(f"⭐ 임계값 {RELEVANCE_THRESHOLD} 이상: {len(relevant)}개")
            for r in relevant:
                logger.info(f"[{r['collection']} | score: {r['score']:.4f}] {r['date_text']} - {r['content'][:30]}...")

            sorted_relevant = sorted(relevant, key=lambda x: -x["score"])
            return sorted_relevant[:3]

        except Exception as e:
            logger.error(f" 기억 검색 실패: {e}")
            return []

    async def store_memory(self, content: str, authKeyId: str, memory_type: str, **kwargs) -> Dict:
        try:
            metadata = {
                "authKeyId": authKeyId,
                "memory_type": memory_type,
                "created_at": datetime.utcnow().isoformat()
            }
            for key in ["item_id", "item_type", "source", "date", "title", "tags", "collection"]:
                if key in kwargs:
                    metadata[key] = kwargs[key]

            store = self._get_store_by_type(memory_type)
            doc = Document(page_content=content, metadata=metadata)
            await store.aadd_documents([doc])

            logger.info(f" 기억 저장 완료: type={memory_type}")
            return {"status": "stored", "collection": store.collection_name}

        except Exception as e:
            logger.error(f" 기억 저장 실패: {e}")
            return {"status": "failed", "error": str(e)}

    async def store_memory_with_metadata(
        self,
        id: str,
        content: str,
        page_content: str,
        memory_type: str,
        **metadata
    ) -> Dict:
        try:
            metadata.update({
                "id": id,
                "memory_type": memory_type,
                "created_at": datetime.utcnow().isoformat()
            })

            store = self._get_store_by_type(memory_type)
            doc = Document(page_content=page_content, metadata=metadata)
            await store.aadd_documents([doc])

            logger.info(f" store_memory_with_metadata 완료: type={memory_type}")
            return {"status": "stored", "collection": store.collection_name}

        except Exception as e:
            logger.error(f" store_memory_with_metadata 실패: {e}")
            return {"status": "failed", "error": str(e)}

    async def delete_memories_with_filter(self, collection_name: str, filter_condition: Dict) -> int:
        try:
            store = self._get_store_by_collection(collection_name)
            await store.adelete(filter=filter_condition)
            logger.info(f" Qdrant에서 삭제 완료: {collection_name} (조건: {filter_condition})")
            return 1
        except Exception as e:
            logger.error(f" delete_memories_with_filter 실패: {e}")
            return 0

    def _get_store_by_type(self, memory_type: str):
        if memory_type == "letter":
            return self.letter_memory_store
        elif memory_type in ["keepsake", "photo"]:
            return self.object_memory_store
        else:
            return self.daily_conversation_store

    def _get_store_by_collection(self, collection_name: str):
        if collection_name == settings.letter_memory_collection:
            return self.letter_memory_store
        elif collection_name == settings.object_memory_collection:
            return self.object_memory_store
        elif collection_name == settings.daily_conversation_collection:
            return self.daily_conversation_store
        else:
            raise ValueError(f"지원하지 않는 컬렉션 이름: {collection_name}")


advanced_rag_service = AdvancedRAGService()