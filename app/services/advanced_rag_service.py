from datetime import datetime, date
from typing import List, Dict
import os
import logging

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant as LangchainQdrant
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from app.config import settings
from app.utils.logger import logger

logger = logging.getLogger("memorial_chat")

# LangSmith 연동
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
        self.daily_conversation_store = LangchainQdrant(
            client=self.qdrant_client,
            collection_name=settings.daily_conversation_collection,
            embeddings=self.embeddings
        )
        self.letter_memory_store = LangchainQdrant(
            client=self.qdrant_client,
            collection_name=settings.letter_memory_collection,
            embeddings=self.embeddings
        )
        self.object_memory_store = LangchainQdrant(
            client=self.qdrant_client,
            collection_name=settings.object_memory_collection,
            embeddings=self.embeddings
        )
        self.collections = {
            "daily": settings.daily_conversation_collection,
            "letter": settings.letter_memory_collection,
            "object": settings.object_memory_collection
        }
        logger.info(" AdvancedRAGService 초기화 완료")

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
        raise ValueError(f" 지원하지 않는 컬렉션 이름: {collection_name}")

    async def search_memories(self, query: str, authKeyId: str) -> List[Dict]:
        try:
            logger.info(f" RAG 검색 시작: query='{query}', authKeyId='{authKeyId}'")
            query_vector = self.embeddings.embed_query(query)
            all_results = []
            TOP_K = 3
            RELEVANCE_THRESHOLD = 0.3

            def boost_score_with_tags(result, query: str) -> float:
                tags = result.get("metadata", {}).get("tags", [])
                score = result.get("score", 0.0)

                for tag in tags:
                    if tag in query:
                        return score + 0.1
                    if len(tag) >= 2 and tag[:2] in query:
                        return score + 0.05

                return score

            for mem_type, collection in self.collections.items():
                search_result = self.qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=15,
                    search_params=SearchParams(hnsw_ef=64),
                    with_payload=True
                )
                logger.info(f" {collection} 검색 결과: {len(search_result)}개")

                filtered = []
                for r in search_result:
                    meta = r.payload or {}
                    if meta.get("authKeyId") != authKeyId or r.score < RELEVANCE_THRESHOLD:
                        continue
                    item = {
                        "content": meta.get("page_content", ""),
                        "metadata": meta,
                        "collection": mem_type,
                        "score": r.score,
                        "date_text": format_date_relative(meta.get("date", ""))
                    }
                    item["boosted_score"] = boost_score_with_tags(item, query)
                    filtered.append(item)

                top_k = sorted(filtered, key=lambda x: -x["boosted_score"])[:TOP_K]
                for r in top_k:
                    logger.info(f"[{r['collection']}] {r['metadata'].get('tags', [])} | {r['score']:.4f} → {r['boosted_score']:.4f}")
                all_results.extend(top_k)

            sorted_results = sorted(all_results, key=lambda x: -x["boosted_score"])
            return sorted_results[:1]  # 최종적으로 가장 높은 1개만 반환

        except Exception as e:
            logger.error(f" 기억 검색 실패: {e}")
            return []

    async def store_memory(self, content: str, authKeyId: str, memory_type: str, **kwargs) -> Dict:
        """간단한 기억 저장용 (텍스트만 저장 시)"""
        try:
            metadata = {
                "authKeyId": authKeyId,
                "memory_type": memory_type,
                "created_at": datetime.utcnow().isoformat()
            }
            for key in ["item_id", "item_type", "source", "date", "title", "tags"]:
                if key in kwargs:
                    metadata[key] = kwargs[key]

            store = self._get_store_by_type(memory_type)
            doc = Document(page_content=content, metadata=metadata)
            store.add_documents([doc])

            logger.info(f" 기억 저장 완료: type={memory_type}")
            return {"status": "stored", "collection": store.collection_name}
        except Exception as e:
            logger.error(f" 기억 저장 실패: {e}")
            return {"status": "failed", "error": str(e)}

    async def store_memory_with_metadata(self, id: str, content: str, page_content: str, memory_type: str, **metadata) -> Dict:
        """ID를 포함한 전체 메타데이터 기억 저장 (주로 이미지/유품 등)"""
        try:
            metadata.update({
                "id": id,
                "memory_type": memory_type,
                "created_at": datetime.utcnow().isoformat()
            })
            store = self._get_store_by_type(memory_type)
            doc = Document(page_content=page_content, metadata=metadata)
            store.add_documents([doc])

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


# 전역 인스턴스
advanced_rag_service = AdvancedRAGService()
