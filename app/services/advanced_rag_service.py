from datetime import datetime, date
from typing import List, Dict, Optional
import os
import logging
import asyncio
import threading

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore 
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings
from app.utils.logger import logger

logger = logging.getLogger("memorial_chat")

# LangSmith 추적 연동
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

def format_date_relative(memory_date: str) -> str:
    """날짜를 상대적 표현으로 변환"""
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
    """ Singleton 패턴으로 중복 초기화 방지"""
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AdvancedRAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        #  중복 초기화 방지
        if self._initialized:
            logger.debug(" AdvancedRAGService 이미 초기화됨 - 재사용")
            return
            
        with self._lock:
            if self._initialized:
                return
                
            logger.info(" AdvancedRAGService 초기화 시작...")
            
            try:
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=settings.openai_api_key
                )

                self.qdrant_client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key
                )

                #  QdrantVectorStore로 변경 (deprecation 경고 해결)
                self.daily_conversation_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=settings.daily_conversation_collection,
                    embedding=self.embeddings  # embeddings -> embedding으로 변경
                )
                self.letter_memory_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=settings.letter_memory_collection,
                    embedding=self.embeddings
                )
                self.object_memory_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=settings.object_memory_collection,
                    embedding=self.embeddings
                )

                # 성능 최적화 설정
                self.RELEVANCE_THRESHOLD = 0.3
                self.MAX_SEARCH_RESULTS = 3
                self.SEARCH_TIMEOUT = 10.0

                self._initialized = True
                logger.info(" AdvancedRAGService 초기화 완료")
                
            except Exception as e:
                logger.error(f" AdvancedRAGService 초기화 실패: {e}")
                raise

    async def search_memories(self, query: str, authKeyId: str) -> List[Dict]:
        """ 최적화된 메모리 검색"""
        try:
            logger.info(f"🔍 RAG 검색 시작: query='{query}', authKeyId='{authKeyId}'")
            
            # 입력 검증
            if not query or not query.strip():
                logger.warning(" 빈 검색어 - 검색 생략")
                return []
                
            if not authKeyId:
                logger.warning(" authKeyId 없음 - 검색 생략")
                return []

            query = query.strip()
            
            #  병렬 검색 with 필터링
            auth_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.authKeyId",
                        match=MatchValue(value=authKeyId)
                    )
                ]
            )

            search_tasks = []
            
            # 각 컬렉션에서 필터링된 검색 수행
            for store_name, store in [
                ("letter", self.letter_memory_store),
                ("object", self.object_memory_store),
                ("daily", self.daily_conversation_store)
            ]:
                task = asyncio.create_task(
                    self._search_with_filter(store, query, auth_filter, store_name)
                )
                search_tasks.append(task)

            # 타임아웃과 함께 병렬 실행
            try:
                search_results = await asyncio.wait_for(
                    asyncio.gather(*search_tasks, return_exceptions=True),
                    timeout=self.SEARCH_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(f" 검색 타임아웃 ({self.SEARCH_TIMEOUT}초) - 부분 결과 반환")
                search_results = []

            # 결과 통합 및 정리
            all_results = []
            for result in search_results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"⚠️ 검색 오류 무시: {result}")

            logger.info(f" 전체 검색 결과: {len(all_results)}개")

            # 점수 기준 필터링 및 정렬
            filtered_results = [
                r for r in all_results 
                if r.get("score", 0) >= self.RELEVANCE_THRESHOLD
            ]
            
            # 점수 기준 내림차순 정렬
            sorted_results = sorted(
                filtered_results, 
                key=lambda x: x.get("score", 0), 
                reverse=True
            )

            # 최대 개수 제한
            final_results = sorted_results[:self.MAX_SEARCH_RESULTS]
            
            logger.info(f" 최종 검색 결과: {len(final_results)}개")
            for r in final_results:
                logger.info(f"   [{r['collection']}] score: {r['score']:.4f} - {r['content'][:30]}...")

            return final_results

        except Exception as e:
            logger.error(f" 기억 검색 실패: {e}")
            return []

    async def _search_with_filter(
        self, 
        store: QdrantVectorStore, 
        query: str, 
        auth_filter: Filter, 
        collection_name: str
    ) -> List[Dict]:
        """필터링과 함께 개별 컬렉션 검색"""
        try:
            # QdrantVectorStore의 새로운 API 사용
            docs_and_scores = await store.asimilarity_search_with_score(
                query=query,
                k=self.MAX_SEARCH_RESULTS,
                filter=auth_filter  # 필터 적용
            )
            
            results = []
            for doc, score in docs_and_scores:
                meta = doc.metadata or {}
                results.append({
                    "content": doc.page_content,
                    "metadata": meta,
                    "collection": collection_name,
                    "score": score,
                    "date_text": format_date_relative(meta.get("date", ""))
                })
                
            logger.debug(f" {collection_name} 검색: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.warning(f" {collection_name} 검색 실패: {e}")
            return []

    async def store_memory(self, content: str, authKeyId: str, memory_type: str, **kwargs) -> Dict:
        """메모리 저장"""
        try:
            metadata = {
                "authKeyId": authKeyId,
                "memory_type": memory_type,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # 추가 메타데이터 병합
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
        """메타데이터와 함께 메모리 저장"""
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
        """필터 조건으로 메모리 삭제"""
        try:
            store = self._get_store_by_collection(collection_name)
            await store.adelete(filter=filter_condition)
            logger.info(f" Qdrant에서 삭제 완료: {collection_name} (조건: {filter_condition})")
            return 1
        except Exception as e:
            logger.error(f" delete_memories_with_filter 실패: {e}")
            return 0

    def _get_store_by_type(self, memory_type: str) -> QdrantVectorStore:
        """메모리 타입별 저장소 반환"""
        if memory_type == "letter":
            return self.letter_memory_store
        elif memory_type in ["keepsake", "photo"]:
            return self.object_memory_store
        else:
            return self.daily_conversation_store

    def _get_store_by_collection(self, collection_name: str) -> QdrantVectorStore:
        """컬렉션명별 저장소 반환"""
        if collection_name == settings.letter_memory_collection:
            return self.letter_memory_store
        elif collection_name == settings.object_memory_collection:
            return self.object_memory_store
        elif collection_name == settings.daily_conversation_collection:
            return self.daily_conversation_store
        else:
            raise ValueError(f" 지원하지 않는 컬렉션 이름: {collection_name}")

#  Singleton 인스턴스 생성
advanced_rag_service = AdvancedRAGService()