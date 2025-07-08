from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import asyncio
import re

from langchain_openai import ChatOpenAI
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from app.utils.logger import logger
from app.services.advanced_rag_service import advanced_rag_service
from app.services.database_service import database_service
from app.prompts.chat_prompt import ChatPrompts

# LangSmith 설정
try:
    from langsmith import traceable
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
except ImportError:
    logger.warning(" LangSmith 패키지가 설치되지 않음 - 추적 기능 비활성화")
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator


class DatabaseChatMessageHistory(BaseChatMessageHistory):
    """데이터베이스 기반 채팅 히스토리 관리"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._messages = []
        self._loaded = False

    async def _load_messages(self):
        """데이터베이스에서 대화 히스토리 로드"""
        if self._loaded:
            return
            
        try:
            conversations = await database_service.get_recent_conversations(
                self.session_id, 
                limit=10
            )
            
            for conv in conversations:
                if conv["sender"] == "USER":
                    self._messages.append(HumanMessage(content=conv["message"]))
                else:
                    self._messages.append(AIMessage(content=conv["message"]))
                    
            self._loaded = True
            logger.debug(f" 히스토리 로드 완료: {len(self._messages)}개 메시지")
            
        except Exception as e:
            logger.error(f" 히스토리 로드 실패: {e}")
            self._loaded = True

    @property
    def messages(self):
        return self._messages

    def add_message(self, message):
        self._messages.append(message)

    def clear(self):
        self._messages.clear()


class ResponseParser:
    """GPT 응답 파싱 및 포맷팅"""
    
    MAX_RESPONSE_LENGTH = 300
    METADATA_PREFIXES = ('요약:', '위험도:', '분석:')
    
    def __call__(self, text: Any) -> Dict[str, Any]:
        if isinstance(text, AIMessage):
            text = text.content
        elif not isinstance(text, str):
            text = str(text)

        logger.debug(f" GPT 원본 응답: {text[:100]}...")
        
        response = self._extract_response(text)
        analysis = self._extract_analysis(text)
        risk = self._extract_risk(text)

        # 응답 길이 제한
        if len(response) > self.MAX_RESPONSE_LENGTH:
            response = response[:self.MAX_RESPONSE_LENGTH - 3] + "..."

        logger.info(f" 파싱 완료 - 응답: {response[:50]}... | 분석: {analysis[:30]}...")

        return {
            "output": {
                "response": response,
                "analysis": analysis,
                "risk": risk
            }
        }
    
    def _extract_response(self, text: str) -> str:
        """[대답]: 포함 응답 파싱 (GPT 출력 형식 대응)"""
        lines = text.strip().split('\n')

        # 0. [대답]: 키워드 기반 우선 파싱
        for line in lines:
            if line.strip().startswith("[대답]:"):
                return line.strip().replace("[대답]:", "").strip().strip('"')

        # 1. 일반 메타데이터 필터 기반
        for line in lines:
            line = line.strip()
            if line and not line.startswith(self.METADATA_PREFIXES):
                if line.startswith('"') and line.endswith('"'):
                    return line[1:-1]
                return line

        # 2. | 구분자 기반
        if "|" in text:
            try:
                response = text.strip().split("|")[0].strip()
                response = response.lstrip("응답 내용:").strip().strip("'\"")
                if response:
                    return response
            except Exception as e:
                logger.warning(f" '|' 파싱 실패: {e}")

        # 3. "응답 내용:" 키워드
        for line in lines:
            if "응답 내용:" in line:
                response = line.split("응답 내용:", 1)[1].strip().strip("'\"")
                if response:
                    return response

        # 4. 첫 문장 fallback
        first_sentence = text.split('.')[0].strip()
        if first_sentence and len(first_sentence) > 5:
            return first_sentence

        # 5. 전체 사용 fallback
        if text.strip() and len(text.strip()) > 10:
            return text.strip()

        # 6. 마지막 fallback
        logger.warning(" GPT 응답이 비어있거나 너무 짧음 - 기본 메시지 사용")
        return "안녕하세요! 무엇을 도와드릴까요?"
    
    def _extract_analysis(self, text: str) -> str:
        """분위기 분석 추출"""
        try:
            if "요약:" in text:
                return text.split("요약:")[1].split("\n")[0].strip()
            elif "분위기 분석 요약:" in text:
                return text.split("분위기 분석 요약:")[1].split("\n")[0].strip()
            elif "|" in text and len(text.split("|")) > 1:
                return text.split("|")[1].strip()
        except Exception:
            pass
        return ""
    
    def _extract_risk(self, text: str) -> str:
        """위험도 추출"""
        try:
            if "위험도:" in text:
                return text.split("위험도:")[1].strip().upper()
            elif "|" in text and len(text.split("|")) > 2:
                return text.split("|")[2].strip().replace("위험도:", "").strip().upper()
        except Exception:
            pass
        return "LOW"


class MemorySearchStrategy:
    """메모리 검색 전략 관리"""
    
    SKIP_KEYWORDS = {"응", "그래", "알겠어", "고마워", "ㅎㅎ", "ㅋㅋ", "잘자", "하하", "헐", "음", "으응", "응응", "어"}
    SIMILARITY_THRESHOLD = 0.6
    
    @classmethod
    def should_skip_search(cls, user_input: str, history: DatabaseChatMessageHistory) -> bool:
        """메모리 검색을 생략할지 결정"""
        cleaned = user_input.strip().lower()
        
        # 1. 너무 짧거나 무의미한 발화
        if len(cleaned) <= 2 or cleaned in cls.SKIP_KEYWORDS:
            logger.info(" 단순 응답 감지 → 기억 검색 생략")
            return True
        
        # 2. 최근 AI 응답과 중복도가 높은 경우
        recent_ai_msgs = [
            m.content.lower() for m in reversed(history.messages[-50:]) 
            if isinstance(m, AIMessage)
        ]
        
        user_keywords = set(cleaned.split())
        for msg in recent_ai_msgs:
            msg_words = set(msg.split())
            overlap = user_keywords & msg_words
            if len(overlap) / max(len(user_keywords), 1) >= cls.SIMILARITY_THRESHOLD:
                logger.info(" 최근 응답과 유사한 내용 발견 → 기억 검색 생략")
                return True
        
        return False


class ChatChain:
    """AI 추모 대화 체인 메인 클래스"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=settings.openai_api_key
        )
        
        self.response_parser = ResponseParser()
        self.base_chain = self._build_chain()
        self.chain_with_history = RunnableWithMessageHistory(
            runnable=self.base_chain,
            get_session_history=self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        self.session_histories = {}
        
        logger.info(" ChatChain 초기화 완료")

    def _build_chain(self) -> Runnable:
        """LangChain 체인 구성"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", ChatPrompts.RESPONSE_GENERATION)
        ])

        return (
            RunnablePassthrough.assign(
                memories=RunnableLambda(self._search_memories),
                deceased_info=RunnableLambda(self._get_deceased_info)
            )
            | RunnablePassthrough.assign(
                memory_context=RunnableLambda(self._format_memories),
                **self._create_context_variables()
            )
            | prompt_template
            | self.llm
            | self.response_parser
        )
    
    def _create_context_variables(self) -> Dict:
        """컨텍스트 변수 생성"""
        return {
            "deceased_name": lambda x: x["deceased_info"].get("name", ""),
            "deceased_nickname": lambda x: x["deceased_info"].get("nickname", ""),
            "personality": lambda x: x["deceased_info"].get("personality", ""),
            "speaking_style": lambda x: x["deceased_info"].get("speaking_style", ""),
            "hobbies": lambda x: x["deceased_info"].get("hobbies", ""),
            "age": lambda x: x["deceased_info"].get("age", ""),
            "user_name": lambda x: x["deceased_info"].get("user_name", ""),
            "relation_to_user": lambda x: x["deceased_info"].get("relation_to_user", ""),
            "conversation_history": lambda x: self._get_recent_messages(
                self._get_session_history(x["authKeyId"])
            ),
            "date_text": lambda x: self._extract_date_text(x.get("memories", [])),
            "emotion_tone": lambda x: x.get("previous_analysis", "")
        }

    def _get_session_history(self, session_id: str) -> DatabaseChatMessageHistory:
        """세션 히스토리 관리"""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = DatabaseChatMessageHistory(session_id)
        return self.session_histories[session_id]

    def _get_recent_messages(self, history: DatabaseChatMessageHistory, limit: int = 10) -> str:
        """최근 대화 메시지 포맷팅"""
        try:
            messages = history.messages[-limit:] if history else []
            formatted = []
            
            for m in messages:
                if isinstance(m, HumanMessage):
                    formatted.append(f"사용자: {m.content}")
                elif isinstance(m, AIMessage):
                    formatted.append(f"AI: {m.content}")
                    
            return "\n".join(formatted) if formatted else "(최근 대화 없음)"
        except Exception as e:
            logger.warning(f" 메시지 포맷팅 실패: {e}")
            return "(최근 대화 없음)"

    def _extract_date_text(self, memories: List[Dict]) -> str:
        """메모리에서 날짜 텍스트 추출"""
        if not memories:
            return "예전 어느 날"
        return memories[0].get("date_text", "한참 전")

    def _get_last_analysis(self, session_id: str) -> str:
        """마지막 감정 분석 결과 가져오기"""
        try:
            history = self._get_session_history(session_id)
            if not history or not history.messages:
                return ""
                
            for msg in reversed(history.messages):
                if isinstance(msg, AIMessage):
                    parsed = self.response_parser(msg.content)
                    return parsed["output"].get("analysis", "")
            return ""
        except Exception as e:
            logger.warning(f" 마지막 분석 추출 실패: {e}")
            return ""

    @traceable(name="generate_response")
    async def generate_response(self, user_input: str, user_id: str, authKeyId: str) -> Dict:
        """메인 응답 생성 함수"""
        try:
            # 세션 히스토리 로드
            session_history = self._get_session_history(authKeyId)
            if isinstance(session_history, DatabaseChatMessageHistory):
                await session_history._load_messages()

            # 메모리 검색 여부 결정
            skip_rag = MemorySearchStrategy.should_skip_search(user_input, session_history)

            # 입력 데이터 구성
            input_data = {
                "input": user_input,
                "user_input": user_input,
                "user_id": user_id,
                "authKeyId": authKeyId,
                "previous_analysis": self._get_last_analysis(authKeyId),
                "memories": []
            }

            # 메모리 검색 (선택적)
            if not skip_rag:
                try:
                    memories = await self._search_memories(input_data)
                    input_data["memories"] = memories
                except Exception as e:
                    logger.warning(f" 메모리 검색 실패: {e}")
                    input_data["memories"] = []

            logger.info(f" 입력: {user_input[:30]}... | RAG 생략: {skip_rag} | 기억 수: {len(input_data['memories'])}")

            # AI 응답 생성
            ai_output = await self.chain_with_history.ainvoke(
                input_data,
                config={"configurable": {"session_id": authKeyId}}
            )

            result = ai_output["output"]

            # 대화 저장 (비동기)
            asyncio.create_task(
                self._save_conversation(authKeyId, user_input, result["response"])
            )

            # 응답 구성
            return {
                "status": "success",
                "response": result["response"],
                "emotion_analysis": result["analysis"],
                "used_memories": self._format_used_memories(input_data.get("memories", [])),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f" 대화 생성 실패: {e}")
            
            # 메모리 없이라도 프롬프트로 응답 생성 시도
            try:
                logger.info("🔄 메모리 없이 재시도")
                fallback_input = {
                    "input": user_input,
                    "user_input": user_input,
                    "user_id": user_id,
                    "authKeyId": authKeyId,
                    "previous_analysis": "",
                    "memories": []
                }
                
                fallback_output = await self.chain_with_history.ainvoke(
                    fallback_input,
                    config={"configurable": {"session_id": authKeyId}}
                )
                
                result = fallback_output["output"]
                logger.info(" 메모리 없이 응답 생성 성공")
                
                return {
                    "status": "success",
                    "response": result["response"],
                    "emotion_analysis": result.get("analysis", ""),
                    "used_memories": [],
                    "timestamp": datetime.now().isoformat(),
                    "fallback": True
                }
                
            except Exception as fallback_error:
                logger.error(f" 폴백 응답도 실패: {fallback_error}")
                return {
                    "status": "success",  # 사용자 경험을 위해 success 유지
                    "response": "안녕하세요! 무엇을 도와드릴까요?",
                    "emotion_analysis": "",
                    "used_memories": [],
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }

    async def _search_memories(self, data: Dict) -> List[Dict]:
        """메모리 검색 실행"""
        try:
            query = data["user_input"].strip()
            
            # 검색어 길이에 따른 타임아웃 조정
            if len(query) <= 2:
                if len(query) == 1:
                    logger.info(" 한 글자 검색어는 검색 생략")
                    return []
                    
                logger.info(f" 짧은 검색어 '{query}' - 빠른 검색 모드")
                timeout = 5.0
                max_results = 3
            else:
                logger.info(f" 일반 검색: '{query}'")
                timeout = 15.0
                max_results = 5

            result = await asyncio.wait_for(
                advanced_rag_service.search_memories(
                    query=query,
                    authKeyId=data["authKeyId"]
                ),
                timeout=timeout
            )
            
            return result[:max_results]

        except asyncio.TimeoutError:
            logger.warning(f" 검색 타임아웃: '{query}' - 빈 결과 반환")
            return []
        except Exception as e:
            logger.error(f" 메모리 검색 실패: {e}")
            return []

    async def _get_deceased_info(self, data: Dict) -> Dict:
        """고인 정보 조회"""
        try:
            return await database_service.get_deceased_by_auth_key(data["authKeyId"])
        except Exception as e:
            logger.warning(f" 고인 정보 조회 실패: {e}")
            # 기본값 반환
            return {
                "name": "소중한 분",
                "nickname": "소중한 분",
                "personality": "친근하고 따뜻한",
                "speaking_style": "다정하고 부드러운",
                "hobbies": "",
                "age": "",
                "user_name": "",
                "relation_to_user": "소중한 사람"
            }

    def _format_memories(self, data: Dict) -> str:
        """메모리 컨텍스트 포맷팅"""
        try:
            memories = data.get("memories", [])
            if not memories:
                return ""
                
            memory_texts = []
            for m in memories[:2]:  # 최대 2개만 사용
                date_text = m.get("date_text", "어느 날")
                content = m.get("content", "")
                
                # 내용 길이 제한
                if len(content) > 50:
                    content = content[:47] + "..."
                    
                memory_texts.append(f"{date_text}에 있었던 일: {content}")
                
            return " 관련 기억:\n" + "\n".join(memory_texts)
        except Exception as e:
            logger.warning(f" 메모리 포맷팅 실패: {e}")
            return ""

    def _format_used_memories(self, memories: List[Dict]) -> List[Dict]:
        """응답용 메모리 정보 포맷팅"""
        try:
            return [
                {
                    "collection": m.get("collection", ""),
                    "content": m.get("content", ""),
                    "score": round(m.get("score", 0.0), 4),
                    "date_text": m.get("date_text", ""),
                    "emotion_tone": m.get("metadata", {}).get("emotion_tone", ""),
                    "tags": m.get("metadata", {}).get("tags", []),
                    "relevance_score": m.get("relevance_score", 0.0)
                }
                for m in memories
            ]
        except Exception as e:
            logger.warning(f" 메모리 포맷팅 실패: {e}")
            return []

    async def _save_conversation(self, authKeyId: str, user_message: str, ai_response: str):
        """대화 저장 (비동기)"""
        try:
            await database_service.save_conversation(
                authKeyId=authKeyId,
                sender="USER",
                message=user_message
            )
            await database_service.save_conversation(
                authKeyId=authKeyId,
                sender="CHATBOT",
                message=ai_response
            )
            logger.debug(" 대화 저장 완료")
        except Exception as e:
            logger.error(f" 대화 저장 실패: {e}")


# 글로벌 인스턴스
chat_chain = ChatChain()