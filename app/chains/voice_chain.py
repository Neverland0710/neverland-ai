# chains/voice_chain.py
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os
import asyncio
import re

from langchain_openai import ChatOpenAI
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.config import settings
from app.utils.logger import logger
from app.services.advanced_rag_service import advanced_rag_service
from app.services.database_service import database_service
from app.prompts.voice_prompt import VoicePrompts
from app.models.conversation import TextConversation
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))


try:
    from langsmith import traceable
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
except ImportError:
    def traceable(name=None):
        def decorator(func): return func
        return decorator


class VoiceMessageHistory(BaseChatMessageHistory):
    """음성 대화 히스토리 관리"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._messages = []
        self._loaded = False

    async def _load_voice_messages(self):
        """데이터베이스에서 음성 대화 히스토리 로드"""
        if self._loaded:
            return
            
        try:
            conversations = await database_service.get_recent_conversations(
                self.session_id, 
                limit=50  # 음성은 더 많은 컨텍스트 필요
            )
            
            for conv in conversations:
                if conv["sender"] == "USER":
                    self._messages.append(HumanMessage(content=conv["message"]))
                else:
                    self._messages.append(AIMessage(content=conv["message"]))
                    
            self._loaded = True
            logger.debug(f" 음성 히스토리 로드 완료: {len(self._messages)}개 메시지")
            
        except Exception as e:
            logger.error(f" 음성 히스토리 로드 실패: {e}")
            self._loaded = True

    @property
    def messages(self):
        return self._messages

    def add_message(self, message):
        self._messages.append(message)

    def clear(self):
        self._messages.clear()


class VoiceResponseParser:
    """음성 응답 전용 파서"""
    
    MAX_RESPONSE_LENGTH = 150  # 음성은 더 짧게
    METADATA_PREFIXES = ('요약:', '위험도:', '분석:')
    VOICE_CLEANUP_PATTERNS = ["ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "~", ".", "^^"]
    
    def __call__(self, text: Any) -> Dict[str, Any]:
        if isinstance(text, AIMessage):
            text = text.content
        elif not isinstance(text, str):
            text = str(text)

        logger.debug(f" 음성 GPT 원본 응답: {text[:100]}...")
        
        response = self._extract_response(text)
        analysis = self._extract_analysis(text)
        risk = self._extract_risk(text)
        
        # 음성용 특수문자 제거
        response = self._clean_for_voice(response)
        
        # 길이 제한 (음성은 더 짧게)
        if len(response) > self.MAX_RESPONSE_LENGTH:
            response = response[:self.MAX_RESPONSE_LENGTH - 3] + "..."

        logger.info(f" 음성 파싱 완료 - 응답: {response[:30]}... | 분석: {analysis[:20]}...")

        return {
            "output": {
                "response": response,
                "voice_analysis": analysis,
                "emotion_risk": risk
            }
        }
    
    def _extract_response(self, text: str) -> str:
        """음성용 실제 대화 응답 추출"""

        lines = text.strip().split('\n')

        # 0. [대답]: 키워드 우선 파싱
        for line in lines:
            if line.strip().startswith("[대답]:"):
                return line.strip().replace("[대답]:", "").strip().strip('"')

        # 1. 첫 번째 줄에서 메타데이터가 아닌 실제 대화 찾기
        for line in lines:
            line = line.strip()
            if line and not line.startswith(self.METADATA_PREFIXES):
                if line.startswith('"') and line.endswith('"'):
                    return line[1:-1]
                return line

        # 2. | 구분자 방식 파싱
        if "|" in text:
            try:
                response = text.strip().split("|")[0].strip()
                response = response.lstrip("응답 내용:").strip().strip("'\"")
                if response:
                    return response
            except Exception as e:
                logger.warning(f" 음성 '|' 파싱 실패: {e}")

        # 3. 라인별 키워드 파싱
        for line in lines:
            if "응답 내용:" in line:
                response = line.split("응답 내용:", 1)[1].strip().strip("'\"")
                if response:
                    return response

        # 4. 첫 문장 사용
        first_sentence = text.split('.')[0].strip()
        if first_sentence and len(first_sentence) > 5:
            return first_sentence

        # 5. 전체 응답 사용
        if text.strip() and len(text.strip()) > 10:
            return text.strip()

        # 6. 마지막 fallback
        logger.warning(" 음성 GPT 응답이 비어있거나 너무 짧음 - 기본 메시지 사용")
        return "안녕하세요! 무슨 이야기를 나누고 싶으신가요?"
    
    def _extract_analysis(self, text: str) -> str:
        """음성 분위기 분석 추출"""
        try:
            if "요약:" in text:
                return text.split("요약:")[1].split("\n")[0].strip()
            elif "음성 분위기 분석:" in text:
                return text.split("음성 분위기 분석:")[1].split("\n")[0].strip()
            elif "|" in text and len(text.split("|")) > 1:
                return text.split("|")[1].strip()
        except Exception:
            pass
        return ""
    
    def _extract_risk(self, text: str) -> str:
        """감정 위험도 추출"""
        try:
            if "위험도:" in text:
                return text.split("위험도:")[1].strip().upper()
            elif "감정 위험도:" in text:
                return text.split("감정 위험도:")[1].strip().upper()
            elif "|" in text and len(text.split("|")) > 2:
                return text.split("|")[2].strip().replace("감정 위험도:", "").strip().upper()
        except Exception:
            pass
        return "LOW"
    
    def _clean_for_voice(self, text: str) -> str:
        """음성용 텍스트 정리"""
        for pattern in self.VOICE_CLEANUP_PATTERNS:
            text = text.replace(pattern, "")
        return text.strip()


class VoiceSearchStrategy:
    """음성용 메모리 검색 전략"""
    
    SKIP_PHRASES = ["뭐라고", "다시 말해줘", "방금 뭐라고", "못 들었어", "안 들려"]
    
    @classmethod
    def should_skip_search(cls, query: str, messages: List[BaseMessage]) -> bool:
        """음성용 메모리 검색 생략 여부 결정"""
        normalized = query.lower().strip()
        
        # 1. 특정 음성 패턴
        if any(phrase in normalized for phrase in cls.SKIP_PHRASES):
            logger.info(" 음성 재확인 요청 감지 → 메모리 검색 생략")
            return True
        
        # 2. 너무 짧은 발화
        if len(query.strip()) <= 2:
            logger.info(" 짧은 음성 입력 → 메모리 검색 생략")
            return True
        
        # 3. 직전 발화와 동일
        if messages and isinstance(messages[-1], HumanMessage):
            if messages[-1].content.strip() == query.strip():
                logger.info(" 중복 음성 입력 → 메모리 검색 생략")
                return True
        
        return False


class VoiceChain:
    """AI 추모 음성 대화 체인 메인 클래스"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # 음성은 더 빠른 모델 사용
            temperature=0.2,
            openai_api_key=settings.openai_api_key
        )
        
        self.voice_parser = VoiceResponseParser()
        self.base_chain = self._build_voice_chain()
        self.chain_with_history = RunnableWithMessageHistory(
            runnable=self.base_chain,
            get_session_history=self._get_voice_session_history,
            input_messages_key="input",
            history_messages_key="voice_history",
        )
        self.voice_session_histories = {}
        
        logger.info(" VoiceChain 초기화 완료")

    def _build_voice_chain(self) -> Runnable:
        """음성 전용 LangChain 체인 구성"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", VoicePrompts.VOICE_RESPONSE_GENERATION)
        ])
        
        return (
            RunnablePassthrough.assign(
                memories=RunnableLambda(self._search_voice_memories),
                deceased_info=RunnableLambda(self._get_deceased_info)
            )
            | RunnablePassthrough.assign(
                memory_context=RunnableLambda(self._format_voice_memories),
                **self._create_voice_context_variables()
            )
            | prompt_template
            | self.llm
            | self.voice_parser
        )
    
    def _create_voice_context_variables(self) -> Dict:
        """음성용 컨텍스트 변수 생성"""
        return {
            "deceased_name": lambda x: x["deceased_info"].get("name", "소중한 분"),
            "deceased_nickname": lambda x: x["deceased_info"].get("nickname", "소중한 분"),
            "personality": lambda x: x["deceased_info"].get("personality", "친근하고 따뜻한"),
            "speaking_style": lambda x: x["deceased_info"].get("speaking_style", "다정하고 부드러운"),
            "hobbies": lambda x: x["deceased_info"].get("hobbies", ""),
            "age": lambda x: x["deceased_info"].get("age", ""),
            "user_name": lambda x: x["deceased_info"].get("user_name", ""),
            "relation_to_user": lambda x: x["deceased_info"].get("relation_to_user", "소중한 사람"),
            "conversation_history": lambda x: self._get_recent_voice_messages(
                self._get_voice_session_history(x["authKeyId"])
            ),
            "date_text": lambda x: self._extract_date_text(x.get("memories", [])),
            "voice_emotion": lambda x: x.get("voice_emotion", "neutral")
        }

    def _get_voice_session_history(self, session_id: str) -> VoiceMessageHistory:
        """음성 세션 히스토리 관리"""
        if session_id not in self.voice_session_histories:
            self.voice_session_histories[session_id] = VoiceMessageHistory(session_id)
        return self.voice_session_histories[session_id]

    def _get_recent_voice_messages(self, history: VoiceMessageHistory, limit: int = 10) -> str:
        """최근 음성 대화 메시지 포맷팅"""
        try:
            messages = history.messages[-limit:] if history else []
            formatted = []
            
            for m in messages:
                if isinstance(m, HumanMessage):
                    formatted.append(f" {m.content}")
                elif isinstance(m, AIMessage):
                    formatted.append(f" {m.content}")
                    
            return "\n".join(formatted) if formatted else "(대화 기록 없음)"
        except Exception as e:
            logger.warning(f" 음성 메시지 포맷팅 실패: {e}")
            return "(대화 기록 없음)"

    def _extract_date_text(self, memories: List[Dict]) -> str:
        """메모리에서 날짜 텍스트 추출"""
        return memories[0].get("date_text", "한참 전") if memories else "예전 어느 날"

    def _get_last_voice_analysis(self, session_id: str) -> str:
        """마지막 음성 분석 결과 가져오기"""
        try:
            history = self._get_voice_session_history(session_id)
            for msg in reversed(history.messages):
                if isinstance(msg, AIMessage) and "|" in msg.content:
                    return msg.content.split("|")[-1].strip()
            return ""
        except Exception as e:
            logger.warning(f" 마지막 음성 분석 추출 실패: {e}")
            return ""

    @traceable(name="generate_voice_response")
    async def generate_voice_response(
        self, 
        user_speech_text: str, 
        user_id: str, 
        authKeyId: str, 
        voice_emotion: str = "neutral"
    ) -> Dict:
        """메인 음성 응답 생성 함수"""
        try:
            # 음성 히스토리 로드
            session_history = self._get_voice_session_history(authKeyId)
            await session_history._load_voice_messages()

            # 입력 데이터 구성
            input_data = {
                "input": user_speech_text,
                "user_input": user_speech_text,
                "user_id": user_id,
                "authKeyId": authKeyId,
                "voice_emotion": voice_emotion,
                "previous_voice_analysis": self._get_last_voice_analysis(authKeyId),
                "memories": []
            }

            # 메모리 검색 (음성용 전략 적용)
            try:
                raw_memories = await self._search_voice_memories(input_data)
                input_data["memories"] = raw_memories
                logger.info(f" 음성 입력: {user_speech_text[:30]}... | 기억 수: {len(raw_memories)}")
            except Exception as e:
                logger.warning(f" 음성 메모리 검색 실패: {e}")
                raw_memories = []

            # AI 응답 생성
            ai_output = await self.chain_with_history.ainvoke(
                input_data,
                config={"configurable": {"session_id": authKeyId}}
            )

            result = ai_output["output"]

            # 대화 저장 (비동기)
            asyncio.create_task(
                self._save_voice_conversation(authKeyId, user_speech_text, result["response"])
            )

            # 응답 구성
            return {
                "status": "success",
                "voice_response": result["response"],
                "voice_analysis": result.get("voice_analysis", ""),
                "emotion_risk": result.get("emotion_risk", "LOW"),
                "used_memories": self._format_used_memories(raw_memories),
                "response_length": len(result["response"]),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f" 음성 응답 생성 실패: {e}")
            
            # 메모리 없이라도 프롬프트로 응답 생성 시도
            try:
                logger.info("🔄 음성 메모리 없이 재시도")
                fallback_input = {
                    "input": user_speech_text,
                    "user_input": user_speech_text,
                    "user_id": user_id,
                    "authKeyId": authKeyId,
                    "voice_emotion": voice_emotion,
                    "previous_voice_analysis": "",
                    "memories": []
                }
                
                fallback_output = await self.chain_with_history.ainvoke(
                    fallback_input,
                    config={"configurable": {"session_id": authKeyId}}
                )
                
                result = fallback_output["output"]
                logger.info(" 음성 메모리 없이 응답 생성 성공")
                
                return {
                    "status": "success",
                    "voice_response": result["response"],
                    "voice_analysis": result.get("voice_analysis", ""),
                    "emotion_risk": result.get("emotion_risk", "LOW"),
                    "used_memories": [],
                    "response_length": len(result["response"]),
                    "timestamp": datetime.now().isoformat(),
                    "fallback": True
                }
                
            except Exception as fallback_error:
                logger.error(f" 음성 폴백 응답도 실패: {fallback_error}")
                return {
                    "status": "success",  # 사용자 경험을 위해 success 유지
                    "voice_response": "안녕하세요! 무슨 이야기를 나누고 싶으신가요?",
                    "voice_analysis": "",
                    "emotion_risk": "LOW",
                    "used_memories": [],
                    "response_length": 0,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }

    async def _search_voice_memories(self, data: Dict) -> List[Dict]:
        """음성용 메모리 검색 실행"""
        try:
            query = data["user_input"].strip()
            history = self._get_voice_session_history(data["authKeyId"])

            # 음성용 검색 생략 전략 적용
            if VoiceSearchStrategy.should_skip_search(query, history.messages):
                return []

            logger.info(f" 음성 메모리 검색: '{query}'")

            # 음성용 타임아웃 (더 짧게)
            timeout = 5.0 if len(query) <= 2 else 8.0
            max_results = 3  # 음성은 적은 수의 메모리만 사용

            result = await asyncio.wait_for(
                advanced_rag_service.search_memories(
                    query=query,
                    authKeyId=data["authKeyId"]
                ),
                timeout=timeout
            )
            
            return result[:max_results]

        except asyncio.TimeoutError:
            logger.warning(f" 음성 검색 타임아웃: '{query}' - 빈 결과 반환")
            return []
        except Exception as e:
            logger.warning(f" 음성 메모리 검색 실패: {e}")
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

    def _format_voice_memories(self, data: Dict) -> str:
        """음성용 메모리 컨텍스트 포맷팅"""
        try:
            memories = data.get("memories", [])
            if not memories:
                return ""
                
            memory_texts = []
            for m in memories[:2]:  # 음성은 최대 2개만
                content = m.get('content', '')
                
                # 음성용 내용 길이 제한 (더 짧게)
                if len(content) > 40:
                    content = content[:37] + "..."
                    
                date_text = m.get('date_text', '언젠가')
                memory_texts.append(f"{date_text}에 {content}")
                
            return " 관련 기억:\n" + "\n".join(memory_texts)
        except Exception as e:
            logger.warning(f" 음성 메모리 포맷팅 실패: {e}")
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
            logger.warning(f" 음성 메모리 포맷팅 실패: {e}")
            return []

    async def _save_voice_conversation(self, authKeyId: str, user_speech: str, ai_response: str):
        """음성 대화 저장 (사용자 입력은 await, 응답은 create_task로 백그라운드 저장)"""
        try:
            now = datetime.now()
            user_time = now
            bot_time = now + timedelta(milliseconds=10)

            # 사용자 발화는 즉시 저장 (await)
            await database_service.save_conversation(
                authKeyId=authKeyId,
                sender="USER",
                message=user_speech,
                metadata={"sent_at": datetime.now(KST).isoformat()}
            )

            # 챗봇 응답은 백그라운드 저장
            asyncio.create_task(
                database_service.save_conversation(
                    authKeyId=authKeyId,
                    sender="CHATBOT",
                    message=ai_response,
                    metadata={"sent_at": datetime.now(KST).isoformat()}
                )
            )

            logger.debug(" 음성 대화 저장 완료 (USER await, CHATBOT 비동기)")

        except Exception as e:
            logger.error(f" 음성 대화 저장 실패: {e}")

# 글로벌 인스턴스
voice_chain = VoiceChain()