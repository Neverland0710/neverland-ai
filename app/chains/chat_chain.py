from typing import Dict, List, Any
from datetime import datetime
import os
import asyncio

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
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._messages = []
        self._loaded = False

    async def _load_messages(self):
        if self._loaded:
            return
        try:
            conversations = await database_service.get_recent_conversations(self.session_id, limit=10)
            for conv in conversations:
                if conv["sender"] == "USER":
                    self._messages.append(HumanMessage(content=conv["message"]))
                else:
                    self._messages.append(AIMessage(content=conv["message"]))
            self._loaded = True
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

class SplitResponseParser:
    MAX_RESPONSE_LENGTH = 300

    def __call__(self, text: Any) -> Dict[str, Any]:
        if isinstance(text, AIMessage):
            text = text.content
        elif not isinstance(text, str):
            text = str(text)

        logger.debug(f"🔎 GPT 원본 응답: {text}")
        response, analysis, risk = "", "", "LOW"

        if "|" in text:
            try:
                parts = text.strip().split("|")
                response = parts[0].strip().lstrip("응답 내용:").strip().strip("'\"")
                analysis = parts[1].strip().lstrip("분위기 분석 요약:").strip()
                risk = parts[2].strip().replace("위험도:", "").strip().upper()
            except Exception as e:
                logger.warning(f"⚠️ '|' 파싱 실패: {e}")
        else:
            lines = text.strip().splitlines()
            for line in lines:
                if "응답 내용:" in line:
                    response = line.split("응답 내용:", 1)[1].strip().strip("'\"")
                elif "분위기 분석 요약:" in line:
                    analysis = line.split("분위기 분석 요약:", 1)[1].strip()
                elif "위험도:" in line:
                    risk = line.split("위험도:", 1)[1].strip().upper()

        if not response:
            logger.warning("⚠️ 응답 파싱 실패 - 기본 메시지로 대체")
            response = "미안해, 지금은 잘 대답이 안 돼. 다시 한 번 이야기해줄래?"

        if len(response) > self.MAX_RESPONSE_LENGTH:
            response = response[:self.MAX_RESPONSE_LENGTH - 3] + "..."

        return {
            "output": {
                "response": response,
                "analysis": analysis,
                "risk": risk
            }
        }

class ChatChain:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=settings.openai_api_key
        )
        self.base_chain = self._build_base_chain()
        self.chain_with_history = RunnableWithMessageHistory(
            runnable=self.base_chain,
            get_session_history=self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        self.session_histories = {}

    def _build_base_chain(self) -> Runnable:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", ChatPrompts.RESPONSE_GENERATION)
        ])

        chain = (
            RunnablePassthrough.assign(
                memories=RunnableLambda(self._search_memories),
                deceased_info=RunnableLambda(self._get_deceased_info)
            )
            | RunnablePassthrough.assign(
                memory_context=RunnableLambda(self._format_memories),
                deceased_name=lambda x: x["deceased_info"]["name"],
                deceased_nickname=lambda x: x["deceased_info"]["nickname"],
                personality=lambda x: x["deceased_info"]["personality"],
                speaking_style=lambda x: x["deceased_info"]["speaking_style"],
                hobbies=lambda x: x["deceased_info"]["hobbies"],
                age=lambda x: x["deceased_info"]["age"],
                user_name=lambda x: x["deceased_info"]["user_name"],
                relation_to_user=lambda x: x["deceased_info"]["relation_to_user"],
                conversation_history=lambda x: self._get_recent_text_messages(self._get_session_history(x["authKeyId"])),
                date_text=lambda x: self._extract_date_text(x.get("memories", [])),
                emotion_tone=lambda x: x.get("previous_analysis", "")
            )
            | prompt_template
            | self.llm
            | SplitResponseParser()
        )

        return chain

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.session_histories:
            self.session_histories[session_id] = DatabaseChatMessageHistory(session_id)
        return self.session_histories[session_id]

    def _get_recent_text_messages(self, history: DatabaseChatMessageHistory, limit: int = 10) -> str:
        messages = history.messages[-limit:] if history else []
        text = []
        for m in messages:
            if isinstance(m, HumanMessage):
                text.append(f" {m.content}")
            elif isinstance(m, AIMessage):
                text.append(f" {m.content}")
        return "\n".join(text) if text else "(최근 대화 없음)"

    def _extract_date_text(self, memories: List[Dict]) -> str:
        if not memories:
            return "예전 어느 날"
        return memories[0].get("date_text", "한참 전")

    def _get_last_analysis(self, session_id: str) -> str:
        history = self._get_session_history(session_id)
        if not history or not history.messages:
            return ""
        for msg in reversed(history.messages):
            if isinstance(msg, AIMessage):
                parsed = SplitResponseParser()(msg.content)
                return parsed["output"].get("analysis", "")
        return ""

    def _should_skip_memory_search_by_content(self, user_input: str, history: DatabaseChatMessageHistory) -> bool:
        recent_ai_msgs = [m.content.lower() for m in reversed(history.messages[-50:]) if isinstance(m, AIMessage)]
        user_keywords = set(user_input.lower().split())
        for msg in recent_ai_msgs:
            msg_words = set(msg.split())
            overlap = user_keywords & msg_words
            if len(overlap) / max(len(user_keywords), 1) >= 0.6:
                logger.info(" 최근 응답과 유사한 내용 발견 → 기억 검색 생략")
                return True
        return False

    @traceable(name="generate_response")
    async def generate_response(self, user_input: str, user_id: str, authKeyId: str) -> Dict:
        try:
            session_history = self._get_session_history(authKeyId)
            if isinstance(session_history, DatabaseChatMessageHistory):
                await session_history._load_messages()

            skip_rag = self._should_skip_memory_search_by_content(user_input, session_history)

            input_data = {
                "input": user_input,
                "user_input": user_input,
                "user_id": user_id,
                "authKeyId": authKeyId,
                "previous_analysis": self._get_last_analysis(authKeyId)
            }   

            if not skip_rag:
                input_data["memories"] = await self._search_memories(input_data)
            else:
                input_data["memories"] = []

            logger.info(f" 입력: {user_input} | RAG 생략: {skip_rag} | 기억 수: {len(input_data['memories'])}")

            ai_output = await self.chain_with_history.ainvoke(
                input_data,
                config={"configurable": {"session_id": authKeyId}}
            )

            result = ai_output["output"]

            await self._save_conversation(authKeyId, user_input, result["response"])

            raw_memories = input_data.get("memories", [])
            used_memories = [
                {
                    "collection": m["collection"],
                    "content": m["content"],
                    "score": round(m.get("score", 0.0), 4),
                    "date_text": m.get("date_text"),
                    "emotion_tone": m["metadata"].get("emotion_tone"),
                    "tags": m["metadata"].get("tags"),
                    "relevance_score": m.get("relevance_score")
                }
                for m in raw_memories
            ]

            return {
                "status": "success",
                "response": result["response"],
                "emotion_analysis": result["analysis"],
                "used_memories": used_memories,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f" 대화 생성 실패: {e}")
            return {
                "status": "error",
                "response": "지금은 생각이 잘 정리되지 않네요. 다시 한 번 말해줄래요?",
                "error": str(e)
            }

    async def _search_memories(self, data: Dict) -> List[Dict]:
        try:
            query = data["user_input"].strip()

            if len(query) <= 2:
                logger.info(f" 짧은 검색어 감지: '{query}' - 빠른 검색 모드")
                if len(query) == 1:
                    logger.info(" 한 글자 검색어는 검색 생략")
                    return []
                try:
                    result = await asyncio.wait_for(
                        advanced_rag_service.search_memories(
                            query=query,
                            authKeyId=data["authKeyId"]
                        ),
                        timeout=5.0
                    )
                    return result[:3]
                except asyncio.TimeoutError:
                    logger.warning(f" 짧은 검색어 '{query}' 타임아웃 - 검색 생략")
                    return []

            logger.info(f" 일반 검색: '{query}'")

            try:
                result = await asyncio.wait_for(
                    advanced_rag_service.search_memories(
                        query=query,
                        authKeyId=data["authKeyId"]
                    ),
                    timeout=15.0
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(f" 검색어 '{query}' 타임아웃 - 빈 결과 반환")
                return []

        except Exception as e:
            logger.error(f" 메모리 검색 실패: {e}")
            return []

    async def _get_deceased_info(self, data: Dict) -> Dict:
        return await database_service.get_deceased_by_auth_key(data["authKeyId"])

    def _format_memories(self, data: Dict) -> str:
        memories = data.get("memories", [])
        if not memories:
            return ""
        memory_texts = []
        for m in memories[:2]:
            date_text = m.get("date_text", "어느 날")
            content = m["content"]
            if len(content) > 50:
                content = content[:47] + "..."
            memory_texts.append(f"{date_text}에 있었던 일: {content}")
        return " 관련 기억:\n" + "\n".join(memory_texts)

    async def _save_conversation(self, authKeyId: str, user_message: str, ai_response: str):
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
        except Exception as e:
            logger.error(f" 대화 저장 실패: {e}")

chat_chain = ChatChain()