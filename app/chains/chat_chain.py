from typing import Dict, List, Any
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
    logger.warning("LangSmith 패키지가 설치되지 않음 - 추적 기능 비활성화")
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
            logger.error(f"히스토리 로드 실패: {e}")
            self._loaded = True

    @property
    def messages(self):
        return self._messages

    def add_message(self, message):
        self._messages.append(message)

    def clear(self):
        self._messages.clear()

class ResponseParser:
    MAX_RESPONSE_LENGTH = 300

    def __call__(self, text: Any) -> AIMessage:
        if isinstance(text, AIMessage):
            text = text.content
        elif not isinstance(text, str):
            text = str(text)

        logger.debug(f"GPT 원본 응답: {text[:100]}...")

        response = self._extract_response(text)
        analysis = self._extract_analysis(text)
        risk = self._extract_risk(text)

        if len(response) > self.MAX_RESPONSE_LENGTH:
            response = response[:self.MAX_RESPONSE_LENGTH - 3] + "..."

        formatted = f"[대답]: {response}\n[분위기]: {analysis}\n[위험도]: {risk}"
        return AIMessage(content=formatted)

    def _extract_response(self, text: str) -> str:
        match = re.search(r"\[대답\]:\s*([\s\S]+?)(?:\n\[|$)", text)
        if match:
            return match.group(1).strip()
        # fallback: 첫 문장 또는 첫 줄 반환
        first_line = text.strip().split("\n")[0]
        return first_line if first_line else ""

    def _extract_analysis(self, text: str) -> str:
        match = re.search(r"\[분위기\]:\s*([^\n]+)", text)
        return match.group(1).strip() if match else ""

    def _extract_risk(self, text: str) -> str:
        match = re.search(r"\[위험도\]:\s*([^\n]+)", text)
        return match.group(1).strip().upper() if match else "LOW"

class ChatChain:
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
        self.session_histories: Dict[str, DatabaseChatMessageHistory] = {}
        logger.info("ChatChain 초기화 완료")

    def _build_chain(self) -> Runnable:
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
        if session_id not in self.session_histories:
            self.session_histories[session_id] = DatabaseChatMessageHistory(session_id)
        return self.session_histories[session_id]

    async def generate_response(self, user_input: str, user_id: str, authKeyId: str) -> Dict:
        try:
            session_history = self._get_session_history(authKeyId)
            await session_history._load_messages()

            input_data = {
                "input": user_input,
                "user_input": user_input,
                "user_id": user_id,
                "authKeyId": authKeyId,
                "previous_analysis": self._get_last_analysis(authKeyId),
                "memories": []
            }

            memories = await self._search_memories(input_data)
            input_data["memories"] = memories[:1]  # 상위 1개 기억만 전달

            ai_output = await self.chain_with_history.ainvoke(
                input_data,
                config={"configurable": {"session_id": authKeyId}}
            )

            logger.debug(f"GPT 응답 원문: {ai_output}")

            parsed = self.response_parser(ai_output)
            logger.debug(f"파싱된 응답: {parsed.content}")

            response = self.response_parser._extract_response(parsed.content)
            analysis = self.response_parser._extract_analysis(parsed.content)
            risk = self.response_parser._extract_risk(parsed.content)

            if not response:
                logger.warning("GPT 응답이 비어있음 또는 파싱 실패 → fallback 응답 반환")
                return {
                    "status": "success",
                    "response": "미안, 지금은 그 이야기를 잘 떠올릴 수가 없어. 조금만 더 얘기해줄래?",
                    "emotion_analysis": "",
                    "risk": "LOW",
                    "used_memories": self._format_used_memories(memories),
                    "timestamp": datetime.now().isoformat(),
                    "fallback": True
                }

            return {
                "status": "success",
                "response": response,
                "emotion_analysis": analysis,
                "risk": risk,
                "used_memories": self._format_used_memories(memories),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"대화 생성 실패: {e}")
            return {
                "status": "success",
                "response": "죄송합니다. 이 요청을 처리할 수 없습니다.",
                "emotion_analysis": "",
                "risk": "LOW",
                "used_memories": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "fallback": True
            }

    async def _search_memories(self, data: Dict) -> List[Dict]:
        try:
            query = data["user_input"].strip()
            return await advanced_rag_service.search_memories(query=query, authKeyId=data["authKeyId"])
        except Exception as e:
            logger.warning(f"메모리 검색 실패: {e}")
            return []

    async def _get_deceased_info(self, data: Dict) -> Dict:
        try:
            return await database_service.get_deceased_by_auth_key(data["authKeyId"])
        except Exception as e:
            logger.warning(f"고인 정보 조회 실패: {e}")
            return {
                "name": "소중한 분",
                "nickname": "소중한 분",
                "personality": "따뜻하고 이해심 많은",
                "speaking_style": "다정하고 따뜻한 말투",
                "hobbies": "",
                "age": "",
                "user_name": "",
                "relation_to_user": "소중한 사람"
            }

    def _format_memories(self, data: Dict) -> str:
        memories = data.get("memories", [])
        if not memories:
            return ""
        m = memories[0]
        date_text = m.get("date_text", "어느 날")
        content = m.get("content", "")
        if len(content) > 50:
            content = content[:47] + "..."
        return f"관련 기억:\n{date_text}에 있었던 일: {content}"

    def _extract_date_text(self, memories: List[Dict]) -> str:
        return memories[0].get("date_text", "예전 어느 날") if memories else "예전 어느 날"

    def _get_recent_messages(self, history: DatabaseChatMessageHistory) -> str:
        messages = history.messages[-10:] if history else []
        formatted = []
        for m in messages:
            if isinstance(m, HumanMessage):
                formatted.append(f"사용자: {m.content}")
            elif isinstance(m, AIMessage):
                formatted.append(f"AI: {m.content}")
        return "\n".join(formatted) if formatted else "(최근 대화 없음)"

    def _get_last_analysis(self, session_id: str) -> str:
        history = self._get_session_history(session_id)
        for msg in reversed(history.messages):
            if isinstance(msg, AIMessage):
                parsed = self.response_parser(msg.content)
                return self.response_parser._extract_analysis(parsed.content)
        return ""

    def _format_used_memories(self, memories: List[Dict]) -> List[Dict]:
        return [
            {
                "collection": m.get("collection", ""),
                "content": m.get("content", ""),
                "score": round(m.get("score", 0.0), 4),
                "date_text": m.get("date_text", ""),
                "tags": m.get("metadata", {}).get("tags", []),
                "relevance_score": m.get("relevance_score", 0.0)
            }
            for m in memories
        ]

chat_chain = ChatChain()
