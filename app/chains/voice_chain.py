# chains/voice_chain.py
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
from langsmith import traceable

from app.config import settings
from app.utils.logger import logger
from app.services.advanced_rag_service import advanced_rag_service
from app.services.database_service import database_service
from app.prompts.voice_prompt import VoicePrompts

# LangSmith 안전 임포트
try:
    from langsmith import traceable
    # LangSmith 설정
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
except ImportError:
    logger.warning("⚠️ LangSmith 패키지가 설치되지 않음 - 추적 기능 비활성화")
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator

class VoiceMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._messages = []
        self._loaded = False

    async def _load_voice_messages(self):
        if self._loaded:
            return
        try:
            conversations = await database_service.get_recent_conversations(
                self.session_id, limit=10
            )
            for conv in conversations:
                if conv["sender"] == "USER":
                    self._messages.append(HumanMessage(content=conv["message"]))
                else:
                    self._messages.append(AIMessage(content=conv["message"]))
            self._loaded = True
        except Exception as e:
            logger.error(f"❌ 대화 히스토리 로드 실패: {e}")
            self._loaded = True

    @property
    def messages(self):
        return self._messages

    def add_message(self, message):
        self._messages.append(message)

    def clear(self):
        self._messages.clear()

class VoiceResponseParser:
    def __call__(self, text: Any) -> Dict[str, Any]:
        if isinstance(text, AIMessage):
            text = text.content
        elif not isinstance(text, str):
            text = str(text)

        logger.debug(f"🎤 음성 GPT 원본 응답: {text}")
        response, analysis, risk = "", "", "LOW"

        if "|" in text:
            try:
                parts = text.strip().split("|")
                response = parts[0].strip().lstrip("응답 내용:").strip().strip("'\"")
                analysis = parts[1].strip().lstrip("음성 분위기 분석:").strip()
                risk = parts[2].strip().replace("감정 위험도:", "").strip().upper()
            except Exception as e:
                logger.warning(f"⚠️ 음성 파싱 실패: {e}")
        else:
            lines = text.strip().splitlines()
            for line in lines:
                if "응답 내용:" in line:
                    response = line.split("응답 내용:", 1)[1].strip().strip("'\"")
                elif "음성 분위기 분석:" in line:
                    analysis = line.split("음성 분위기 분석:", 1)[1].strip()
                elif "감정 위험도:" in line:
                    risk = line.split("감정 위험도:", 1)[1].strip().upper()

        if not response:
            logger.warning("⚠️ 음성 응답 파싱 실패 - 기본 메시지로 대체")
            response = "어... 잠깐만, 뭐라고 했지? 다시 한 번 말해줄래?"

        if len(response) > 150:
            response = response[:147] + "..."

        replacements = {
            "ㅋㅋ": "",
            "ㅎㅎ": "",
            "ㅠㅠ": "",
        }
        for bad, good in replacements.items():
            response = response.replace(bad, good)

        return {
            "output": {
                "response": response,
                "voice_analysis": analysis,
                "emotion_risk": risk
            }
        }

class VoiceChain:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            openai_api_key=settings.openai_api_key
        )
        self.base_chain = self._build_voice_chain()
        self.chain_with_history = RunnableWithMessageHistory(
            runnable=self.base_chain,
            get_session_history=self._get_voice_session_history,
            input_messages_key="input",
            history_messages_key="voice_history",
        )
        self.voice_session_histories = {}

    def _build_voice_chain(self) -> Runnable:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", VoicePrompts.VOICE_RESPONSE_GENERATION)
        ])

        chain = (
            RunnablePassthrough.assign(
                memories=RunnableLambda(self._search_voice_memories),
                deceased_info=RunnableLambda(self._get_deceased_info)
            )
            | RunnablePassthrough.assign(
                memory_context=RunnableLambda(self._format_voice_memories),
                deceased_name=lambda x: x["deceased_info"]["name"],
                deceased_nickname=lambda x: x["deceased_info"]["nickname"],
                personality=lambda x: x["deceased_info"]["personality"],
                speaking_style=lambda x: x["deceased_info"]["speaking_style"],
                hobbies=lambda x: x["deceased_info"].get("hobbies"),
                age=lambda x: x["deceased_info"].get("age"),
                user_name=lambda x: x["deceased_info"]["user_name"],
                relation_to_user=lambda x: x["deceased_info"]["relation_to_user"],
                conversation_history=lambda x: self._get_recent_voice_messages(
                    self._get_voice_session_history(x["authKeyId"])
                ),
                date_text=lambda x: self._extract_date_text(x.get("memories", [])),
                voice_emotion=lambda x: x.get("voice_emotion", "neutral"),
            )
            | prompt_template
            | self.llm
            | VoiceResponseParser()
        )

        return chain

    def _get_voice_session_history(self, session_id: str) -> VoiceMessageHistory:
        if session_id not in self.voice_session_histories:
            self.voice_session_histories[session_id] = VoiceMessageHistory(session_id)
        return self.voice_session_histories[session_id]

    def _get_recent_voice_messages(self, history: VoiceMessageHistory, limit: int = 5) -> str:
        messages = history.messages[-limit:] if history else []
        text = []
        for m in messages:
            if isinstance(m, HumanMessage):
                text.append(f"👤 {m.content}")
            elif isinstance(m, AIMessage):
                text.append(f"🤖 {m.content}")
        return "\n".join(text) if text else "(대화 기록 없음)"

    def _extract_date_text(self, memories: List[Dict]) -> str:
        if not memories:
            return "예전 어느 날"
        return memories[0].get("date_text", "한참 전")

    def _get_last_voice_analysis(self, session_id: str) -> str:
        history = self._get_voice_session_history(session_id)
        if not history or not history.messages:
            return ""
        for msg in reversed(history.messages):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and "|" in msg.content:
                return msg.content.split("|")[-1].strip()
        return ""

    @traceable(name="generate_voice_response")
    async def generate_voice_response(
        self, 
        user_speech_text: str, 
        user_id: str, 
        authKeyId: str,
        voice_emotion: str = "neutral"
    ) -> Dict:
        try:
            session_history = self._get_voice_session_history(authKeyId)
            if isinstance(session_history, VoiceMessageHistory):
                await session_history._load_voice_messages()

            input_data = {
                "input": user_speech_text,
                "user_input": user_speech_text,
                "user_id": user_id,
                "authKeyId": authKeyId,
                "voice_emotion": voice_emotion,
                "previous_voice_analysis": self._get_last_voice_analysis(authKeyId)
            }

            ai_output = await self.chain_with_history.ainvoke(
                input_data,
                config={"configurable": {"session_id": authKeyId}}
            )

            result = ai_output["output"]

            await self._save_voice_conversation(
                authKeyId, user_speech_text, result["response"]
            )

            raw_memories = await self._search_voice_memories(input_data)

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
                "voice_response": result["response"],
                "voice_analysis": result["voice_analysis"],
                "emotion_risk": result["emotion_risk"],
                "used_memories": used_memories,
                "response_length": len(result["response"]),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ 음성 응답 생성 실패: {e}")
            return {
                "status": "error",
                "voice_response": "어... 잠깐만, 무슨 말인지 잘 안 들렸어. 다시 한 번 말해줄래?",
                "error": str(e)
            }

    async def _search_voice_memories(self, data: Dict) -> List[Dict]:
        try:
            query = data["user_input"].strip()
            if len(query) <= 2:
                logger.info(f"🔍 짧은 검색어 감지: '{query}' - 빠른 검색 모드")
                if len(query) == 1:
                    logger.info("🚫 한 글자 검색어는 검색 생략")
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
                    logger.warning(f"⏰ 짧은 검색어 '{query}' 타임아웃 - 검색 생략")
                    return []

            logger.info(f"🔍 일반 검색: '{query}'")

            try:
                result = await asyncio.wait_for(
                    advanced_rag_service.search_memories(
                        query=query,
                        authKeyId=data["authKeyId"]
                    ),
                    timeout=15.0
                )
                return result[:5]
            except asyncio.TimeoutError:
                logger.warning(f"⏰ 검색어 '{query}' 타임아웃 - 빈 결과 반환")
                return []

        except Exception as e:
            logger.error(f"❌ 메모리 검색 실패: {e}")
            return []

    async def _get_deceased_info(self, data: Dict) -> Dict:
        return await database_service.get_deceased_by_auth_key(data["authKeyId"])

    def _format_voice_memories(self, data: Dict) -> str:
        memories = data.get("memories", [])
        if not memories:
            return ""
        memory_texts = []
        for i, m in enumerate(memories[:2], 1):
            date_text = m.get('date_text', '언젠가')
            content = m['content']
            if len(content) > 50:
                content = content[:47] + "..."
            memory_texts.append(f"{date_text}에 {content}")
        return "🎤 관련 기억:\n" + "\n".join(memory_texts)

    async def _save_voice_conversation(
        self, 
        authKeyId: str, 
        user_speech: str, 
        ai_response: str
    ):
        try:
            await database_service.save_conversation(
                authKeyId=authKeyId,
                sender="USER",
                message=user_speech
            )
            await database_service.save_conversation(
                authKeyId=authKeyId,
                sender="CHATBOT",
                message=ai_response
            )
        except Exception as e:
            logger.error(f"❌ 대화 저장 실패: {e}")

voice_chain = VoiceChain()