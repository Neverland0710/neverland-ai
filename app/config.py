# app/config.py
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="allow")
    
    # OpenAI
    openai_api_key: str
    
    # ElevenLabs (음성 서비스용)
    elevenlabs_api_key: str
    
    # Qdrant Cloud
    qdrant_url: str
    qdrant_api_key: str
    
    # Collection Names
    daily_conversation_collection: str = "daily_conversations"
    letter_memory_collection: str = "letter_memories"
    object_memory_collection: str = "object_memories"
    
    # MySQL
    mysql_host: str 
    mysql_port: int 
    mysql_user: str
    mysql_password: str
    mysql_database: str
    
    # LangChain LangSmith 트래킹 관련
    langsmith_tracing: Optional[bool] = False
    langsmith_endpoint: Optional[str] = "https://api.smith.langchain.com"
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = "default"
    
    # Voice Service Settings
    voice_audio_dir: str = "static/audio"
    default_voice_id: str = "AW5wrnG1jVizOYY7R1Oo" 
    max_voice_length: int = 150  # 음성 응답 최대 길이
    
    # App Settings
    debug: bool = True
    log_level: str = "INFO"

settings = Settings()