
# app/schemas/__init__.py
"""
스키마 패키지 - 현재 구조에 맞춘 재정비
순환 import 방지를 위해 필요한 것만 노출
"""

# 기본적으로 자주 사용되는 스키마들만 노출
from .commons_schemas import BaseResponse, DeceasedInfo, MemoryMetadata

# 각 모듈별로 필요할 때 직접 import하도록 함
# from .chat_schemas import ChatRequest, ChatResponse
# from .letter_schemas import LetterProcessRequest, LetterProcessResponse  
# from .memory_schemas import ProcessKeepsakeRequest, MemoryProcessResponse
# from .admin_schemas import DeleteRequest, DeleteResponse
# from .search_schemas import SearchRequest, SearchResponse
