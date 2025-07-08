from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv
import os

# .env에서 환경 변수 로딩
load_dotenv()

#  환경 변수 가져오기 (대소문자 정확히 일치!)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

#  Qdrant 클라이언트 생성
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# 벡터 크기 (OpenAI text-embedding-3-small: 1536차원)
VECTOR_SIZE = 1536

#  생성할 컬렉션 목록
collections = [
    "daily_conversations",
    "letter_memories",
    "object_memories"
]

#  안전하게 컬렉션 생성 (존재하면 삭제 후 생성)
for name in collections:
    if client.collection_exists(name):
        client.delete_collection(name)
        print(f" Deleted existing: {name}")

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f" Created: {name}")
