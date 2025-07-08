# 1. 베이스 이미지
FROM python:3.11-slim

# 2. 작업 디렉토리
WORKDIR /app

# 3. 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#  aiomysql 누락 시 예외 방지
RUN pip install aiomysql

# 5. 소스 복사
COPY . .

#  .env 강제 로드용 (선택적으로 ENV 복사해도 무방)
# COPY .env .  ← 필요 시 함께 복사

# 6. FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]