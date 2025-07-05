# 1. 베이스 이미지
FROM python:3.11-slim

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 의존성 설치에 필요한 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스코드 복사
COPY . .

# 6. FastAPI 실행 명령 (uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
