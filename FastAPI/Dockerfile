# Python 3.11 slim 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv 설치 (빠른 패키지 관리자)
RUN pip install uv

# requirements.txt 복사 및 의존성 설치
# context가 ./FastAPI이므로 현재 디렉토리의 requirements.txt를 복사
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then \
    echo "Using requirements.txt"; \
    uv pip install --system -r requirements.txt; \
    else \
    echo "No requirements.txt found, installing basic FastAPI dependencies"; \
    uv pip install --system fastapi uvicorn[standard] pydantic openai python-multipart requests python-dotenv; \
    fi

# FastAPI 앱 파일들 복사 (context가 ./FastAPI이므로 현재 디렉토리의 모든 파일)
COPY . .

# startup 스크립트 처리
RUN if [ -f startup.sh ]; then \
    chmod +x startup.sh; \
    else \
    echo "Creating default startup script"; \
    echo '#!/bin/bash' > startup.sh; \
    echo 'echo "Starting FastAPI application..."' >> startup.sh; \
    echo 'uvicorn main:app --host 0.0.0.0 --port 8000' >> startup.sh; \
    chmod +x startup.sh; \
    fi

# 환경변수 설정 (프로덕션용)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 포트 8000 노출
EXPOSE 8000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# startup 스크립트 실행 또는 직접 uvicorn 실행
CMD if [ -f startup.sh ]; then \
    ./startup.sh; \
    else \
    uvicorn main:app --host 0.0.0.0 --port 8000; \
    fi