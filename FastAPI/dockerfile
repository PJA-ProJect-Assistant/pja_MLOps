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
# FastAPI 디렉토리의 requirements.txt 우선 확인, 없으면 루트의 requirements.txt 사용
COPY FastAPI/requirements.txt* ./
COPY requirements.txt* ./
RUN if [ -f FastAPI/requirements.txt ]; then \
        echo "Using FastAPI/requirements.txt"; \
        uv pip install --system -r FastAPI/requirements.txt; \
    elif [ -f requirements.txt ]; then \
        echo "Using requirements.txt"; \
        uv pip install --system -r requirements.txt; \
    else \
        echo "No requirements.txt found, installing basic FastAPI dependencies"; \
        uv pip install --system fastapi uvicorn[standard] pydantic openai python-multipart requests python-dotenv; \
    fi

# 전체 프로젝트 복사
COPY . .

# FastAPI 앱이 있는 디렉토리로 작업 디렉토리 변경
WORKDIR /app/FastAPI

# startup 스크립트가 있다면 복사하고 실행 권한 부여
# startup.sh가 루트에 있는 경우와 FastAPI 디렉토리에 있는 경우 모두 처리
RUN if [ -f /app/startup.sh ]; then \
        cp /app/startup.sh /app/FastAPI/startup.sh; \
    fi && \
    if [ -f /app/FastAPI/startup.sh ]; then \
        chmod +x /app/FastAPI/startup.sh; \
    else \
        echo "Creating default startup script"; \
        echo '#!/bin/bash' > /app/FastAPI/startup.sh; \
        echo 'echo "Starting FastAPI application..."' >> /app/FastAPI/startup.sh; \
        echo 'uvicorn main:app --host 0.0.0.0 --port 8000' >> /app/FastAPI/startup.sh; \
        chmod +x /app/FastAPI/startup.sh; \
    fi

# 환경변수 설정 (프로덕션용)
ENV PYTHONPATH=/app:/app/FastAPI
ENV PYTHONUNBUFFERED=1

# 포트 8000 노출
EXPOSE 8000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# startup 스크립트 실행 또는 직접 uvicorn 실행
CMD if [ -f /app/FastAPI/startup.sh ]; then \
        /app/FastAPI/startup.sh; \
    else \
        uvicorn main:app --host 0.0.0.0 --port 8000; \
    fi