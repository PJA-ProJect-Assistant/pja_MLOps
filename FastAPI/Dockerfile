# Python 3.11 slim 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# uv 설치 (빠른 패키지 관리자)
RUN pip install uv

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then \
    echo "📋 Installing from requirements.txt"; \
    uv pip install --system -r requirements.txt; \
    else \
    echo "❌ No requirements.txt found, installing basic FastAPI dependencies"; \
    uv pip install --system fastapi uvicorn[standard] pydantic openai python-multipart requests python-dotenv langchain langchain-community scikit-learn konlpy pandas jinja2; \
    fi

# FastAPI 앱 파일들 복사
COPY . .

# 환경변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 포트 8000 노출
EXPOSE 8000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
