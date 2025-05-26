# Dockerfile
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 저장소 clone (필요시 private일 경우 SSH 설정 필요)
RUN git clone https://github.com/listgreen/PJA_MLOps.git /app

# requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 포트 열기 (Streamlit, Flask, FastAPI용)
EXPOSE 8501 5000 8000

# 실행 명령
CMD ["python", "/app/app/main.py"]
