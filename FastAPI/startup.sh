#!/bin/bash

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 배너 출력
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════╗"
echo "║                  PJA_Project                 ║"
echo "║          FastAPI + OpenAI Integration        ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# 현재 작업 디렉토리 확인 및 FastAPI 디렉토리로 이동
echo -e "${BLUE}📁 현재 작업 디렉토리: $(pwd)${NC}"

# FastAPI 디렉토리가 있는지 확인하고 이동
if [ -d "/app/FastAPI" ] && [ -f "/app/FastAPI/main.py" ]; then
    echo -e "${GREEN}✅ FastAPI 디렉토리로 이동합니다.${NC}"
    cd /app/FastAPI
elif [ -f "/app/main.py" ]; then
    echo -e "${GREEN}✅ 루트 디렉토리에서 main.py를 찾았습니다.${NC}"
    cd /app
elif [ -f "main.py" ]; then
    echo -e "${GREEN}✅ 현재 디렉토리에서 main.py를 찾았습니다.${NC}"
else
    echo -e "${RED}❌ main.py 파일을 찾을 수 없습니다.${NC}"
    echo -e "${YELLOW}🔍 파일 구조를 확인합니다...${NC}"
    find /app -name "main.py" -type f 2>/dev/null | head -5
    exit 1
fi

echo -e "${BLUE}📁 FastAPI 실행 디렉토리: $(pwd)${NC}"

# OpenAI API 키 확인 (환경변수에서)
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.${NC}"
    exit 1
fi

# API 키 유효성 간단 체크 (길이 확인)
if [ ${#OPENAI_API_KEY} -lt 20 ]; then
    echo -e "${RED}❌ 유효하지 않은 API 키입니다.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ API 키가 설정되었습니다.${NC}"

# main.py 파일 존재 확인
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ main.py 파일이 현재 디렉토리에 없습니다.${NC}"
    exit 1
fi

# Python 경로 설정
export PYTHONPATH="/app:/app/FastAPI:$PYTHONPATH"

# FastAPI 앱 import 테스트
echo -e "${BLUE}🧪 FastAPI 앱 import 테스트 중...${NC}"
python3 -c "
try:
    from main import app
    print('✅ FastAPI 앱 import 성공')
except ImportError as e:
    print(f'❌ Import 오류: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️ 기타 오류: {e}')
    exit(1)
" || {
    echo -e "${RED}❌ FastAPI 앱을 import할 수 없습니다.${NC}"
    exit 1
}

# 서버 시작 정보 출력
echo -e "${BLUE}"
echo "🚀 FastAPI 서버를 시작합니다..."
echo "📡 포트: 8000"
echo "🌐 접속 주소: http://13.209.5.218:8000"
echo "📚 API 문서: http://13.209.5.218:8000/docs"
echo ""
echo "🔗 사용 가능한 엔드포인트:"
echo "  • POST /api/PJA/requirements/generate (요구사항 생성)"
echo "  • POST /api/PJA/json_text/generate (JSON 생성)"
echo "  • POST /api/PJA/json_ERDAPI/generate (ERDAPI 생성)"
echo "  • POST /api/PJA/recommend/generate (프로젝트 진행 추천)"
echo ""
echo -e "${YELLOW}서버를 중지하려면 Ctrl+C를 누르세요.${NC}"
echo -e "${NC}"

# FastAPI 서버 실행
echo -e "${GREEN}🎯 uvicorn main:app --host 0.0.0.0 --port 8000 실행 중...${NC}"
exec uvicorn main:app --host 0.0.0.0 --port 8000