# main.py 파일 끝에 주석 한 줄 추가
# 업데이트 테스트: 2024-06-15 12:51
from fastapi import FastAPI
from routers import json_summury, requirements, json_ERDAPI, recommendation

app = FastAPI(
    title="FastAPI LLM Project",
    description="PJA_ProJect LLM 사용",
    version="0.0.1"
)

# 라우터 등록
app.include_router(requirements.router, prefix="/api/PJA", tags=["요구사항 명세서 생성"])    
app.include_router(json_summury.router, prefix="/api/PJA", tags=["프로젝트 요약 생성"])
app.include_router(json_ERDAPI.router, prefix="/api/PJA", tags=["ERD, API 명세서 생성"])
app.include_router(recommendation.router, prefix="/api/PJA", tags=["프로젝트 진행 추천"])