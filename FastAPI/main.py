# main.py
# FastAPI 애플리케이션을 생성하고 라우터를 등록하는 코드1
from fastapi import FastAPI
from routers import json_summury, requirements, json_ERD, json_API, recommendation

app = FastAPI(
    title="FastAPI LLM Project",
    description="PJA_ProJect LLM 사용",
    version="0.0.1"
)

# 라우터 등록
app.include_router(requirements.router, prefix="/api/PJA", tags=["요구사항 명세서 생성"])    
app.include_router(json_summury.router, prefix="/api/PJA", tags=["프로젝트 요약 생성"])
app.include_router(json_ERD.router, prefix="/api/PJA", tags=["ERD 명세서 생성"])
app.include_router(json_API.router, prefix="/api/PJA", tags=["API 명세서 생성"])
app.include_router(recommendation.router, prefix="/api/PJA", tags=["프로젝트 진행 추천"])