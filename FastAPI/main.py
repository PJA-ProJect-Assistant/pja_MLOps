# main.py
from fastapi import FastAPI
from routers import json_summury, requirements, json_ERDAPI

app = FastAPI(
    title="FastAPI LLM Project",
    description="PJA_ProJect LLM 사용",
    version="0.0.1"
)

# 라우터 등록
app.include_router(requirements.router, prefix="/api/pja", tags=["요구사항 명세서 생성"])    
app.include_router(json_summury.router, prefix="/api/pja", tags=["프로젝트 요약 생성"])
app.include_router(json_ERDAPI.router, prefix="/api/pja", tags=["ERD, API 명세서 생성"])