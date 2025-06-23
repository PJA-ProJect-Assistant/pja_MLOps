# main.py
# FastAPI 애플리케이션을 생성하고 라우터를 등록하는 코드
from fastapi import FastAPI
from routers import search_subject

app = FastAPI(
    title="FastAPI LLM Project",
    description="PJA_ProJect LLM 사용",
    version="1.0.0"
)

# 라우터 등록
app.include_router(search_subject.router, prefix="/api/PJA", tags=["유사한 프로젝트 검색"])    
