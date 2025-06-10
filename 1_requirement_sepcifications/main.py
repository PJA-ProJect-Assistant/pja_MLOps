# main.py
from fastapi import FastAPI
from routers import requirements

app = FastAPI(
    title="FastAPI LLM Project",
    description="FastAPI와 LangChain을 사용한 LLM 프로젝트",
    version="1.0.0"
)

# 라우터 등록
app.include_router(requirements.router, prefix="/api/v1", tags=["requirements"])    