from fastapi import FastAPI
from routers import json_summury, requirements, json_ERD, json_API, recommendation, stats, task_generate, search_subject

app = FastAPI(
    title="FastAPI LLM Project",
    description="PJA_ProJect LLM 사용",
    version="1.0.0"
)

# 헬스체크 엔드포인트 추가
@app.get("/")
async def root():
    return {"message": "PJA Project API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "PJA Project API is running"}

# 라우터 등록
app.include_router(requirements.router, prefix="/api/PJA", tags=["요구사항 명세서 생성"])    
app.include_router(json_summury.router, prefix="/api/PJA", tags=["프로젝트 요약 생성"])
app.include_router(json_ERD.router, prefix="/api/PJA", tags=["ERD 명세서 생성"])
app.include_router(json_API.router, prefix="/api/PJA", tags=["API 명세서 생성"])
app.include_router(recommendation.router, prefix="/api/PJA", tags=["프로젝트 진행 추천"])
app.include_router(stats.router, prefix="/api/PJA", tags=["대시보드용 통계 파이프라인"])
app.include_router(task_generate.router, prefix="/api/PJA", tags=["카테고리&기능&액션 추천"])
app.include_router(search_subject.router, prefix="/api/PJA", tags=["유사한 프로젝트 검색"])
