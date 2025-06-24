# models/response.py (기존 코드에 추가)
from pydantic import BaseModel, Field
from typing import Dict, List, Any

class RequirementsResponse(BaseModel):
    requirements: List[dict] = Field(..., description="생성된 요구사항 목록")
    model: str = Field(..., description="사용된 모델 이름")
    total_tokens: int = Field(..., description="총 사용된 토큰 수")
    prompt_tokens: int = Field(..., description="입력 프롬프트의 토큰 수")
    completion_tokens: int = Field(..., description="생성된 응답의 토큰 수")

class SummuryResponse(BaseModel):
    json_data: Dict[str, Any] = Field(..., alias="json", description="json 형태 답변 내용")
    model: str = Field(..., description="사용된 모델 이름")
    total_tokens: int = Field(..., description="총 사용된 토큰 수")
    prompt_tokens: int = Field(..., description="입력 프롬프트의 토큰 수")
    completion_tokens: int = Field(..., description="생성된 응답의 토큰 수")

class ERDResponse(BaseModel):
    json_data: Dict[str, Any] = Field(..., alias="json", description="json 형태 답변 내용")
    model: str = Field(..., description="사용된 모델 이름")
    total_tokens: int = Field(..., description="총 사용된 토큰 수")
    prompt_tokens: int = Field(..., description="입력 프롬프트의 토큰 수")
    completion_tokens: int = Field(..., description="생성된 응답의 토큰 수")

class APIResponse(BaseModel):
    json_data: Dict[str, Any] = Field(..., alias="json", description="json 형태 답변 내용")
    model: str = Field(..., description="사용된 모델 이름")
    total_tokens: int = Field(..., description="총 사용된 토큰 수")
    prompt_tokens: int = Field(..., description="입력 프롬프트의 토큰 수")
    completion_tokens: int = Field(..., description="생성된 응답의 토큰 수")

class RecommendationResponse(BaseModel):
    recommendations: Dict[str, Any] = Field(..., description="추천된 카테고리 목록")
    model: str = Field(..., description="사용된 모델 이름")
    total_tokens: int = Field(..., description="총 사용된 토큰 수")
    prompt_tokens: int = Field(..., description="입력 프롬프트의 토큰 수")
    completion_tokens: int = Field(..., description="생성된 응답의 토큰 수")

# 대시보드 파이프라인
class DashboardResponse(BaseModel) :
    task_imbalance : Dict[str, Any] = Field(..., description="작업 불균형 json_data")
    processing_time : Dict[str, Any] = Field(..., description="평균작업 처리 시간 json_data")

class TaskGenerateResponse(BaseModel):
<<<<<<< HEAD
    generated_tasks : Dict[str, Any] = Field(..., description="생성된 category, feature, actions 초안 json")
=======
    generated_tasks : Dict[str, Any] = Field(..., description="생성된 category, feature, actions 초안 json")

# 유사도 검색
class SearchshimilerResponse(BaseModel) :
    similer_ID : Dict[str, Any] = Field(..., description="project_index")
>>>>>>> 8f30b9539ae4266b4232a188a0ddb2922a35aaac
