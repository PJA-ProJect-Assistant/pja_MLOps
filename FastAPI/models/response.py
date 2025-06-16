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

class jsonResponse(BaseModel):
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