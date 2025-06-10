# models/response.py (기존 코드에 추가)
from pydantic import BaseModel, Field
from typing import List, Any

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI 응답 텍스트")
    model: str = Field(..., description="사용된 모델 이름")
    total_tokens: int = Field(..., description="총 사용된 토큰 수")
    prompt_tokens: int = Field(..., description="입력 프롬프트의 토큰 수")
    completion_tokens: int = Field(..., description="생성된 응답의 토큰 수")

class RequirementsResponse(BaseModel):
    requirements: List[dict] = Field(..., description="생성된 요구사항 목록")
    model: str = Field(..., description="사용된 모델 이름")
    total_tokens: int = Field(..., description="총 사용된 토큰 수")
    prompt_tokens: int = Field(..., description="입력 프롬프트의 토큰 수")
    completion_tokens: int = Field(..., description="생성된 응답의 토큰 수")