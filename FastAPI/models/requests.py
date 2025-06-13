# models/requests.py (기존 코드에 추가)
from pydantic import BaseModel, Field
from typing import List, Optional, Any


class Message(BaseModel):
    role: str = Field(..., description="메시지 작성자의 역할 (system, user, assistant)")
    content: str = Field(..., description="메시지 내용")

class RequirementsRequest(BaseModel):
    project_overview: str = Field(..., description="사용자의 아이디어 작성 내용")
    existing_requirements: str = Field(..., description="기존 요구사항 목록")
    additional_count: int = Field(5, ge=1, le=20, description="추가로 생성할 요구사항 개수")
    max_tokens: Optional[int] = Field(4000, ge=1, le=8000, description="생성할 최대 토큰 개수")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="생성 창의성 정도")
    model: str = Field("gpt-4o-mini", description="사용할 모델 이름")

class SummuryRequest(BaseModel):
    project_overview: str = Field(..., description="사용자의 아이디어 작성 내용")
    requirements : str = Field(..., description="추가 요구사항 목록")
    max_tokens: Optional[int] = Field(4000, ge=1, le=4000, description="생성할 최대 토큰 개수")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="생성 창의성 정도")
    model: str = Field("ft:gpt-4o-mini-2024-07-18:test::BebIPMSD", description="사용할 모델 이름")

class jsonRequest(BaseModel):
    project_overview: str = Field(..., description="사용자의 아이디어 작성 내용")
    requirements : str = Field(..., description="추가 요구사항 목록")
    project_summury : str = Field(..., description="프로젝트 요약 내용용")
    max_tokens: Optional[int] = Field(8000, ge=1, le=8000, description="생성할 최대 토큰 개수")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="생성 창의성 정도")
    model: str = Field("ft:gpt-4o-mini-2024-07-18:test::BebIPMSD", description="사용할 모델 이름")

class RecommendationRequest(BaseModel):
    project_list: str = Field(..., description="팀 프로젝트에 작성된 리스트")
    max_tokens: Optional[int] = Field(3000, ge=1, le=3000, description="생성할 최대 토큰 개수")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="생성 창의성 정도")
    model: str = Field("gpt-4o-mini", description="사용할 모델 이름")