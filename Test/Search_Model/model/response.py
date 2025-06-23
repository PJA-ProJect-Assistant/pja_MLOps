# models/response.py (기존 코드에 추가)
from pydantic import BaseModel, Field
from typing import Dict, List, Any

# 대시보드 파이프라인
class SearchshimilerResponse(BaseModel) :
    similer_ID : dict = Field(..., description="project_index")