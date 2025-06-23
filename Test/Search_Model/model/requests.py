# models/requests.py (기존 코드에 추가)
from pydantic import BaseModel, Field
from typing import List, Optional, Any

# 유사도 검색
class SearchshimilerRequest(BaseModel) :
    project_info : str = Field(..., description="유저의 project_info")
    top_k : int = Field(..., ge=10, le = 40, description = "")
    