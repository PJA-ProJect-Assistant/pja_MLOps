# routers/Recommendation.py
from fastapi import APIRouter, HTTPException
import openai
import os
from dotenv import load_dotenv
from models.requests import RecommendationRequest
from models.response import RecommendationResponse
import json

# 환경변수 로드
load_dotenv()

router = APIRouter()

# OpenAI 클라이언트 초기화
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 최적화된 시스템 프롬프트
OPTIMIZED_SYSTEM_PROMPT = """
당신은 프로젝트 진행 상황을 분석하고 다음 작업을 추천하는 전문 AI 어시스턴트입니다.

## 주요 역할:
- 프로젝트의 현재 진행 상태를 분석
- 우선순위가 높은 다음 작업 추천
- 작업의 중요도와 예상 소요 시간 산정

## 응답 형식:
추천 작업은 다음 JSON 형식으로 제공되어야 합니다:
1. workspaceId: 작업 공간 ID
2. categoryId: 카테고리 ID
3. featureId: 기능 ID
4. recommendedActions: 추천 작업 목록

**중요: 응답은 반드시 순수한 JSON 형태로만 제공하세요.**
"""

@router.post("/recommend/generate", response_model=RecommendationResponse)
async def recommendation(request: RecommendationRequest):
    try:
        # 더미 데이터 로드 (실제 구현 시에는 ML 모델로 대체)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dummy_path = os.path.join(current_dir, "..", "backend_to_mlops_dummy_data.json")
        
        with open(dummy_path, "r", encoding="utf-8") as f:
            dummy_data = json.load(f)
        
        # 샘플 응답 데이터 구성
        sample_response = {
            "workspaceId": request.workspaceId,
            "categoryId": 3,
            "featureId": 10,
            "recommendedActions": [
                {
                    "name": "비밀번호 암호화 처리",
                    "importance": 3,
                    "startDate": "2024-06-01T00:00:00",
                    "endDate": "2024-06-02T00:00:00"
                },
                {
                    "name": "에러 메시지 예외 처리",
                    "importance": 2,
                    "startDate": "2024-06-02T00:00:00",
                    "endDate": "2024-06-03T00:00:00"
                }
            ]
        }

        return RecommendationResponse(
            result="success",
            recommendedActions=sample_response["recommendedActions"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 생성 오류: {str(e)}")