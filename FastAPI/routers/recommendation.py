# routers/recommendation.py
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
- 제공된 작업 공간과 카테고리 정보를 기반으로 다음 작업을 추천
- 각 작업의 중요도 평가 및 일정 계획 수립
- 프로젝트의 효율적인 진행을 위한 작업 우선순위 결정

## 응답 형식:
추천 작업은 다음 JSON 형식으로 제공되어야 합니다:

{
  "workspaceId": 1,
  "categoryId": 3,
  "featureId": 10,
  "recommendedActions": [
    {
      "name": "비밀번호 암호화 처리"
      "importance": 3,
      "startDate": LocalDateTime형식의 날짜,
      "endDate": LocalDateTime형식의 날짜
    },
    {
      "name": "에러 메시지 예외 처리"
      "importance": 3,      
      "startDate": LocalDateTime형식의 날짜,
      "endDate": LocalDateTime형식의 날짜
    }
  ]
}

**중요: 응답은 반드시 순수한 JSON 형태로만 제공하세요.**
"""

@router.post("/recommend/generate", response_model=RecommendationResponse)
async def recommendation(request: RecommendationRequest):
    try:
        # 프롬프트 구성
        enhanced_prompt = f"""
        팀 프로젝트 작업 리스트 : {request.project_list}
        위 정보를 바탕으로 다음 작업을 추천해주세요.

        **강제 준수 규칙:**
        1. 순수 JSON만 응답 (마크다운 블록 절대 금지)
        2. 마지막 요소 뒤 쉼표 절대 금지        
        """

        # OpenAI API 호출
        response = await client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # JSON 파싱
        content = response.choices[0].message.content
        json_data = json.loads(content)
        
        # 응답 반환
        return RecommendationResponse(
            recommendations=[json_data],
            model=request.model,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 생성 오류: {str(e)}")