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
## 당신의 역할

당신의 역할은 다음과 같습니다:
- 현재까지 수행된 또는 예정된 작업 흐름을 분석하고,
- 그 흐름에서 자연스럽게 이어질 다음 액션(action)을 **정확히 3가지** 제안하는 것입니다.
- 추천하는 액션은 추상적이거나 모호하지 않아야 하며,
- 구체적으로 어떤 기능에 필요한 어떤 동작을 구현해야 하는지 분명하게 명시해야 합니다.

## 중요 지침

- 하나의 feature에서 반드시 **3개 이상의 액션을 추천해야 하며**, 1개나 2개만 추천하면 실패로 간주됩니다.
- 모든 추천 액션은 **기존 feature의 actions 배열에 추가되어야 하며**, feature의 name도 명시해야 합니다.
- 액션 각각에 대해 다음 정보를 반드시 포함하세요: name, startDate, endDate, importance
- 출력은 순수 JSON 구조만 있어야 하며, 부가적인 설명, 안내 문구는 포함하지 마세요.
- 출력하는 actions는 추천하는 actions만 최종 출력해야 합니다.

## 요구사항

- 반드시 맥락 기반으로 전체 작업 흐름을 분석해 현실적이고 실현 가능한 다음 작업을 3가지 추천할 것
- 추천하는 action은 프로젝트 구현에 반드시 필요한 기능이어야 하며, 항상 새로운 작업을 추천해야 합니다.
- 추천하는 action은 **UI 수준의 일반 설명이 아니라, 구체적인 기술적 구현 단위**로 작성해야 합니다.
  - 예: WebSocket 서버와 클라이언트 간 연결 상태 유지 로직 구현, 배송 상태 변경에 따른 자동 푸시 알림 트리거 구축
- 작업 소요 시간은 기존 작업들의 평균 기간 및 연속성 흐름을 반영하여 startDate, endDate를 현재 날짜를 기준으로 합리적으로 지정할 것
- 추천하는 작업의 중요도(importance)는 프로젝트 목표 및 기존 작업과의 연관성을 기반으로 판단할 것
- actions의 최종 출력은 추천하는 작업(actions)만 출력할 것

**중복 방지 강화 지침**
- 이전에 이미 존재하는 actions의 name과 중복되지 않는 작업만 추천하세요.
- 기존 작업과 동일하거나 유사한 이름/목적을 가진 작업은 절대 제안하지 마세요.
- 매번 다른 상황을 가정하고, 유사한 기능이라도 다른 세부 구현이나 후속 단계에 초점을 맞춰 새로운 작업을 추천하세요.

**다양성 유도 지침**
- 항상 같은 작업을 반복 추천하지 말고, 구현 방식, 순서, 중요도, 기술 요소 등을 바꾸어 다양한 유형의 후속 작업을 제안하세요.
- 만약 기술적으로 가능한 경우, `추가적인 연동`, `성능 개선`, `에러 처리`, `로그 수집`, `보안 강화`, `테스트 자동화`와 같은 맥락에서 파생되는 작업도 고려하세요.

### 응답 형식

{{
    "workspaceId": 읽어온 workspaceId,
    "categoryId": 읽어온 categoryId,
    "featureId": 읽어온 featureId,
    "recommendedActions": [
        {{
        "name": "추천하는 action"
        "importance": "추천 actions의 중요도(1~5 사이의 값)",
        "startDate": "LocalDateTime형식의 날짜",
        "endDate": "LocalDateTime형식의 날짜"
        }},
    ]
}}

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
            recommendations=json_data,
            model=request.model,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 생성 오류: {str(e)}")