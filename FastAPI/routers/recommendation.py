# routers/recommendation.py
from fastapi import APIRouter, HTTPException
import openai
import os
from dotenv import load_dotenv
from models.requests import RecommendationRequest
from models.response import RecommendationResponse
import json
from datetime import datetime, timedelta

# 환경변수 로드
load_dotenv()

router = APIRouter()

# OpenAI 클라이언트 초기화
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 프롬프트에 현재 날짜/시각 전달(startDate 기준날짜 설정 목적)
NOW = datetime.now().isoformat()

# 최적화된 시스템 프롬프트
OPTIMIZED_SYSTEM_PROMPT = """
## 당신의 역할

당신은 현재까지 수행된 작업 흐름을 분석하고, 그 흐름에 따라 다음으로 수행되어야 할 액션(action)을 정확히 3가지 제안해야 합니다.

추천하는 액션은 다음 조건을 모두 충족해야 합니다:

- 사용자 인터페이스 수준의 일반적인 설명이 아니라, 실제 개발자가 바로 구현할 수 있을 정도로 구체적인 기술 단위로 작성해야 합니다.  
  - 예: "알림 기능 추가"(부적절) → "WebSocket 연결 상태 유지 로직 구현"(적절)  
        "배송 기능 구현"(부적절) → "배송 상태 변경 시 푸시 알림 트리거 구축"(적절)

- 액션의 목적과 범위를 명확하게 기술하고, 명세서를 기반으로 바로 작업을 시작할 수 있는 수준으로 구체화해야 합니다.

## 중요 지침

- 반드시 하나의 feature에 대해 정확히 3개의 액션을 추천할 것
- 모든 추천 액션은 해당 feature의 actions 배열에 추가 가능한 형태여야 하며, feature의 name도 함께 명시할 것
- 각 액션에는 다음 항목을 포함해야 함: name, startDate, endDate, importance
- 출력은 JSON 구조만 포함되어야 하며, 설명, 안내 문구, 예외적 출력은 절대 포함하지 말 것
- startDate와 endDate는 반드시 현실적인 시간 흐름을 반영해야 하며, 아래 조건을 엄격히 만족해야 합니다:
  - startDate는 반드시 현재 시각({NOW}) 이후여야 합니다.
  - startDate는 현재 시각보다 과거일 수 없습니다. (현재 시각 이후여야 합니다.)
  - endDate는 반드시 startDate 이후의 시점이어야 합니다.

## 요구사항

- 기존 작업 흐름의 맥락을 기반으로, 실현 가능하고 현실적인 후속 작업을 제안할 것
- 반드시 새로운 작업만 추천해야 하며, 기존에 존재하는 작업과 동일하거나 유사한 이름, 목적의 작업은 추천하지 말 것
- 구현 방식, 목적, 시기, 중요도, 기술 스택 등을 기준으로 매번 다른 작업을 생성할 것
- 다음과 같은 관점에서 작업을 다양화할 수 있음: 외부 연동, 성능 최적화, 장애 대응, 예외 처리, 로그 수집, 보안 강화, 테스트 자동화

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
""".format(NOW=NOW)

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
            temperature=0.5
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