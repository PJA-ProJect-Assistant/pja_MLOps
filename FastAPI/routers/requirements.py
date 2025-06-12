# routers/requirements.py
from fastapi import APIRouter, HTTPException
import openai
import os
from dotenv import load_dotenv
from models.requests import RequirementsRequest
from models.response import RequirementsResponse
import json

# 환경변수 로드
load_dotenv()

router = APIRouter()

# OpenAI 클라이언트 초기화
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 최적화된 시스템 프롬프트
OPTIMIZED_SYSTEM_PROMPT = """
당신은 소프트웨어 프로젝트의 요구사항 분석 전문가입니다.
주어진 프로젝트 개요를 바탕으로 현실적이고 구체적인 기능 요구사항과 성능 요구사항을 생성하는 것이 목표입니다.

생성 규칙:
1. 기능 요구사항(FUNCTIONAL): 사용자 관점에서 시스템이 제공해야 하는 구체적인 기능
2. 성능 요구사항(PERFORMANCE): 시스템의 응답시간, 처리량, 가용성, 확장성 등 측정 가능한 수치 포함
3. 각 요구사항은 명확하고 구현 가능하며 테스트 가능해야 함
4. 프로젝트 규모와 성격에 적합한 현실적인 수준이어야 함
5. 반드시 지정된 JSON 형식으로만 응답할 것
"""

@router.post("/requirements/generate", response_model=RequirementsResponse)
async def generate_requirements(request: RequirementsRequest):
    """프로젝트 요구사항을 생성하는 엔드포인트"""
    
    enhanced_prompt = f"""
    프로젝트 개요:
    {request.project_overview}

    현재 기존 요구사항 목록:
    {request.existing_requirements}

    위 기존 요구사항들을 분석하여 추가로 {request.additional_count}개의 새로운 요구사항을 생성해주세요.

    **생성 조건:**
    - 기존 요구사항들과 일관성을 유지하되, 중복되지 않는 새로운 관점의 요구사항
    - 프로젝트의 도메인과 맥락에 적합한 현실적이고 구현 가능한 요구사항
    - 기능 요구사항(FUNCTIONAL)과 성능 요구사항(PERFORMANCE)을 적절히 조합
    - 각 요구사항은 구체적이고 측정 가능하며 테스트 가능해야 함

    **중요: 신규 요구사항만 생성하세요. 기존 요구사항은 포함하지 마세요.**

    **응답 형식 (JSON만):**
    반드시 아래 JSON 형식으로만 응답하고, 다른 설명이나 주석은 절대 포함하지 마세요.

    [
    {{"requirementType": "FUNCTIONAL", "content": "구체적인 기능 요구사항 설명"}},
    {{"requirementType": "FUNCTIONAL", "content": "구체적인 기능 요구사항 설명"}},
    {{"requirementType": "PERFORMANCE", "content": "구체적인 성능 요구사항 설명 (수치 포함)"}},
    {{"requirementType": "PERFORMANCE", "content": "구체적인 성능 요구사항 설명 (수치 포함)"}}
    ]

    주의: 
    1. 새로 생성된 {request.additional_count}개의 요구사항만 출력
    2. JSON 배열로만 시작하고 끝나야 하며, 배열 외부에 어떤 텍스트도 포함하지 마세요
    3. 기존 요구사항과 중복되지 않는 완전히 새로운 요구사항만 작성
    4. 기능 요구사항을 먼저, 성능 요구사항을 나중에 배치
    """
    
    try:
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
        requirements_data = json.loads(content)
        
        # 토큰 사용량 정보
        usage = response.usage
        
        return RequirementsResponse(
            requirements=requirements_data,
            model=request.model,
            total_tokens=usage.total_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요구사항 생성 오류: {str(e)}")