# routers/requirements.py
from fastapi import APIRouter, HTTPException
import openai
import os
from dotenv import load_dotenv
from models.requests import RequirementsRequest
from models.response import RequirementsResponse
import json
from utils.json_parsing import clean_and_parse_response, validate_json_structure

# 환경변수 로드
load_dotenv()
router = APIRouter()

# OpenAI 클라이언트 초기화
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 개선된 시스템 프롬프트
ENHANCED_SYSTEM_PROMPT = """
당신은 소프트웨어 프로젝트의 요구사항 분석 전문가입니다.
주어진 간단한 프로젝트 개요를 분석하고, 이를 바탕으로 체계적인 요구사항을 생성합니다.

## 작업 프로세스:
1. **프로젝트 개요 분석**: 주어진 개요에서 핵심 목적, 대상 사용자, 주요 기능을 파악
2. **도메인 지식 적용**: 해당 분야의 일반적인 요구사항과 베스트 프랙티스 고려
3. **요구사항 생성**: 분석 결과를 바탕으로 현실적이고 구현 가능한 요구사항 도출

## 요구사항 생성 원칙:
- **기능 요구사항(FUNCTIONAL)**: 사용자가 시스템을 통해 달성하고자 하는 구체적인 기능
- **성능 요구사항(PERFORMANCE)**: 시스템의 성능, 가용성, 확장성 등 정량적 측정 가능한 요구사항

## 품질 기준:
- 각 요구사항은 SMART 원칙을 따름 (Specific, Measurable, Achievable, Relevant, Time-bound)
- 프로젝트 규모와 복잡도에 적합한 현실적 수준
- 테스트 가능하고 검증 가능한 명확한 기준 제시
- 기존 요구사항과 일관성 유지하되 중복 방지

## 절대 준수사항:
- 백슬래시(\) 문자 사용 절대 금지
- JSON 형식 외의 모든 추가 설명이나 주석 금지
- 지정된 구조 외의 다른 필드나 형식 사용 금지
"""

# 프로젝트 개요 증강을 위한 프롬프트
PROJECT_ANALYSIS_PROMPT = """
다음 프로젝트 개요를 분석하고 증강하여 더 구체적인 정보를 도출해주세요:

## 프로젝트 개요:
{project_overview}

## 분석 관점:
1. **프로젝트 유형**: 웹 애플리케이션, 모바일 앱, 데스크톱 소프트웨어, API 서비스 등
2. **주요 사용자**: 일반 사용자, 관리자, 개발자, 기업 고객 등
3. **핵심 기능**: 사용자가 수행할 수 있는 주요 작업들
4. **기술적 특성**: 예상되는 기술 스택, 데이터 처리 방식, 보안 요구사항
5. **비즈니스 맥락**: 해결하려는 문제, 기대 효과, 경쟁 우위

위 분석을 바탕으로 프로젝트의 특성에 맞는 요구사항을 생성하세요.
"""

@router.post("/requirements/generate", response_model=RequirementsResponse)
async def generate_requirements(request: RequirementsRequest):
    """프로젝트 요구사항을 생성하는 엔드포인트"""
    
    # 1단계: 프로젝트 개요 분석 및 증강
    analysis_prompt = PROJECT_ANALYSIS_PROMPT.format(
        project_overview=request.project_overview
    )
    
    # 2단계: 증강된 정보를 바탕으로 요구사항 생성
    requirements_prompt = f"""
## 프로젝트 개요 분석:
{analysis_prompt}

## 기존 요구사항:
{request.existing_requirements}

## 생성 요청:
위 분석을 바탕으로 {request.additional_count}개의 새로운 요구사항을 생성하세요.

### 생성 조건:
1. **기존 요구사항 분석**: 현재 요구사항의 패턴과 수준을 파악하여 일관성 유지
2. **누락된 영역 식별**: 일반적으로 필요하지만 기존 요구사항에서 누락된 부분 보완
3. **프로젝트 특성 반영**: 해당 도메인의 특수한 요구사항 고려
4. **실무적 관점**: 실제 개발과 운영에서 필요한 현실적 요구사항

### 우선순위 고려사항:
- 사용자 경험(UX) 관련 기능
- 보안 및 개인정보 보호
- 성능 및 확장성
- 유지보수 및 모니터링
- 접근성 및 호환성

### 응답 형식:
반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 텍스트나 설명은 포함하지 마세요.
백슬래시(\) 문자는 절대 사용하지 마세요.

[
  {{"requirementType": "FUNCTIONAL", "content": "구체적인 기능 요구사항 (사용자 관점에서 명확한 기능 설명)"}},
  {{"requirementType": "FUNCTIONAL", "content": "구체적인 기능 요구사항 (사용자 관점에서 명확한 기능 설명)"}},
  {{"requirementType": "PERFORMANCE", "content": "구체적인 성능 요구사항 (측정 가능한 수치와 기준 포함)"}},
  {{"requirementType": "PERFORMANCE", "content": "구체적인 성능 요구사항 (측정 가능한 수치와 기준 포함)"}}
]

### 중요 사항:
- 정확히 {request.additional_count}개의 요구사항만 생성
- 기존 요구사항과 중복되지 않는 완전히 새로운 요구사항
- 기능 요구사항과 성능 요구사항을 적절히 조합
- JSON 배열 형식 외의 어떤 텍스트도 포함하지 마세요
"""

    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
                {"role": "user", "content": requirements_prompt}
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # JSON 파싱
        content = response.choices[0].message.content
        requirements_data = clean_and_parse_response(content, response_type="list")
        
        if requirements_data is None:
            raise ValueError("요구사항 파싱에 실패했습니다.")

        # 구조 검증
        if not validate_json_structure(requirements_data):
            raise ValueError("생성된 요구사항의 구조가 올바르지 않습니다")

        # 요구사항 개수 검증
        if len(requirements_data) != request.additional_count:
            raise ValueError(f"요청된 {request.additional_count}개와 다른 {len(requirements_data)}개의 요구사항이 생성되었습니다")

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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요구사항 생성 오류: {str(e)}")