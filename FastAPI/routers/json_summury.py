# routers/json_summury.py
from fastapi import APIRouter, HTTPException
import openai
import os
from dotenv import load_dotenv
from models.requests import SummuryRequest
from models.response import SummuryResponse
import json
from utils.json_parsing import clean_and_parse_response, validate_json_structure

# 환경변수 로드
load_dotenv()

router = APIRouter()

# OpenAI 클라이언트 초기화
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 개선된 시스템 프롬프트 - 내용 증강 버전
OPTIMIZED_SYSTEM_PROMPT = """
당신은 간단한 프로젝트 아이디어를 받아 전문적이고 구체적인 프로젝트 계획으로 확장하는 전문 컨설턴트입니다.

## 핵심 역할
사용자가 제공한 간략한 프로젝트 개요(보통 200자 내외)를 바탕으로:
1. 프로젝트의 숨겨진 잠재력과 가능성을 발굴
2. 시장 동향과 사용자 니즈를 고려한 기능 확장
3. 실현 가능한 기술 스택과 구현 방안 제시
4. 비즈니스 관점에서의 가치와 차별점 도출

## 증강 분석 프로세스
1. **컨텍스트 추론**: 간단한 설명에서 숨겨진 의도와 목표 파악
2. **시장 분석**: 유사 서비스 분석을 통한 차별화 포인트 발굴
3. **기능 확장**: 핵심 기능 외 필요한 부가 기능들 추론
4. **기술 매칭**: 프로젝트 성격에 가장 적합한 기술 스택 선정
5. **사용자 페르소나**: 구체적이고 현실적인 타겟 사용자 정의

## 응답 규칙
1. 반드시 유효한 JSON 형식으로만 응답
2. 백슬래시(\) 문자 사용 금지
3. 사용자 입력이 모호해도 합리적 추론으로 구체화
4. 실현 가능성과 시장성을 고려한 현실적 제안
5. 각 필드는 구체적이고 실행 가능한 내용으로 작성

## 필수 JSON 구조
{
  "project_info": {
    "title": "매력적이고 기억하기 쉬운 프로젝트명",
    "category": "웹앱/모바일앱/데스크톱/게임/AI서비스/IoT/기타 중 선택",
    "target_users": ["구체적인 타겟유저1", "타겟유저2", "잠재적유저3"],
    "core_features": ["핵심기능1", "핵심기능2", "부가기능1", "차별화기능"],
    "technology_stack": ["프론트엔드", "백엔드", "데이터베이스", "클라우드/인프라", "기타도구"],
    "problem_solving": {
      "current_problem": "시장에서 발견되는 구체적이고 명확한 문제점",
      "solution_idea": "프로젝트의 핵심 해결 방안을 상세히 설명한 3-5문장의 구체적인 솔루션 설명. 기술적 접근법, 구현 방식, 사용자 경험, 차별화 포인트를 포함하여 어떻게 문제를 해결할지 명확하게 서술",
      "expected_benefits": ["사용자혜택1", "비즈니스가치2", "사회적영향3"]
    }
  }
}

## 증강 원칙
- 간단한 입력도 풍부한 가능성으로 해석
- 현재 기술 트렌드와 시장 니즈 반영
- 실제 개발 가능한 현실적 범위 내에서 제안
- 사용자의 의도를 존중하되 전문적 관점으로 확장
- **solution_idea는 반드시 상세한 설명문으로 작성**: 기술적 구현 과정, 사용자 경험 개선 방법, 기존 솔루션과의 차별점을 포함한 종합적인 해결 방안 서술

비프로젝트 질문의 경우: "이 서비스는 프로젝트 아이디어 분석 및 기획 전용입니다"
"""

@router.post("/json_Summury/generate", response_model=SummuryResponse)
async def generate_project_json(request: SummuryRequest):   
    # 내용 증강 중심의 사용자 프롬프트
    user_prompt = f"""
다음은 사용자가 간략히 작성한 프로젝트 아이디어입니다:

📝 프로젝트 개요: {request.project_overview}
📋 추가 요구사항: {request.requirements if request.requirements else "특별한 요구사항 없음"}

## 분석 및 증강 요청사항:

당신의 전문적 관점에서 이 간단한 아이디어를 다음과 같이 확장해주세요:

### 1. 프로젝트 해석 및 확장
- 사용자의 핵심 의도 파악
- 숨겨진 가능성과 잠재적 기능 발굴
- 시장에서의 포지셔닝과 차별점 도출

### 2. 실용적 구체화
- 실제 구현 가능한 기술 스택 추천
- 명확하고 구체적인 타겟 사용자 정의
- 개발 우선순위를 고려한 기능 리스트

### 3. 비즈니스 관점 보강
- 해결하는 문제의 시장 가치 분석
- **솔루션 아이디어**: 기술적 구현 방법, 사용자 경험 설계, 차별화 포인트를 포함한 3-5문장의 상세한 해결 방안 서술
- 경쟁 우위와 성장 가능성 평가

**솔루션 작성 가이드**:
- 단순한 한 줄 설명이 아닌 구체적인 해결 과정과 방법론 설명
- 기술적 접근법과 사용자 관점의 가치 제공 방식을 모두 포함
- 왜 이 방법이 효과적인지에 대한 논리적 근거 제시

**중요**: 사용자가 간단히 작성했더라도, 그 속에 담긴 가능성을 최대한 발굴하여 완성도 높은 프로젝트 계획으로 발전시켜 주세요. 단, 사용자의 원래 의도는 반드시 존중해야 합니다.

위에서 정의한 JSON 구조에 맞춰 전문적이고 구체적인 분석 결과를 제공해주세요.
"""

    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            response_format={"type": "json_object"}  # JSON 형식 강제
        )
        
        # JSON 파싱
        content = response.choices[0].message.content
        json_data = clean_and_parse_response(content, response_type="dict")

        if json_data is None:
            raise ValueError("요구사항 파싱에 실패했습니다.")
        
        # 구조 검증
        if not validate_json_structure(json_data):
            raise ValueError("생성된 요구사항의 구조가 올바르지 않습니다")    
      
        # 토큰 사용량 정보
        usage = response.usage
        
        return SummuryResponse(
            json=json_data,
            model=request.model,
            total_tokens=usage.total_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 오류: {str(e)}")