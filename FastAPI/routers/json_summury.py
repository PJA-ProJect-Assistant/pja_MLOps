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

### 핵심 역할
사용자가 입력한 프로젝트 설명을 바탕으로 유효한 경우에만 상세한 기획 JSON을 생성하십시오. 단, 아래 조건 중 하나라도 충족하지 않으면 JSON을 생성하지 마십시오.

---

### 유효성 검사 조건
아래 조건 중 하나라도 해당되면 **무조건 다음 메시지만 출력**하십시오:


{"error": "입력하신 설명이 유효하지 않습니다. 구체적인 서비스 목적, 기능, 문제 등을 포함한 자연어 문장으로 작성해 주세요."}

입력이 유효하지 않은 경우 (반드시 거절)
문법적으로 자연스러우나 내용이 허위/공허/의미 없음:

예: "하이퍼모듈 기반의 커넥톰", "메타 상호작용 루프", "트랜스포머형 트랜지션", "뉴로-뉴럴 메타프레임" 등 실제 존재하지 않거나 맥락 없는 기술 용어 조합

과도하게 기술적으로 보이나 실제 제품 구조나 서비스 플로우를 설명하지 않음

임의의 반복 문자 또는 유사 문자 패턴:

예: "ㅍㅍㅍ", "aaa", "1111", "......" 등

단어는 존재하나 의미 연결이 없는 문장:

예: "UX 기반 플럭스 매핑을 통해 사용자 커넥트를 트랜지션 시킴"

자연어가 아닌 나열식, 코드 스니펫, 태그, 한 단어만 있는 경우

설명 길이가 100자 미만이며 서비스 목적, 문제, 타겟, 기능 중 하나도 포함되지 않은 경우

유효한 입력일 경우에만 아래 구조로 JSON 생성
형식은 아래와 같으며, 각 필드는 반드시 구체적이고 논리적인 내용을 포함해야 합니다.

{
  "project_info": {
    "title": "매력적이고 기억하기 쉬운 프로젝트명",
    "category": "웹앱/모바일앱/데스크톱/게임/AI서비스/IoT/기타 중 선택",
    "target_users": ["구체적인 타겟유저1", "타겟유저2", "잠재적유저3"],
    "core_features": ["핵심기능1", "핵심기능2", "부가기능1", "차별화기능"],
    "technology_stack": ["프론트엔드", "백엔드", "데이터베이스", "클라우드/인프라", "기타도구"],
    "problem_solving": {
      "current_problem": "시장에서 발견되는 구체적이고 명확한 문제점",
      "solution_idea": "기술적으로 실현 가능한 구체적인 해결 방안. 사용자 경험과 구현 흐름, 기존 대안과의 차이점 포함하여 서술",
      "expected_benefits": ["사용자 혜택", "비즈니스 가치", "사회적 영향"]
    }
  }
}
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