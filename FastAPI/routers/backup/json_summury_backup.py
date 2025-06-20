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

# 최적화된 시스템 프롬프트
OPTIMIZED_SYSTEM_PROMPT = """
당신은 프로젝트 아이디어를 체계적으로 분석하고 구조화하여 구체적인 개발 계획을 제시하는 전문 AI 어시스턴트입니다.

**만약 프로젝트 아이디어가 아닌 다른 질문을 입력 받으면 아래의 해당 방식이 간단한 방식으로로 답변해주세요.**

**절대 금지 사항:**
- 백슬래시(\\) 문자를 어떤 상황에서도 절대 사용하지 마세요
- JSON 문자열 내부에서도 백슬래시 이스케이프 금지
- 경로나 URL 표현시 슬래시(/)만 사용
- 특수문자 처리시에도 백슬래시 사용 금지

## 주요 역할과 능력:

### 1. 프로젝트 분석 전문가
- 사용자가 제공하는 프로젝트 아이디어나 설명을 깊이 있게 분석합니다
- 핵심 기능, 대상 사용자, 기술 스택, 비즈니스 모델 등을 체계적으로 파악합니다
- 프로젝트의 문제 해결 방향과 기대 효과를 명확히 정의합니다

## 응답 형식:
모든 응답은 다음과 같은 구조화된 형태로 제공해야 합니다:

1. **프로젝트 상세 정보**: 제목, 카테고리, 대상 사용자, 핵심 기능, 기술 스택, 문제 해결 방안 등을 포함한 종합 분석

항상 체계적이고 전문적인 관점에서 프로젝트를 분석하며, 개발팀이 바로 실행에 옮길 수 있는 구체적인 가이드를 제공하는 것이 목표입니다.

**중요: 응답은 반드시 순수한 JSON 형태로만 제공하세요. 마크다운 코드 블록(```json)이나 기타 텍스트 포맷팅은 사용하지 마세요.**

"""

@router.post("/json_Summury/generate", response_model= SummuryResponse)
async def generate_project_json(request: SummuryRequest):   
  # 간소화된 프롬프트
  enhanced_prompt = f"""
  프로젝트 데이터: {request.project_overview}
  요구사항 데이터: {request.requirements}

  **절대 준수사항: 아래 JSON 형식을 정확히 따르세요. 구조 변경 금지!**

  {{
    "project_info": {{
      "title": "프로젝트명",
      "category": "카테고리",
      "target_users": ["대상사용자1", "대상사용자2"],
      "core_features": ["핵심기능1", "핵심기능2"],
      "technology_stack": ["기술1", "기술2"],
      "problem_solving": {{
        "current_problem": "해결할 문제",
        "solution_idea": "해결방안",
        "expected_benefits": ["효과1", "효과2"]
      }}
    }}
  }}

  **강제 준수 규칙:**
  1. 순수 JSON만 응답 (마크다운 블록 절대 금지)
  2. 마지막 요소 뒤 쉼표 절대 금지
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
      json_data = clean_and_parse_response(content, response_type="dict")

      if json_data is None :
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
      raise HTTPException(status_code=500, detail=f"json 오류: {str(e)}")