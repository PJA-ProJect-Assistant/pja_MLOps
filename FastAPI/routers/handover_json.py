# routers/handover_json.py
from fastapi import APIRouter, HTTPException
import openai
import os
from dotenv import load_dotenv
from models.requests import jsonRequest
from models.response import jsonResponse
import json

# 환경변수 로드
load_dotenv()

router = APIRouter()

# OpenAI 클라이언트 초기화
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 최적화된 시스템 프롬프트
OPTIMIZED_SYSTEM_PROMPT = """
당신은 프로젝트 아이디어를 체계적으로 분석하고 구조화하여 구체적인 개발 계획을 제시하는 전문 AI 어시스턴트입니다.

**만약 프로젝트 아이디어가 아닌 다른 질문을 입력 받으면 아래의 해당 방식이 간단한 방식으로로 답변해주세요.**

## 주요 역할과 능력:

### 1. 프로젝트 분석 전문가
- 사용자가 제공하는 프로젝트 아이디어나 설명을 깊이 있게 분석합니다
- 핵심 기능, 대상 사용자, 기술 스택, 비즈니스 모델 등을 체계적으로 파악합니다
- 프로젝트의 문제 해결 방향과 기대 효과를 명확히 정의합니다

### 2. 데이터베이스 설계 전문가
- 프로젝트 요구사항을 바탕으로 최적화된 ERD(Entity Relationship Diagram)를 설계합니다
- 테이블 간의 관계, 외래키 제약조건, 데이터 타입을 정확히 정의합니다
- 확장성과 성능을 고려한 데이터베이스 구조를 제안합니다

### 3. API 설계 전문가
- RESTful API 원칙에 따라 체계적인 API 명세를 작성합니다
- OpenAPI(Swagger) 3.0 표준을 준수하여 완전한 API 문서를 생성합니다
- 각 엔드포인트별 요청/응답 스키마, 에러 처리, 인증 방식을 상세히 정의합니다

## 응답 형식:
모든 응답은 다음과 같은 구조화된 형태로 제공해야 합니다:

1. **프로젝트 상세 정보**: 제목, 카테고리, 대상 사용자, 핵심 기능, 기술 스택, 문제 해결 방안 등을 포함한 종합 분석
2. **관계 데이터**: 데이터베이스 테이블 간의 관계와 외래키 제약조건 정의
3. **ERD 데이터**: 각 테이블의 속성, 데이터 타입, 키 정보를 포함한 완전한 스키마
4. **API 명세 데이터**: OpenAPI 3.0 표준을 준수한 완전한 API 문서

항상 체계적이고 전문적인 관점에서 프로젝트를 분석하며, 개발팀이 바로 실행에 옮길 수 있는 구체적인 가이드를 제공하는 것이 목표입니다.

**중요: 응답은 반드시 순수한 JSON 형태로만 제공하세요. 마크다운 코드 블록(```json)이나 기타 텍스트 포맷팅은 사용하지 마세요.**

"""

@router.post("/json_text/generate", response_model=jsonResponse)
async def generate_project_json(request: jsonRequest):   
# 간소화된 프롬프트
  enhanced_prompt = f"""
  프로젝트 데이터: {request.project_overview}
  요구사항 데이터: {request.requirements}

  **중요: 유효한 JSON만 생성하세요. 마지막 요소 뒤에 쉼표를 절대 붙이지 마세요.**

  아래 JSON 형식으로 응답하세요:

  {{
    "project_summary": {{
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
    }},
    "erd_tables": [{{
      "name": "테이블명",
      "erd_columns": [{{
        "name": "컬럼명",
        "data_type": "타입",
        "is_primary_key": true,
        "is_foreign_key": false,
        "is_nullable": true
      }}]
    }}],
    "erd_relationships": [{{
      "from_table": "시작테이블",
      "to_table": "끝테이블",
      "relationship_type": "관계타입",
      "foreign_key": "외래키명",
      "constraint_name": "제약조건명"
    }}],
    "apiSpecifications": [{{
      "title": "API명",
      "tag": "태그",
      "path": "/경로",
      "http_method": "POST",
      "request": [{{
        "field": "필드명",
        "type": "string",
        "example": "예시값"
      }}],
      "response": {{
        "success": {{
          "status_code": "200",
          "message": "성공 메시지",
          "data": {{
            "field1": "값1",
            "field2": "값2",
            "field3": ["배열값1", "배열값2"]
          }}
        }},
        "error": {{
          "status_code": "400",
          "message": "에러 메시지",
          "data": null
        }}
      }}
    }}]
  }}

  **필수 요구사항:**
  1. 각 엔티티별 최소 5개 API (CRUD + 검색 + 추가기능)
  2. 최소 15개 이상 총 API 명세 작성
  3. 인증, 파일처리, 통계, 알림 등 실무 API 포함
  4. 순수 JSON만 응답 (마크다운 블록 금지)
  5. 완전한 request/response 스키마 작성
  6. response.data는 단일 객체로 구성하고, 배열이 필요한 경우 객체 내부의 속성으로 배열 사용

  **response 구조 예시:**
  - 단일 데이터: {{"data": {{"id": 1, "name": "example"}}}}
  - 리스트 데이터: {{"data": {{"items": [{{"id": 1}}, {{"id": 2}}], "total": 2}}}}
  - 페이징 데이터: {{"data": {{"items": [], "page": 1, "limit": 10, "total": 100}}}}

  API를 최대한 많이, 상세하게 작성하세요!
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
      json_data = json.loads(content)
      
      # 토큰 사용량 정보
      usage = response.usage
      
      return jsonResponse(
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