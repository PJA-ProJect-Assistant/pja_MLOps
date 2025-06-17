# json_ERDAPI.py
from fastapi import APIRouter, HTTPException
import openai
import os
from dotenv import load_dotenv
from models.requests import APIRequest
from models.response import APIResponse
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

**절대 금지 사항 - 백슬래시 사용 금지:**
- 백슬래시(\\) 문자를 절대로 사용하지 마세요
- JSON 이스케이프에서도 백슬래시 금지
- 경로 표현시 슬래시(/)만 사용
- 문자열 내부의 특수문자도 백슬래시 없이 처리
- API 경로에서도 /api/users 형태로만 작성

## 주요 역할과 능력:

### 3. API 설계 전문가
- RESTful API 원칙에 따라 체계적인 API 명세를 작성합니다
- OpenAPI(Swagger) 3.0 표준을 준수하여 완전한 API 문서를 생성합니다
- 각 엔드포인트별 요청/응답 스키마, 에러 처리, 인증 방식을 상세히 정의합니다

## 응답 형식:
모든 응답은 다음과 같은 구조화된 형태로 제공해야 합니다:

1. **관계 데이터**: 데이터베이스 테이블 간의 관계와 외래키 제약조건 정의
2. **ERD 데이터**: 각 테이블의 속성, 데이터 타입, 키 정보를 포함한 완전한 스키마
3. **API 명세 데이터**: OpenAPI 3.0 표준을 준수한 완전한 API 문서

항상 체계적이고 전문적인 관점에서 프로젝트를 분석하며, 개발팀이 바로 실행에 옮길 수 있는 구체적인 가이드를 제공하는 것이 목표입니다.

**중요: 응답은 반드시 순수한 JSON 형태로만 제공하세요. 마크다운 코드 블록(```json)이나 기타 텍스트 포맷팅은 사용하지 마세요.**

"""

@router.post("/json_API/generate", response_model=APIResponse)
async def generate_project_json(request: APIRequest):   
  # 간소화된 프롬프트
  enhanced_prompt = f"""
  프로젝트 데이터: {request.project_overview}
  요구사항 데이터: {request.requirements}
  프로젝트 요약 데이터 : {request.project_summury}

    **절대 금지 사항 - 백슬래시 사용 금지:**
  - 백슬래시(\\) 문자를 절대로 사용하지 마세요
  - JSON 이스케이프에서도 백슬래시 금지
  - 경로 표현시 슬래시(/)만 사용
  - 문자열 내부의 특수문자도 백슬래시 없이 처리
  - API 경로에서도 /api/users 형태로만 작성

  {{"apiSpecifications": [{{
      "title": "API명",
      "tag": "태그",
      "path": "/경로",
      "http_method": "post/get/put/delete",
      "request": [{{
        "field": "필드명",
        "type": "타입",
        "example": "예시"
      }}],
      "response": [{{
        "status_code": "상태코드",
        "message": "메시지",
        "data": [{{
          "field": "필드명",
          "type": "타입",
          "example": "예시"
        }}]
      }}]
    }}]
  }}

  **강제 준수 규칙:**
  1. apiSpecifications 구조 절대 변경 금지
  2. request: [{{field, type, example}}] 형태 필수
  3. response: [{{status_code, message, data: [{{field, type, example}}]}}] 형태 필수
  4. data 배열 안의 각 필드는 개별 객체로 분리
  6. 최소 15개 이상 총 API 명세 작성
  7. 인증, 파일처리, 통계, 알림 등 실무 API 포함
  8. 순수 JSON만 응답 (마크다운 블록 절대 금지)
  9. 마지막 요소 뒤 쉼표 절대 금지

  **data 배열 구조 예시:**
  "data": [
    {{"field": "userId", "type": "integer", "example": 1}},
    {{"field": "username", "type": "string", "example": "john_doe"}},
    {{"field": "email", "type": "string", "example": "john@example.com"}},
    {{"field": "createdAt", "type": "datetime", "example": "2024-01-15T10:30:00Z"}}
  ]

  **백엔드 규격 엄수:**
  - request와 response는 반드시 배열 형태
  - data 안의 각 반환 필드는 {{field, type, example}} 객체로 개별 분리
  - 여러 필드 반환시 data 배열에 각각 별도 객체로 추가
  - http_method는 소문자로 작성
  - status_code는 문자열 타입

  위 형식을 정확히 지켜서 API를 최대한 많이, 상세하게 작성하세요!
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
      
      return APIResponse(
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