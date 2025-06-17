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

# 최적화된 시스템 프롬프트 (각 배열당 1개 객체만)
OPTIMIZED_SYSTEM_PROMPT = """
당신은 프로젝트 API 명세서를 작성하는 전문가입니다.

**절대 금지 사항:**
- 백슬래시(\\) 문자를 절대로 사용하지 마세요
- JSON 중첩 배열 구조를 잘못 작성하지 마세요
- 배열 내부에서 잘못된 중첩 구조를 만들지 마세요
- 반드시 올바른 JSON 형식을 유지하세요
**JSON 구조 규칙:**
1. apiSpecifications는 단일 배열이어야 합니다
2. 각 API는 배열의 개별 객체여야 합니다
3. request 배열에는 객체 1개만 포함
4. response 배열에는 객체 1개만 포함
5. data 배열에는 객체 1개만 포함
6. 배열 구분자는 쉼표(,)만 사용하세요
**응답 형식:**
반드시 아래 정확한 형식을 따르세요:
{
  "apiSpecifications": [
    {
      "title": "API명",
      "tag": "태그",
      "path": "/경로",
      "http_method": "post/get/put/delete",
      "request": [
        {
          "field": "필드명",
          "type": "타입",
          "example": "예시"
        }
      ],
      "response": [
        {
          "status_code": "상태코드",
          "message": "메시지",
          "data": [
            {
              "field": "필드명",
              "type": "타입",
              "example": "예시"
            }
          ]
        }
      ]
    }
  ]
}
**중요: 응답은 반드시 순수한 JSON 형태로만 제공하세요.**
"""

@router.post("/json_API/generate", response_model=APIResponse)
async def generate_project_json(request: APIRequest):
  # 간소화된 프롬프트 (단일 문자열 데이터)
  enhanced_prompt = f"""
  프로젝트 데이터: {request.project_overview}
  요구사항 데이터: {request.requirements}
  프로젝트 요약 데이터: {request.project_summury}
  **절대 준수 사항:**
  1. 백슬래시(\\) 문자 사용 금지
  2. 올바른 JSON 배열 구조 유지
  3. 각 배열당 객체 1개씩만 포함
  4. apiSpecifications는 단일 배열로 구성
  **API 생성 요구사항:**
  - 최소 15개 이상의 API 명세 작성
  - 각 엔티티별 CRUD 작업 포함
  - 인증, 파일처리, 통계, 알림 등 실무 API 포함
  - http_method는 소문자로 작성
  - request 배열에는 1개 객체만, response 배열에는 1개 객체만, data 배열에는 1개 객체만
  **JSON 구조 예시:**
  {{
    "apiSpecifications": [
      {{
        "title": "사용자 등록",
        "tag": "사용자",
        "path": "/api/users/register",
        "http_method": "post",
        "request": [
          {{
            "field": "username",
            "type": "string",
            "example": "john_doe"
          }}
        ],
        "response": [
          {{
            "status_code": "201",
            "message": "사용자 등록 성공",
            "data": [
              {{
                "field": "userId",
                "type": "integer",
                "example": 1
              }}
            ]
          }}
        ]
      }},
      {{
        "title": "사용자 로그인",
        "tag": "사용자",
        "path": "/api/users/login",
        "http_method": "post",
        "request": [
          {{
            "field": "email",
            "type": "string",
            "example": "john@example.com"
          }}
        ],
        "response": [
          {{
            "status_code": "200",
            "message": "로그인 성공",
            "data": [
              {{
                "field": "token",
                "type": "string",
                "example": "jwt.token.here"
              }}
            ]
          }}
        ]
      }}
    ]
  }}
  **중요한 제약사항:**
  - request 배열: 반드시 1개 객체만 포함 (가장 중요한 필드 1개만)
  - response 배열: 반드시 1개 객체만 포함
  - data 배열: 반드시 1개 객체만 포함 (가장 중요한 응답 필드 1개만)
  위 형식을 정확히 지켜서 프로젝트에 적합한 API를 최대한 많이, 상세하게 작성하세요!
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