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
# 백슬래시 완전 제거 최적화 시스템 프롬프트
OPTIMIZED_SYSTEM_PROMPT = """
당신은 프로젝트 API 명세서를 작성하는 전문가입니다.

**절대 절대 절대 금지 사항**
- 백슬래시 문자를 어떤 경우에도 사용하지 마세요
- 이스케이프 문자나 특수문자 처리에 백슬래시 사용 금지
- JSON 문자열 내에서도 백슬래시 사용 절대 금지
- example 필드에서 백슬래시 사용 절대 금지

**JSON 구조 규칙:**
1. apiSpecifications는 단일 배열
2. 각 API는 배열의 개별 객체
3. request 배열에는 객체 1개만 포함
4. response 배열에는 객체 1개만 포함
5. data 배열에는 객체 1개만 포함

**응답 형식:**
{
 "apiSpecifications": [
   {
     "title": "API명",
     "tag": "태그",
     "path": "/경로",
     "http_method": "post",
     "request": [
       {
         "field": "필드명",
         "type": "타입",
         "example": "예시값"
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
             "example": "예시값"
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
  # 백슬래시 완전 제거 프롬프트
  enhanced_prompt = f"""
  프로젝트 데이터: {request.project_overview}
  요구사항 데이터: {request.requirements}
  프로젝트 요약 데이터: {request.project_summury}

  **백슬래시 사용 절대 금지**
  - 백슬래시 문자를 절대로 사용하지 마세요
  - 모든 예시값은 백슬래시 없이 작성하세요
  - JSON 이스케이프 문자 사용 금지
  - 특수문자 처리에도 백슬래시 사용 금지

  **API 생성 요구사항:**
  - 최소 15개 이상의 API 명세 작성
  - 각 엔티티별 CRUD 작업 포함
  - 인증, 파일처리, 통계, 알림 등 실무 API 포함
  - http_method는 소문자로 작성
  - request 배열에는 1개 객체만
  - response 배열에는 1개 객체만  
  - data 배열에는 1개 객체만

  **올바른 JSON 구조 예시 (백슬래시 절대 없음):**
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
      "tag": "인증",
      "path": "/api/auth/login", 
      "http_method": "post",
      "request": [
        {{
          "field": "email",
          "type": "string",
          "example": "user@example.com"
        }}
      ],
      "response": [
        {{
          "status_code": "200",
          "message": "로그인 성공",
          "data": [
            {{
              "field": "accessToken",
              "type": "string", 
              "example": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            }}
          ]
        }}
      ]
    }},
    {{
      "title": "상품 목록 조회",
      "tag": "상품",
      "path": "/api/products",
      "http_method": "get",
      "request": [
        {{
          "field": "page",
          "type": "integer",
          "example": 1
        }}
      ],
      "response": [
        {{
          "status_code": "200", 
          "message": "상품 목록 조회 성공",
          "data": [
            {{
              "field": "products",
              "type": "array",
              "example": "상품배열데이터"
            }}
          ]
        }}
      ]
    }}
  ]
  }}

  **핵심 제약사항:**
  - 백슬래시 문자 사용 절대 금지
  - request 배열: 반드시 1개 객체만 (핵심 필드 1개)
  - response 배열: 반드시 1개 객체만
  - data 배열: 반드시 1개 객체만 (핵심 응답 필드 1개)
  - 모든 example 값은 간단하고 명확하게 작성
  - JSON 문법 오류 절대 금지

  위 형식을 정확히 지켜서 프로젝트에 적합한 API를 최대한 많이 작성하세요!
  백슬래시가 포함된 응답은 절대 허용되지 않습니다!
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
          raise ValueError("생성된 결과물의 파싱에 실패했습니다.")
      
      # 구조 검증
      if not validate_json_structure(json_data):
          raise ValueError("생성된 결과물의 구조가 올바르지 않습니다")  
      
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
      raise HTTPException(status_code=500, detail=f"JSON 파싱 오류: {str(e)}\n응답: {content}")
  except Exception as e:
      raise HTTPException(status_code=500, detail=f"json 오류: {str(e)}\n응답: {content}")