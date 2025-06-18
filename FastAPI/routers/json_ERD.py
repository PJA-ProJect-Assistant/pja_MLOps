# json_ERDAPI.py
from fastapi import APIRouter, HTTPException
import openai
import os
from dotenv import load_dotenv
from models.requests import ERDRequest
from models.response import ERDResponse
import json
from utils.json_parsing import clean_and_parse_response, validate_json_structure


# 환경변수 로드
load_dotenv()

router = APIRouter()

# OpenAI 클라이언트 초기화
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 최적화된 시스템 프롬프트
OPTIMIZED_SYSTEM_PROMPT = """
당신은 ERD 설계 전문가입니다. 프로젝트 요구사항을 분석하여 완전한 데이터베이스 스키마를 생성합니다.
**핵심 원칙:**
- 백슬래시(\\) 절대 사용 금지
- erd_relationships의 모든 테이블과 외래키는 반드시 erd_tables에 존재해야 함
- 외래키 컬럼은 is_foreign_key: true 설정 필수
- 순수 JSON만 응답 (마크다운 블록 금지)
**설계 순서:**
1. 엔티티 식별 → 2. 속성/기본키 정의 → 3. 관계 분석/외래키 추가 → 4. 관계 정보 작성
**중요: 모든 관계의 테이블명과 외래키가 테이블 정의와 정확히 일치해야 합니다.**
"""


@router.post("/json_ERD/generate", response_model=ERDResponse)
async def generate_project_json(request: ERDRequest):   
# 개선된 프롬프트
  enhanced_prompt = f"""
  프로젝트: {request.project_overview}
  요구사항: {request.requirements}
  요약: {request.project_summury}
  **필수 JSON 형식:**
  {{
    "erd_tables": [{{
      "name": "테이블명",
      "erd_columns": [{{
        "name": "컬럼명",
        "data_type": "타입",
        "is_primary_key": true/false,
        "is_foreign_key": true/false,
        "is_nullable": true/false
      }}]
    }}],
    "erd_relationships": [{{
      "from_table": "시작테이블",
      "to_table": "끝테이블",
      "relationship_type": "one-to-many",
      "foreign_key": "외래키명",
      "constraint_name": "제약조건명"
    }}]
  }}
  **핵심 규칙:**
  1. 관계의 모든 테이블명이 erd_tables에 존재해야 함
  2. 관계의 모든 foreign_key가 해당 테이블 컬럼에 존재해야 함
  3. 외래키는 is_foreign_key: true 설정
  4. 최소 5개 테이블, 백슬래시 금지, 순수 JSON만
  위 규칙을 지켜 완전한 ERD를 생성하세요!
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
      
      return ERDResponse(
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