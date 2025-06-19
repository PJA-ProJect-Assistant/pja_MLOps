import dotenv
import os
import openai
import json
import os

# .env 파일 로드 (있다면)
dotenv.load_dotenv()

# 환경변수 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# 환경변수 설정하기 싫으면 주석 삭제
# OPENAI_API_KEY = "YOUR_OPEN_API_KEY"

# OpenAI 클라이언트 설정
client = openai.OpenAI()

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

"""

# 파인튜닝된 모델 ID
MODEL_ID = "ft:gpt-4o-mini-2024-07-18:test::BebIPMSD"

def analyze_project(project_description):
    """프로젝트 분석 함수"""
    
    # JSON 템플릿을 별도 변수로 정의 (f-string 문제 해결)
    json_template = '''
  {
  "project_summury": {
    "title": "프로젝트 제목",
    "category": "카테고리",
    "target_users": [
      "대상 사용자 1",
      "대상 사용자 2"
    ],
    "core_features": [
      "핵심 기능 1",
      "핵심 기능 2"
    ],
    "technology_stack": [
      "기술 스택 1",
      "기술 스택 2"
    ],
    "problem_solving": {
      "current_problem": "현재 문제",
      "solution_idea": "해결 아이디어",
      "expected_benefits": [
        "예상 효과 1",
        "예상 효과 2"
      ]
    }
  }
  "erd_data": {
    "erd_table": [
      {
        "name": "테이블명",
        "erd_column": [
          {
            "name": "컬럼명",
            "data_type": "데이터타입",
            "is_primary_key": true,
            "is_foreign_key": false,
            "is_nullable": false
          }
        ]
      }
    ],
    "erd_relationships": [
      {
        "from_erd_table_id": "시작테이블",
        "to_erd_table_id": "끝테이블",
        "type": "관계타입",
        "foreign_key": "외래키명",
        "constraint_name": "제약조건명"
      }
    ]
  }
}
'''
    
    # 구조화된 요청 프롬프트 (f-string 사용하지 않음)
    enhanced_prompt = project_description + f"""

다음 형식으로 체계적인 분석을 제공해주세요:

1. **프로젝트 상세 정보**
2. **ERD 데이터**
3. **관계 데이터** 
4. **API 명세 데이터**

{json_template}

위 1,2,3번의 내용은 JSON 형식에 정확히 맞춰서 분석 결과를 제공해주세요.
실무에서 바로 활용 가능한 구체적이고 완전한 분석을 부탁드립니다."""

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"오류 발생: {e}"
    
# my_project에 개요 작성
my_project = "이 프로젝트는 3인칭 카드게임기반 MMORPG 게임 개발 프로젝트입니다. 창업을 목적으로 게임 데이터를 끌어다가 프로젝트를 하려고합니다. 주제는 간단하게 3인칭 카드게임을 mmorpg형식으로 만드려고 합니다. 전체적으로 어떻게 만들생각이냐면 시간이 10초 지날 때마다 하나씩 카드가 드롭되게 하는 형식으로 게임을 만들고싶습니다. 또한, 스토리가 있었으면 좋겠고 선택하는 스토리라인에 따라서 드롭되는 카드의 형식이 달랐으면 좋겠습니다. 이 게임을 통해서 사용자가 랜덤 가챠 + 순간적인 판단으로 컨트롤하는 능력이 늘었으면 좋겠다는 생각으로 프로젝트를 기획하였습니다."
response = analyze_project(my_project)
print(response)