# routers/task_generate.py
from jinja2 import Template
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import ast
from fastapi import APIRouter, HTTPException
from models.requests import TaskGenerateRequest
from models.response import TaskGenerateResponse

# OpenAI API Key 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=openai_api_key)

router = APIRouter()

RCMD_PROMPT = """
당신은 신입/초보 개발자 팀을 이끄는 시니어 테크 리드입니다.  
다음은 한 프로젝트에 대한 주요 정보입니다.  
이 정보를 바탕으로 해당 프로젝트를 성공적으로 구현하기 위한 작업 구조를 작성해야 합니다.

---

- 프로젝트 정보 JSON:
{{ input }}

# 작업 조건:
1. Category: 패키지 수준의 상위 기능 그룹, 5개 이상  
2. Feature: 각 Category 아래 기능 모듈, 5개 이상  
3. Action: 각 Feature에 대응하는 실제 구현 단위, 3개 이상  
4. importance: 각 Actions의 기능 중요도 (1~5 사이 정수)  
5. 출력은 반드시 JSON 형식의 문자열만 반환 (설명 X, 주석 X)

---

아래의 **조건을 반드시 모두 충족한 작업 구조**를 작성하십시오.  
하나라도 누락될 경우, 출력은 무효입니다. 반드시 모든 조건을 만족해야 합니다:

1. **Category**는 5개 이상이어야 합니다.  
2. **각 Category마다 Feature는 무조건 5개 이상이어야 합니다. 예외는 없습니다.**  
3. **각 Feature에는 Action이 3개 이상 포함되어야 합니다.**  
4. **모든 Action에는 중요도를 나타내는 `importance` 값(1~5 정수)이 있어야 합니다.**
5. **반드시 JSON 형식만 출력하십시오. 주석, 설명, 텍스트, 문장 절대 금지입니다.**

## Action 작성 시 반드시 지켜야 할 추가 조건:

- 각 Action은 **UI 수준의 일반 설명이 아니라, 구체적인 기술적 구현 단위**로 작성해야 합니다.
- 프로젝트에 명시된 기술 스택 (예: `react`, `typescript`, `springboot`, `langchain`, `컴퓨터 비전`)을 **적극적으로 반영**하십시오.
- 예시:
- `"LangChain을 이용한 문맥 기반 응답 처리"`
- `"SpringBoot로 작성된 API에 JWT 인증 로직 추가"`
- `"React에서 Zustand를 활용한 상태 관리 구현"`
- `"OpenCV를 사용한 얼굴 특징점 검출"`
- Action에는 **프레임워크, 라이브러리, API, 설계 패턴 등을 명시**해도 좋습니다.
- 추상적 Action (예: "기능 구현", "UI 구성")은 금지하며, 구체적인 기술 조치로 기술하십시오.

출력은 다음 예시 형식을 엄격히 따르십시오:

{
    "workspace_id": "워크스페이스 ID",
    "recommendedCategories": [
        {
            "name": "category 이름",
            "features": [
                {
                    "name": "feature 이름",
                    "actions": [
                        { "name": "action 이름", "importance": 정수 }
                    ]
                }
            ]
        }
    ]
}
"""

@router.post("/task_generate/generate", response_model=TaskGenerateResponse)
def task_generate(request: TaskGenerateRequest):
    """
    프로젝트 정보를 바탕으로 카테고리, 기능, 액션을 생성하는 엔드포인트
    """
    try:
        print("=== 작업 생성 시작 ===")
        
        # 1. 입력 데이터 파싱
        try:
            # JSON 형식 파싱 시도
            json_data = json.loads(request.project_summary)
            print("✓ JSON 형식으로 파싱 성공")
        except json.JSONDecodeError:
            try:
                # Python 딕셔너리 형식 파싱 시도
                json_data = ast.literal_eval(request.project_summary)
                print("✓ Python 딕셔너리 형식으로 파싱 성공")
            except (ValueError, SyntaxError) as e:
                print(f"✗ 파싱 실패: {str(e)}")
                raise HTTPException(
                    status_code=422, 
                    detail=f"유효하지 않은 JSON 또는 딕셔너리 형식입니다: {str(e)}"
                )

        # 2. 데이터 구조 정규화
        if isinstance(json_data, dict): 
            json_data = [json_data]
            print("✓ 단일 객체를 리스트로 변환")

        # 3. 프로젝트 정보 추출
        def extract_project_summary(input_json):
            """프로젝트 정보에서 필요한 필드만 추출"""
            if not input_json:
                return None
                
            item = input_json[0]
            
            # 안전한 필드 추출 (기본값 제공)
            return {
                "category": item.get("category", ""),
                "core_features": item.get("core_features", []),
                "problem_solving": {
                    "solutionIdea": item.get("problem_solving", {}).get("solutionIdea", ""),
                    "currentProblem": item.get("problem_solving", {}).get("currentProblem", ""),
                    "expectedBenefits": item.get("problem_solving", {}).get("expectedBenefits", [])
                },
                "target_users": item.get("target_users", []),
                "technology_stack": item.get("technology_stack", []),
                "title": item.get("title", ""),
                "workspace_id": item.get("workspace_id", 0)
            }

        project_summary = extract_project_summary(json_data)
        
        if not project_summary:
            print("✗ 프로젝트 정보 추출 실패")
            raise HTTPException(status_code=404, detail="프로젝트 정보를 찾을 수 없습니다.")
        
        print("✓ 프로젝트 정보 추출 성공:")
        print(json.dumps(project_summary, ensure_ascii=False, indent=2))

        # 4. 프롬프트 템플릿 렌더링
        try:
            template = Template(RCMD_PROMPT)
            rendered = template.render(input=json.dumps(project_summary, ensure_ascii=False, indent=2))
            print("✓ 템플릿 렌더링 성공")
        except Exception as e:
            print(f"✗ 템플릿 렌더링 실패: {str(e)}")
            raise HTTPException(status_code=500, detail=f"템플릿 렌더링 오류: {str(e)}")

        # 5. OpenAI API 호출 (최신 방식)
        try:
            print("OpenAI API 호출 시작...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 신입/초보 개발자 팀을 이끄는 시니어 테크 리드입니다. 주어진 정보를 바탕으로 해당 프로젝트를 성공적으로 구현하기 위한 작업 구조를 작성해야 합니다."
                    },
                    {
                        "role": "user", 
                        "content": rendered
                    }
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            generated = response.choices[0].message.content.strip()
            print("✓ OpenAI API 응답 생성 성공")
            print(f"응답 길이: {len(generated)} 문자")
            
        except Exception as e:
            print(f"✗ OpenAI API 호출 실패: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI API 호출 오류: {str(e)}")

        # 6. 응답 JSON 파싱
        try:
            generated_json = json.loads(generated)
            print("✓ LLM 응답 JSON 파싱 성공")
        except json.JSONDecodeError as e:
            print(f"✗ LLM 응답 JSON 파싱 실패")
            print(f"원본 응답: {generated[:500]}...")
            raise HTTPException(
                status_code=500, 
                detail=f"LLM 출력이 유효한 JSON 형식이 아닙니다: {str(e)}"
            )

        # 7. 응답 반환
        print("=== 작업 생성 완료 ===")
        return TaskGenerateResponse(generated_tasks=generated_json)
        
    except HTTPException:
        # 이미 HTTPException인 경우 그대로 재발생
        raise
    except Exception as e:
        # 예상치 못한 오류
        print(f"✗ 예상치 못한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"서버 내부 오류가 발생했습니다: {str(e)}"
        )

# 디버깅용 테스트 엔드포인트
@router.post("/task_generate/test")
def test_parsing(request: TaskGenerateRequest):
    """
    파싱 기능만 테스트하는 엔드포인트
    """
    try:
        # JSON 파싱 시도
        json_data = json.loads(request.project_summary)
        return {
            "status": "success",
            "parsing_method": "JSON",
            "data": json_data
        }
    except json.JSONDecodeError:
        try:
            # Python 딕셔너리 파싱 시도
            json_data = ast.literal_eval(request.project_summary)
            return {
                "status": "success", 
                "parsing_method": "Python Dictionary",
                "data": json_data
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "input_preview": request.project_summary[:200] + "..."
            }
