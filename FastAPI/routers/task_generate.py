# routers/generate.py
from jinja2 import Template
import openai
from dotenv import load_dotenv
import json
import os
import random
from fastapi import APIRouter, HTTPException
from models.requests import TaskGenerateRequest
from models.response import TaskGenerateResponse

# OpenAI API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

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
            "name": "cagegory 이름",
            "features": [
                {
                    "name": "feature 이름",
                    "actions": [
                        { "name": "action 이름", "importance": 정수 },
                ]
                }
            ]
        }
    ]
}
"""

@router.post("/task_generate/generate", response_model=TaskGenerateResponse)
def task_generate(request: TaskGenerateRequest):

    # Read data
    JSON_FILE = request.project_summary
    # json_data = json.loads(JSON_FILE)
    json_data = JSON_FILE

    # 단일 객체인 경우 리스트로 감싸기 (지금 extract_project_summary는 리스트 기반)
    if isinstance(json_data, dict): json_data = [json_data]

    # 프로젝트 개요 중 지정 컬럼 읽기
    def extract_project_summary(input_json):
        for item in input_json:
            return {
                "category": item["category"],
                "core_features": item["core_features"],
                "problem_solving": {
                    "solutionIdea": item["problem_solving"]["solutionIdea"],
                    "currentProblem": item["problem_solving"]["currentProblem"],
                    "expectedBenefits": item["problem_solving"]["expectedBenefits"]
                },
                "target_users": item["target_users"],
                "technology_stack": item["technology_stack"],
                "title": item["title"],
                "workspace_id": item["workspace_id"]
            }
        return None

    # peject_id로 읽어온 프로젝트 개요
    project_summary = extract_project_summary(json_data)

    if project_summary:
        print(json.dumps(project_summary, ensure_ascii=False, indent=2))
    else:
        print("** 해당 project_id를 찾을 수 없습니다. **")

    # Jinja2 template 객체 생성
    template = Template(RCMD_PROMPT)

    # Rendering
    rendered = template.render(input=json.dumps(project_summary, ensure_ascii=False, indent=2))

    # 답변 생성
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 신입/초보 개발자 팀을 이끄는 시니어 테크 리드입니다. 주어진 정보를 바탕으로 해당 프로젝트를 성공적으로 구현하기 위한 작업 구조를 작성해야 합니다."},
            {"role": "user", "content": rendered}
        ],
        temperature=0.3
    )
    generated = response.choices[0].message.content.strip()

    try:
        generated = json.loads(generated)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM 출력이 JSON 아님: {str(e)}")


    # return TaskGenerateResponse(generated_tasks=generated)
    return TaskGenerateResponse(generated_tasks=json.loads(generated))
    # return  TaskGenerateResponse(generated_tasks = {"data": f"{generated}"})