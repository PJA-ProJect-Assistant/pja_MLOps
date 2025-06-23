from jinja2 import Template
import openai
from dotenv import load_dotenv
import json
import os
import random

# OpenAI API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Files path
JSON_FILE_PATH = "data\\project_summary.json"
RCMD_PROMPT_PATH = "prompts\\cat-feat_recommend.md"

# Read data
with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Read prompt
with open(RCMD_PROMPT_PATH, "r", encoding="utf-8", errors="replace") as f:
    RCMD_PROMPT = f.read()

# Project ID list : 예시로 받은 데이터에서 테스트용으로 뽑은 id list
proj_id_list = [1, 11, 4, 12, 6, 16, 15]
project_id = int(random.choice(proj_id_list))

# peject_id로 프로젝트 개요 중 지정 컬럼 읽어오는 함수
def extract_project_summary(input_json, id):
    for item in input_json:
        if item.get("project_info_id") == id:
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
project_summary = extract_project_summary(json_data, project_id)

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
