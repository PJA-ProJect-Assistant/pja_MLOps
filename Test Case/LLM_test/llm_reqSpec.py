import dotenv
import os
import openai
import json
import os
import time

# .env 파일 로드 (있다면)
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Read prompt
with open("C:/Users/user/Documents/pja_MLOps/LLM_test/reqSpec_prompt.md", "r", encoding="utf-8") as f:
    OPTIMIZED_SYSTEM_PROMPT = f.read()
    # print(OPTIMIZED_SYSTEM_PROMPT)  # test

def req_spec_func(project_description, func_spec, perf_spec):

    # python 객체를 문자열로 변환, ∵gpt는 문자열만 처리 가능
    func_prompt = json.dumps(func_spec, ensure_ascii=False, indent=4)  # ensure_ascii=False -> 비ASCII 문자를 unicode escape X
    perf_prompt = json.dumps(func_spec, ensure_ascii=False, indent=4)  # indent=4           -> 들여쓰기(formatting)
    
    project_context = f"""
    [프로젝트 설명]
    {project_description}

    [기능 요구사항]
    {func_prompt}

    [성능 요구사항]
    {perf_prompt}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
                {"role": "user", "content": project_context}
            ],
            # max_tokens=256,
            temperature=0.,
            frequency_penalty=0.2
        )
        print(response.model_dump()["model"])  # model response check

        for i, choice in enumerate(response.choices):  # 전체 response 확인
            print(f"response {i+1} :\n{choice.message.content}\n")
        
        return 0
        
        # return response.choices[0].message.content
        
    except Exception as e:  return f"오류 발생: {e}"

# 사용자 입력 데이터
my_project = "GQ 서비스는 학습 중심의 스터디 그룹 운영을 지원하는 웹 플랫폼으로, 단순한 그룹 관리에서 나아가 그룹 내 학습 내용을 복습할 수 있는 기능을 제공하여 학습 효과를 극대화하는 것을 목표로 한다.\
              사용자들은 회원가입과 로그인을 통해 서비스를 이용할 수 있으며, 아이디 및 비밀번호 찾기 기능으로 편의성을 높였다.\
              사용자는 새로운 스터디 그룹을 생성하거나 기존 그룹을 검색해 가입할 수 있으며, 그룹 내에서는 퀴즈를 생성하고 참여하여 학습한 내용을 효과적으로 반복 학습할 수 있다.\
              또한 공지사항 게시판을 통해 그룹 내 전달사항을 공유할 수 있고, 과제 게시판은 체크리스트 형태로 구성되어 과제 수행 여부를 명확히 기록할 수 있다.\
              이러한 기능들은 사용자 간의 소통을 강화하고 학습 동기를 부여하는 데 기여한다.\
              서비스는 Java Servlet과 JSP 기반으로 구현되었으며, HTML, CSS, JavaScript를 통해 사용자 친화적인 인터페이스를 제공한다.\
              데이터베이스는 MySQL을 사용하여 안정적인 데이터 저장과 처리를 가능하게 하며, 전반적인 시스템은 학습 관리에 최적화된 구조로 설계되었다."

func_spec = [
    {
        "requirementType": "FUNCTIONAL",
        "content": "사용자는 회원가입 및 로그인 기능을 통해 개인 계정을 생성하고 서비스에 접근할 수 있어야 한다."
    },
    {
        "requirementType": "FUNCTIONAL",
        "content": "사용자는 스터디 그룹을 생성하거나 검색하여 가입할 수 있어야 하며, 각 그룹 내에서는 퀴즈 생성, 퀴즈 참여, 공지사항 작성, 과제 등록이 가능해야 한다."
    },
    {
        "requirementType": "FUNCTIONAL",
        "content": "스터디 그룹 내에서 사용자는 퀴즈를 생성하고 참여할 수 있으며, 과제 게시판은 체크리스트 형태로 구성되어 과제 수행 여부를 시각적으로 확인 가능해야 한다."
    }
]            

perf_spec = [
    {
        "requirementType": "PERFORMANCE",
        "content": "사용자는 로그인, 그룹 가입, 퀴즈 제출 등의 주요 기능 수행 시 평균 1초 이내에 응답을 받아야 한다."
    },
    {
        "requirementType": "PERFORMANCE",
        "content": "시스템은 최소 100명의 동시 접속 사용자가 문제 없이 퀴즈 응시 및 그룹 내 활동을 수행할 수 있도록 안정적으로 작동해야 한다."
    },
    {
        "requirementType": "PERFORMANCE",
        "content": "과제 제출, 퀴즈 응답, 공지사항 등록 등에서 발생하는 모든 데이터는 MySQL을 통해 즉시 저장되어야 하며, 동시에 여러 사용자가 접근할 경우에도 데이터 충돌이나 손실이 없어야 한다."
    }
]            

start = time.time()
response = req_spec_func(my_project, func_spec, perf_spec)
end = time.time()
resp_time = f"{end - start:.5f} sec"

print("Response Time: ", resp_time)
print(response)