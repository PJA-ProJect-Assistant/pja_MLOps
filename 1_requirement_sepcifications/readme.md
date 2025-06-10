** 폴더 구조 **

1_REQUIREMENT_SEPCIFICATIONS/
 ├── main.py 
 ├── routers/ 
 │ ├── __init__.py  
 │ └── requirements.py 
 ├── models/ 
 │ ├── __init__.py 
 │ ├── requests.py 
 │ └── response.py 
 ├── .env 
 ├── requirements.txt 
 └── README.md

 * 실행 방법

✅ 우분투 환경(Ubuntu22.04)에서 작업할 때
uv가 깔려있는 환경이라면

1) 가상환경 생성
uv venv .venv

2) 가상환경 진입
source .venv/bin/activate

3) 가상환경 폴더 들어가기
cd .venv

4) 환경 구성
uv pip install -r requirements.txt

5) FastAPI 접속
uvicorn main:app --reload 

6) 나오는 주소
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

이게 나오면 Ctrl+ 좌글릭으로 접속

7) 그러면 그 주소에서 
http://127.0.0.1:8000/docs

로 접속하고

8) requirements 에서 Try it out 버튼을 클릭하고
Request body에
"project_overview" 부분에는 사용자가 입력하는 프로젝트 정보를 입력해주고
"existing_requirements" 부분에는 요구사항 명세서 정보를 입력해주고
뒤에 공백을 지워준다.

9) 밑에 Execute 버튼으로 실행

10) 최종결과물 200으로 뜨면 정상작동.