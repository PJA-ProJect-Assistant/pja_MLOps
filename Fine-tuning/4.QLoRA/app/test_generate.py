import requests

url = "http://localhost:8000/generate"

data = {
    "prompt": "### 질문:\nFastAPI로 파인튜닝된 모델을 어떻게 서빙하나요?\n\n### 답변:",
    "max_new_tokens": 100,
    "temperature": 0.7
}

response = requests.post(url, json=data)

print("응답 결과:\n", response.json()["response"])