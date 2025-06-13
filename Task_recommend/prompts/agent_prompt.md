Question: 아래의 JSON 작업 흐름을 바탕으로 각 feature에 대해 이어질 작업 3개씩 추천해주세요. 추천 작업은 feature의 actions 배열에 포함되어야 하며, name, startDate, endDate, importance가 반드시 포함되어야 합니다. 결과는 반드시 JSON 형식으로 출력하고, 추가 설명 없이 JSON만 출력해야 합니다. 출력 예시는 아래 형식을 따릅니다:

{
  "workspaceId": "작업 중인 workspaceId",
  "recommendedCategories": [
    {
      "name": "카테고리 이름",
      "startDate": "YYYY-MM-DDTHH:MM:SS",
      "endDate": "YYYY-MM-DDTHH:MM:SS",
      "importance": 정수,
      "features": [
        {
          "name": "기능 이름",
          "startDate": "YYYY-MM-DDTHH:MM:SS",
          "endDate": "YYYY-MM-DDTHH:MM:SS",
          "importance": 정수,
          "actions": [
            {
              "name": "추천 action",
              "startDate": "YYYY-MM-DDTHH:MM:SS",
              "endDate": "YYYY-MM-DDTHH:MM:SS",
              "importance": 정수
            },
            {
              "name": "추천 action",
              "startDate": "YYYY-MM-DDTHH:MM:SS",
              "endDate": "YYYY-MM-DDTHH:MM:SS",
              "importance": 정수
            },
            {
              "name": "추천 action",
              "startDate": "YYYY-MM-DDTHH:MM:SS",
              "endDate": "YYYY-MM-DDTHH:MM:SS",
              "importance": 정수
            },
            ...
          ]
        }
      ]
    }
  ]
}

Thought: recommendedCategories, features, actions를 분석하여 각 feature에 자연스럽게 이어질 다음 actions 3개씩을 구상합니다.
Action: ParseWorkflow
Action Input: {input}

Observation: 전체 워크스페이스와 feature의 작업 흐름 및 현재 진행 상황을 파악했습니다.

Thought: 각 feature에 적절한 추천 작업 3개씩을 생성합니다.
Action: Final Answer
Action Input:
{
  // 위에서 언급한 JSON 형식의 출력값
}
