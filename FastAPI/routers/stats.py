# routers/stats.py
from fastapi import APIRouter, HTTPException
import pandas as pd
import os
from models.requests import DashboardRequest
from models.response import DashboardResponse
import json

router = APIRouter()

@router.post("/stats/generate", response_model=DashboardResponse)
def pipeline_data(request: DashboardRequest):

    # Back -> MLOps 데이터 전송
    LOG_DATA_PATH = request.user_log

    # 데이터 normalize
    data = json.loads(LOG_DATA_PATH)
    df = pd.json_normalize(data)

    # stat 1/2 공통 작업
    # userId 기준으로 df 분리
    user_dfs = {}

    for i in df['userId'].unique():
        user_dfs[f'df_user{i}'] = df[df['userId'] == i]

    # dashboard 1
    ### stat 1 start
    group_list = {}

    # state, importance 기준 grouping -> count 목적
    for i, (name, user_df) in enumerate(user_dfs.items(), start=1):
        group_list[f'grouped{i}'] = (user_df.groupby(['userId', 'details.state', 'details.importance']).size().reset_index(name='count'))

    # dict type(json)으로 변환
    for name, gr in group_list.items():
        json_temp_1 = gr.to_dict(orient='records')

#############################################################################################################

    ### stat 2 start
    filtered_users = {}

    # 필요한 컬럼만 추출
    for name, df in user_dfs.items():
        filtered = df[df['details.state'] == 'DONE'][
            ['userId', 'details.state', 'details.importance', 'details.startDate', 'details.endDate']
        ]
        filtered_users[name] = filtered

    # 각 사용자의 중요도별 평균 작업 시간 df 추출
    for name, df in filtered_users.items():
        # 날짜형 변환
        df['details.startDate'] = pd.to_datetime(df['details.startDate'], utc=True)
        df['details.endDate'] = pd.to_datetime(df['details.endDate'], utc=True)

        # 소요 시간 계산 (시간 단위)
        df['duration_hours'] = (df['details.endDate'] - df['details.startDate']).dt.total_seconds() / 3600

        # 컬럼 drop
        df = df[['userId', 'details.importance', 'duration_hours']]

        # 평균 계산
        df = df.groupby(['userId', 'details.importance'])['duration_hours'].mean().reset_index(name='mean_hours')

        # 딕셔너리 업데이트
        filtered_users[name] = df

    # dict type(json)으로 변환
    for name, fdf in filtered_users.items():
        json_temp_2 = fdf.to_dict(orient='records')
        
    return DashboardResponse(
        task_imbalance = {"data": f"{json_temp_1}"},
        processing_time = {"data": f"{json_temp_2}"}
    )