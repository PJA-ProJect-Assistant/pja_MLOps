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
    # participants 기반으로 user-task 복제
    df = df.explode('details.participants')
    df['participant_userId'] = df['details.participants'].apply(lambda x: x.get('userId') if isinstance(x, dict) else None)

    # 필요한 필드만 정리
    df = df[[
        'participant_userId',
        'details.state',
        'details.importance',
        'details.startDate',
        'timestamp'
    ]]

    df = df.dropna(subset=['participant_userId'])

    # userId별로 분리
    user_dfs = {}
    for uid in df['participant_userId'].unique():
        user_dfs[f'df_{uid}'] = df[df['participant_userId'] == uid].copy()

    ### stat 1 start
    group_list = {}

    # state, importance 기준 grouping -> count 목적
    for i, (name, user_df) in enumerate(user_dfs.items(), start=1):
        group_list[f'grouped{i}'] = (user_df.groupby(['participant_userId', 'details.state', 'details.importance']).size().reset_index(name='count'))

    # stat1 result: dict type(json)으로 변환
    stat1_result = {}

    for name, gr in group_list.items():
        stat1_result[name] = gr.to_dict(orient='records')

    ### stat 2 start
    filtered_users = {}

    # 필요한 컬럼만 추출
    for name, df in user_dfs.items():
        filtered = df[df['details.state'] == 'DONE'][
            ['participant_userId', 'details.state', 'details.importance', 'details.startDate', 'timestamp']
        ]
        filtered_users[name] = filtered

    # 각 사용자의 중요도별 평균 작업 시간 df 추출
    for name, df in filtered_users.items():
        # 날짜형 변환
        df['details.startDate'] = pd.to_datetime(df['details.startDate'], utc=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # 소요 시간 계산 (시간 단위)
        df['duration_hours'] = (df['timestamp'] - df['details.startDate']).dt.total_seconds() / 3600
        df = df.dropna(subset=['duration_hours'])
        df = df[df['duration_hours'] >= 0]

        # 컬럼 drop
        df = df[['participant_userId', 'details.importance', 'duration_hours']]

        # 평균 계산
        df = df.groupby(['participant_userId', 'details.importance'])['duration_hours'].mean().reset_index(name='mean_hours')

        # 딕셔너리 업데이트
        filtered_users[name] = df

    # stat2 result: dict type(json)으로 변환
    stat2_result = {}

    for name, fdf in filtered_users.items():
        stat2_result[name] = fdf.to_dict(orient='records')
        
    return DashboardResponse(
        task_imbalance = {"data": stat1_result},
        processing_time = {"data": stat2_result}
    )