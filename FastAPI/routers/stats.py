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

    # actionId 단위의 최신 이벤트만
    latest_actions = (
        df
        .sort_values("timestamp", ascending=False)
        .drop_duplicates(
            subset=["workspaceId", "details.actionId"],
            keep="first"
        )
    )

    # stat 1/2 공통 작업 - participants 펼치기
    exploded = latest_actions.explode('details.participants')
    exploded['participants_userId'] = exploded['details.participants'].apply(lambda x: x.get('userId') if isinstance(x, dict) else None)

    # 필요한 필드 정리
    filtered = exploded[[
        'event',
        'userId',
        'participants_userId',
        'timestamp',
        'workspaceId',
        'details.actionId',
        'details.name',
        'details.state',
        'details.importance',
        'details.startDate',
        'details.endDate',
        'details.participants'
    ]]

    filtered = filtered.dropna(subset=['participants_userId'])

    # action DELETE 이벤트가 발생한 actions Id
    deleted_ids = latest_actions[latest_actions["event"] == "DELETE_PROJECT_PROGRESS_ACTION"]["details.actionId"].unique()

    # DELETE 이벤트 발생했던 actionId 제거
    filtered = filtered[~filtered["details.actionId"].isin(deleted_ids)]

    # Dtype 정리
    filtered['userId'] = filtered['userId'].astype(int)
    filtered['participants_userId'] = filtered['participants_userId'].astype(int)
    filtered['timestamp'] = pd.to_datetime(filtered['timestamp'])
    filtered['details.startDate'] = pd.to_datetime(filtered['details.startDate'])
    filtered['details.endDate'] = pd.to_datetime(filtered['details.endDate'])

    # 모든 사용자 ID (이벤트 발생자 + 참여자)
    all_user_ids = set(filtered['userId'].unique()) | set(filtered['participants_userId'].unique())

    user_dfs = {}
    for uid in all_user_ids:
        # 해당 사용자가 발생시킨 이벤트 OR 참여한 이벤트
        user_data = filtered[(filtered['userId'] == uid) | (filtered['participants_userId'] == uid)].copy()
        user_dfs[f'df_{uid}'] = user_data


    # ===== Statistics 1 Start =====
    group_list = {}

    # state, importance 기준 grouping -> count 목적
    # 이벤트 발생자와 참여자 모두 집계
    all_stat1_results = []

    for i, (name, user_df) in enumerate(user_dfs.items(), start=1):
        
        # 이벤트 발생자 집계
        initiator_grouped = user_df.groupby(['userId', 'details.state', 'details.importance']).size().reset_index(name='count')
        initiator_grouped['role'] = 'initiator'
        
        # 참여자 집계
        participants_grouped = user_df.groupby(['participants_userId', 'details.state', 'details.importance']).size().reset_index(name='count')
        participants_grouped['role'] = 'participant'
        participants_grouped = participants_grouped.rename(columns={'participants_userId': 'userId'})

        # 합치기
        combined = pd.concat([initiator_grouped, participants_grouped], ignore_index=True)
        all_stat1_results.append(combined)
        
        # 모든 유저의 결과 합치기
        if all_stat1_results:
            final_stat1 = pd.concat(all_stat1_results, ignore_index=True)
            # 같은 userId, state, importance, role 조합이 있다면 count 합계
            stat1_result = final_stat1.groupby(['userId', 'details.state', 'details.importance'])['count'].sum().reset_index().to_dict(orient='records')
        else:
            stat1_result = []

        # dict type(json)으로 변환
        for name, gr in group_list.items():
            stat1_result = gr.to_dict(orient='records')

        # ===== Statistics 2 Start=====
        filtered_users = {}

        # DONE 상태의 action만 뽑아오기
        done_df = exploded[exploded['details.state'] == 'DONE'][[
            'userId', 'participants_userId', 'details.state', 'details.importance', 
            'details.startDate', 'details.endDate', 'workspaceId', 
            'details.actionId', 'details.name', 'timestamp', 'event'
        ]].copy()

        done_df = done_df.dropna(subset=['participants_userId'])

        deleted_action_ids = latest_actions[latest_actions['event'] == 'DELETE_PROJECT_PROGRESS_ACTION']['details.actionId'].unique()

        # 삭제된 actionId 제거
        done_df = done_df[~done_df['details.actionId'].isin(deleted_action_ids)]

        # 날짜형 변환 및 duration 계산
        done_df['details.startDate'] = pd.to_datetime(done_df['details.startDate'], utc=True)
        done_df['details.endDate'] = pd.to_datetime(done_df['details.endDate'], utc=True)
        done_df['duration_hours'] = ((done_df['details.endDate'] - done_df['details.startDate']).dt.total_seconds() / 3600)

        # 결측치, 음수 제거
        done_df = done_df.dropna(subset=['duration_hours'])
        done_df = done_df[done_df['duration_hours'] >= 0]

        # 참여자용 데이터 (participants_userId 기준 중복 제거)
        participants_done_df = (
            done_df
            .sort_values('timestamp', ascending=False)
            .drop_duplicates(
                subset=[
                    'workspaceId',
                    'details.actionId',
                    'participants_userId'
                ],
                keep="first"
            )
        )

        # 이벤트 발생자용 데이터 (userId 기준 중복 제거)
        initiator_done_df = (
            done_df
            .sort_values('timestamp', ascending=False)
            .drop_duplicates(
                subset=[
                    'workspaceId',
                    'details.actionId',
                    'userId'
                ],
                keep="first"
            )
        )

        # 참여자 통계
        participants_result = (
            participants_done_df
            .groupby(['participants_userId', 'details.importance'])
            ['duration_hours']
            .mean()
            .reset_index(name='mean_hours')
            .rename(columns={'participants_userId': 'userId'})
        )
        participants_result['role'] = 'participant'

        # 이벤트 발생자 통계
        initiator_result = (
            initiator_done_df
            .groupby(['userId', 'details.importance'])
            ['duration_hours']
            .mean()
            .reset_index(name='mean_hours')
        )
        initiator_result['role'] = 'initiator'

        final_result = pd.concat([participants_result, initiator_result], ignore_index=True)
        final_result = final_result.drop('role', axis=1)
        filtered_users['stat2_result'] = final_result

        stat2_result = final_result.to_dict(orient='records')

    return DashboardResponse(
        task_imbalance = {"data": stat1_result},
        processing_time = {"data": stat2_result}
    )