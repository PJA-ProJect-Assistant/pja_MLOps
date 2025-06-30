# routers/stats.py
from fastapi import APIRouter
import pandas as pd
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

    # state, importance 기준 grouping -> count 목적
    # 이벤트 발생자와 참여자 모두 집계
    all_stat1_results = []

    for user_df in user_dfs.values():
        # 발생자 데이터 준비
        initiator_data = user_df[['userId', 'details.state', 'details.importance', 'details.actionId']].copy()
        initiator_data['role'] = 'initiator'
        initiator_data = initiator_data.rename(columns={'userId': 'final_userId'})
        
        # 참여자 데이터 준비  
        participant_data = user_df[['participants_userId', 'details.state', 'details.importance', 'details.actionId']].copy()
        participant_data['role'] = 'participant'
        participant_data = participant_data.rename(columns={'participants_userId': 'final_userId'})
        
        # 합치기
        user_all_data = pd.concat([initiator_data, participant_data], ignore_index=True)
        all_stat1_results.append(user_all_data)

    # 모든 유저 데이터 합치기
    if all_stat1_results:
        combined_stats = pd.concat(all_stat1_results, ignore_index=True)
        
        # 중복 제거: 같은 final_userId, state, importance, role, actionId 조합의 중복 행 제거
        dedup_stats = combined_stats.drop_duplicates(
            subset=['final_userId', 'details.state', 'details.importance', 'details.actionId']
        )
        
        # 최종 집계
        stat1_result = (
            dedup_stats
            .groupby(['final_userId', 'details.state', 'details.importance'])
            .size()
            .reset_index(name='count')
            .rename(columns={'final_userId': 'userId'})
            .to_dict(orient='records')
        )
    else:
        stat1_result = []


    # ===== Statistics 2 Start=====
    
    # DONE 상태의 action만 뽑아오기
    done_df = filtered[filtered['details.state'] == 'DONE'].copy()

    # 날짜형 변환 및 duration 계산
    done_df['details.startDate'] = pd.to_datetime(done_df['details.startDate'], utc=True)
    done_df['details.endDate'] = pd.to_datetime(done_df['details.endDate'], utc=True)
    done_df['duration_hours'] = ((done_df['timestamp'] - done_df['details.startDate']).dt.total_seconds() / 3600)

    # 결측치, 음수 제거
    done_df = done_df.dropna(subset=['duration_hours'])
    done_df = done_df[done_df['duration_hours'] >= 0]

    # 참여자용 중복 제거
    participants_done_df = (
        done_df
        .sort_values('timestamp', ascending=False)
        .drop_duplicates(
            subset=['workspaceId', 'details.actionId', 'participants_userId'],
            keep="first"
        )
    )

    # 발생자용 중복 제거  
    initiator_done_df = (
        done_df
        .sort_values('timestamp', ascending=False)
        .drop_duplicates(
            subset=['workspaceId', 'details.actionId', 'userId'],
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

    # 발생자 통계
    initiator_result = (
        initiator_done_df
        .groupby(['userId', 'details.importance'])
        ['duration_hours']
        .mean()
        .reset_index(name='mean_hours')
    )
    initiator_result['role'] = 'initiator'

    # 결과 통합 (role 정보 유지)
    final_result = pd.concat([participants_result, initiator_result], ignore_index=True)
    final_result = final_result.drop('role', axis=1)
    stat2_result = final_result.to_dict(orient='records')
        
    return DashboardResponse(
        task_imbalance = {"data": stat1_result},
        processing_time = {"data": stat2_result}
    )