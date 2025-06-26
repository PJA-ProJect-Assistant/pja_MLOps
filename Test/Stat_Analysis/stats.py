import json
import pandas as pd

LOG_DATA_PATH = "Stat_Analysis\\data\\final_user-actions_dummy.json"

# 중첩 구조 평탄화해서 읽기
with open(LOG_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    
df = pd.json_normalize(data)

# # stat 1/2 공통 작업 - participants 기반으로 user-task 복제
df = df.explode('details.participants')
df['participant_userId'] = df['details.participants'].apply(lambda x: x.get('userId') if isinstance(x, dict) else None)

# 필요한 필드만 정리
df = df[[
    'participant_userId',
    'details.state',
    'details.importance',
    'details.startDate',
    'details.endDate'
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

# dict type(json)으로 변환
for name, gr in group_list.items():
    json_temp = gr.to_dict(orient='records')

### stat 2 start
filtered_users = {}

# 필요한 컬럼만 추출
for name, df in user_dfs.items():
    filtered = df[df['details.state'] == 'DONE'][
        ['participant_userId', 'details.state', 'details.importance', 'details.startDate', 'details.endDate']
    ]
    filtered_users[name] = filtered

# 각 사용자의 중요도별 평균 작업 시간 df 추출
for name, df in filtered_users.items():
    # 날짜형 변환
    df['details.startDate'] = pd.to_datetime(df['details.startDate'], utc=True)
    df['details.endDate'] = pd.to_datetime(df['details.endDate'], utc=True)

    # 소요 시간 계산 (시간 단위)
    df['duration_hours'] = (df['details.endDate'] - df['details.startDate']).dt.total_seconds() / 3600
    df = df.dropna(subset=['duration_hours'])
    df = df[df['duration_hours'] >= 0]

    # 컬럼 drop
    df = df[['participant_userId', 'details.importance', 'duration_hours']]

    # 평균 계산
    df = df.groupby(['participant_userId', 'details.importance'])['duration_hours'].mean().reset_index(name='mean_hours')

    # 딕셔너리 업데이트
    filtered_users[name] = df

# dict type(json)으로 변환
for name, fdf in filtered_users.items():
    json_temp = fdf.to_dict(orient='records')

    # # 파일 저장
    # with open(f"Stat_Analysis\\data\\stat2-{name}.json", "w", encoding="utf-8") as f:
    #     for item in json_temp:  # data는 List[Dict]
    #         json_line = json.dumps(item, ensure_ascii=False)
    #         f.write(json_line + "\n")
