# 1. Airflow DAG - 데이터 업데이트 파이프라인
# dags/project_data_pipeline.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import json
import psycopg2
from sqlalchemy import create_engine
import re

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'project_data_pipeline',
    default_args=default_args,
    description='프로젝트 데이터 ETL 파이프라인',
    schedule_interval=timedelta(hours=1),  # 1시간마다 실행
    catchup=False
)

def extract_project_data(**context):
    """데이터 소스에서 프로젝트 정보 추출"""
    # 실제 데이터 소스 연결 (API, 다른 DB 등)
    # 예시: API에서 데이터 가져오기
    import requests
    
    try:
        # API 호출 또는 다른 데이터 소스
        response = requests.get('YOUR_DATA_SOURCE_API')
        project_data = response.json()
        
        # 임시 저장
        with open('/tmp/raw_project_data.json', 'w', encoding='utf-8') as f:
            json.dump(project_data, f, ensure_ascii=False)
            
        return '/tmp/raw_project_data.json'
    except Exception as e:
        raise Exception(f"데이터 추출 실패: {e}")

def transform_project_data(**context):
    """데이터 변환 및 전처리"""
    import re
    
    # 추출된 데이터 로드
    with open('/tmp/raw_project_data.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 데이터 변환
    transformed_data = []
    for item in raw_data:
        transformed_item = {
            'project_id': item.get('id'),
            'subject': item['problem_solving']['solutionIdea'],
            'stack': item['technology_stack'],
            'processed_subject': preprocess_text(item['problem_solving']['solutionIdea']),
            'updated_at': datetime.now().isoformat()
        }
        transformed_data.append(transformed_item)
    
    # 변환된 데이터 저장
    df = pd.DataFrame(transformed_data)
    df.to_json('/tmp/transformed_project_data.json', orient='records', force_ascii=False)
    
    return '/tmp/transformed_project_data.json'

def load_to_database(**context):
    """데이터베이스에 로드"""
    # 데이터베이스 연결
    engine = create_engine('postgresql://username:password@localhost:5432/projectdb')
    
    # 변환된 데이터 로드
    df = pd.read_json('/tmp/transformed_project_data.json')
    
    # 데이터베이스에 저장 (기존 데이터 대체)
    df.to_sql('project_data', engine, if_exists='replace', index=False)
    
    print(f"총 {len(df)}개의 프로젝트 데이터가 업데이트되었습니다.")

def preprocess_text(text):
    """텍스트 전처리 함수"""
    if pd.isna(text):
        return ""
    text = re.sub('<.*?>', '', str(text))
    text = re.sub('[^가-힣a-zA-Z0-9\s]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

# 태스크 정의
extract_task = PythonOperator(
    task_id='extract_project_data',
    python_callable=extract_project_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_project_data', 
    python_callable=transform_project_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_to_database',
    python_callable=load_to_database,
    dag=dag
)

# 의존성 설정
extract_task >> transform_task >> load_task