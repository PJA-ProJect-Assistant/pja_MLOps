from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
import re
from models.requests import SearchshimilerRequest
from models.response import SearchshimilerResponse
import json
import ast
import requests

router = APIRouter()

##################################################
# DB에 이으는 과정 필요! (DB 연결 구역)
##################################################

# 임시 데이터로 작성
# project_info DB 연결필요



def read_DB() :
    # url 접속
    DB_url = 'http://13.125.204.95:8080/api/workspaces/project-info'
    response = requests.get(url = DB_url)
    data = response.json()['data']

    # 방법 1: JSON string을 .json 확장자로 저장 (기본)
    user_project_info = ast.literal_eval(data)

    workspaceID_lst = []
    solutionIdea_lst = []
    stack_lst = []

    for i in range(len(user_project_info)) :
        workspaceID_lst.append(user_project_info[i]['workspaceId'])
        solutionIdea_lst.append(user_project_info[i]['problemSolving']['solutionIdea'])
        stack_lst.append(user_project_info[i]['technologyStack'])

    send_dataset = pd.DataFrame(index = workspaceID_lst, data = {
        'subject': solutionIdea_lst,
        'stack': stack_lst
    })

    return send_dataset
####################################################

# 텍스트 전처리 함수
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # HTML 태그 제거
    text = re.sub('<.*?>', '', str(text))
    # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
    text = re.sub('[^가-힣a-zA-Z0-9\s]', ' ', text)
    # 연속된 공백을 하나로 변환
    text = re.sub('\s+', ' ', text)
    return text.strip()

# TfidfVectorizer 진행
vectorizer = TfidfVectorizer(
    max_features=1000,  # 최대 특성 수
    ngram_range=(1, 2),  # 1-gram과 2-gram 사용
    min_df=1,  # 최소 문서 빈도
    max_df=0.95,  # 최대 문서 빈도 (너무 자주 나오는 단어 제외)
    stop_words=None
)

class ProjectRecommender :
    def __init__(self,  sample_df, vectorizer):
        self.sample_df = sample_df
        self.vectorizer = vectorizer

    def filter_stack_recomend_subjects(self, input_text, target_stack, recommendation_threshold, top_k = 10) :
        # stack 관련 단어 데이터 소문자화 
        target_stack = [item.lower() for item in target_stack]
        self.sample_df['stack'] = self.sample_df['stack'].astype(str).str.lower().apply(ast.literal_eval)

        # 벡터화 실행
        tfidf_matrix = vectorizer.fit_transform(self.sample_df['processed_subject'])

        # 사용자의 특정 개요의 유사도 기반 추천
        processed_input = preprocess_text(input_text)
        input_vector = vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
        self.sample_df['sim_score'] = similarities

        # 같은 스택 있는지에 대한 필터
        has_stack = self.sample_df[self.sample_df['stack'].apply(lambda x: bool(set(x) & set(target_stack)))]
        has_stack_sort = has_stack.sort_values(by = 'sim_score', ascending= False)
        has_stack_filter = has_stack_sort[has_stack_sort['sim_score'] > recommendation_threshold].head(top_k)
        original_similar_indices = has_stack_filter.index.tolist()

        return original_similar_indices

@router.post("/search_project/generate", response_model = SearchshimilerResponse)
async def generate_search_project(request: SearchshimilerRequest):
    try :
        # 데이터 준비
        send_dataset = read_DB()

        sample_df = send_dataset.copy()

        # subject 컬럼 전처리
        sample_df['processed_subject'] = sample_df['subject'].apply(preprocess_text)

        # 추천 시스템 초기화
        recommender = ProjectRecommender(sample_df, vectorizer)
        
        # 요청에서 데이터 추출
        input_text = ast.literal_eval(request.project_info)['problemSolving']['solutionIdea']
        target_stack = ast.literal_eval(request.project_info)['technologyStack']
        top_k = request.top_k
        score = request.recommendation_threshold

        recommend_id = recommender.filter_stack_recomend_subjects(input_text, target_stack, score, top_k)

        return SearchshimilerResponse(
            similer_ID = {"project_ID" : recommend_id})

    except KeyError as e:
        # 키 오류 처리
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required key in project_info: {str(e)}"
        )

    except Exception as e:
        # 기타 오류 처리
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )