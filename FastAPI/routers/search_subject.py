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


router = APIRouter()

##################################################
# DB에 이으는 과정 필요! (DB 연결 구역)
##################################################

# 임시 데이터로 작성
# project_info DB 연결필요

def read_DB() :
    with open("DB/project_info_DB.json", "r", encoding="utf-8") as f: 
        user_project_info = ast.literal_eval(json.load(f))
    
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

    def filter_stack_recomend_subjects(self, input_text ,target_stack, top_k = 10) :
        
        # 공통되는 요소가 하나라도 있는 행
        has_stack = self.sample_df[self.sample_df['stack'].apply(lambda x: bool(set(x) & set(target_stack)))]

        # 원본 인덱스 저장
        original_indices = has_stack.index.tolist()

        # 벡터화 실행
        tfidf_matrix = vectorizer.fit_transform(has_stack['processed_subject'])

        # 사용자의 특정 개요의 유사도 기반 추천
        processed_input = preprocess_text(input_text)
        input_vector = vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
        similar_indices = similarities.argsort()[::-1][:top_k]

        # 원본 DataFrame의 인덱스로 변환
        original_similar_indices = [original_indices[i] for i in similar_indices]

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

        recommend_id = recommender.filter_stack_recomend_subjects(input_text, target_stack, top_k)

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