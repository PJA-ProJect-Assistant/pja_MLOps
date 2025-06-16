#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프로젝트 진행 추천을 위한 더미 데이터 생성기
파일명: create_project_dummy_data.py
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

class ProjectDummyDataGenerator:
    def __init__(self):
        # 프로젝트 카테고리별 기능과 액션 정의
        self.categories_data = {
            "회원 관리 기능": {
                "features": {
                    "회원 가입": [
                        "이메일 인증 로직 구현", "회원 정보 유효성 검증", "비밀번호 암호화 처리",
                        "소셜 로그인 연동", "프로필 이미지 업로드", "이메일 중복 검사"
                    ],
                    "로그인/로그아웃": [
                        "JWT 토큰 발급", "세션 관리", "자동 로그인 기능", "비밀번호 찾기",
                        "다중 기기 로그인 제한", "로그인 기록 관리"
                    ],
                    "프로필 관리": [
                        "개인정보 수정", "프로필 이미지 변경", "계정 탈퇴", "비밀번호 변경",
                        "알림 설정", "개인정보 보호 설정"
                    ]
                }
            },
            "상품 관리 기능": {
                "features": {
                    "상품 등록": [
                        "상품 정보 입력 폼", "이미지 다중 업로드", "카테고리 분류",
                        "가격 설정", "재고 관리", "상품 옵션 설정"
                    ],
                    "상품 조회": [
                        "상품 목록 페이징", "검색 필터링", "정렬 기능", "상품 상세 조회",
                        "관련 상품 추천", "상품 리뷰 표시"
                    ],
                    "재고 관리": [
                        "재고 수량 업데이트", "재고 부족 알림", "입출고 기록",
                        "재고 통계", "자동 발주 시스템", "재고 실사"
                    ]
                }
            },
            "주문 관리 기능": {
                "features": {
                    "주문 처리": [
                        "장바구니 기능", "결제 연동", "주문 확인서 발송", "주문 상태 업데이트",
                        "배송 추적", "주문 취소 처리"
                    ],
                    "결제 시스템": [
                        "결제 게이트웨이 연동", "결제 보안", "환불 처리", "할인 쿠폰 적용",
                        "포인트 적립", "결제 실패 처리"
                    ],
                    "배송 관리": [
                        "배송업체 연동", "배송비 계산", "배송 상태 추적", "배송 완료 알림",
                        "반품/교환 처리", "배송 지연 알림"
                    ]
                }
            },
            "게시판 기능": {
                "features": {
                    "게시글 관리": [
                        "게시글 작성", "게시글 수정/삭제", "파일 첨부", "게시글 검색",
                        "조회수 관리", "게시글 신고"
                    ],
                    "댓글 시스템": [
                        "댓글 작성", "대댓글 기능", "댓글 수정/삭제", "댓글 좋아요",
                        "댓글 신고", "댓글 알림"
                    ],
                    "관리 기능": [
                        "게시판 권한 관리", "스팸 필터링", "게시글 승인", "통계 관리",
                        "백업/복구", "게시판 설정"
                    ]
                }
            },
            "알림 시스템": {
                "features": {
                    "실시간 알림": [
                        "WebSocket 연결", "푸시 알림", "이메일 알림", "SMS 알림",
                        "알림 설정 관리", "알림 기록"
                    ],
                    "알림 관리": [
                        "알림 템플릿", "알림 스케줄링", "알림 통계", "알림 필터링",
                        "알림 우선순위", "알림 그룹화"
                    ]
                }
            },
            "데이터 분석": {
                "features": {
                    "통계 대시보드": [
                        "실시간 통계", "차트 생성", "보고서 생성", "데이터 시각화",
                        "KPI 모니터링", "성능 지표"
                    ],
                    "사용자 분석": [
                        "사용자 행동 분석", "접속 통계", "이탈률 분석", "전환율 분석",
                        "A/B 테스트", "개인화 추천"
                    ]
                }
            }
        }
        
        self.importance_levels = [1, 2, 3, 4, 5]  # 중요도 레벨
        
    def generate_random_date(self, start_days_from_now: int = 0, duration_days: int = 30) -> datetime:
        """랜덤한 날짜 생성"""
        start_date = datetime.now() + timedelta(days=start_days_from_now)
        return start_date + timedelta(days=random.randint(0, duration_days))
    
    def generate_backend_to_mlops_data(self, count: int = 100) -> List[Dict[str, Any]]:
        """백엔드에서 MLOps로 보내는 데이터 생성"""
        dummy_data = []
        
        for i in range(count):
            workspace_id = random.randint(1, 20)
            
            # 랜덤하게 카테고리 선택
            category_name = random.choice(list(self.categories_data.keys()))
            category_id = list(self.categories_data.keys()).index(category_name) + 1
            
            recommended_categories = []
            
            # 1-3개의 카테고리 추천
            num_categories = random.randint(1, 3)
            selected_categories = random.sample(list(self.categories_data.keys()), num_categories)
            
            for cat_name in selected_categories:
                cat_id = list(self.categories_data.keys()).index(cat_name) + 1
                features = []
                
                # 각 카테고리당 1-3개의 기능
                num_features = random.randint(1, 3)
                feature_names = random.sample(list(self.categories_data[cat_name]["features"].keys()), 
                                            min(num_features, len(self.categories_data[cat_name]["features"])))
                
                for feature_name in feature_names:
                    feature_id = random.randint(1, 100)
                    actions = []
                    
                    # 각 기능당 1-4개의 액션
                    num_actions = random.randint(1, 4)
                    action_names = random.sample(self.categories_data[cat_name]["features"][feature_name], 
                                                min(num_actions, len(self.categories_data[cat_name]["features"][feature_name])))
                    
                    for action_name in action_names:
                        start_date = self.generate_random_date(0, 7)
                        end_date = start_date + timedelta(days=random.randint(1, 14))
                        
                        actions.append({
                            "name": action_name,
                            "importance": random.choice(self.importance_levels),
                            "startDate": start_date.isoformat(),
                            "endDate": end_date.isoformat()
                        })
                    
                    features.append({
                        "featureId": feature_id,
                        "name": feature_name,
                        "actions": actions
                    })
                
                recommended_categories.append({
                    "categoryId": cat_id,
                    "name": cat_name,
                    "features": features
                })
            
            dummy_data.append({
                "workspaceId": workspace_id,
                "recommendedCategories": recommended_categories
            })
        
        return dummy_data
    
    def generate_mlops_to_backend_data(self, count: int = 100) -> List[Dict[str, Any]]:
        """MLOps에서 백엔드로 보내는 데이터 생성"""
        dummy_data = []
        
        for i in range(count):
            workspace_id = random.randint(1, 20)
            category_name = random.choice(list(self.categories_data.keys()))
            category_id = list(self.categories_data.keys()).index(category_name) + 1
            
            # 랜덤 기능 선택
            feature_name = random.choice(list(self.categories_data[category_name]["features"].keys()))
            feature_id = random.randint(1, 100)
            
            # 추천 액션 생성
            recommended_actions = []
            available_actions = self.categories_data[category_name]["features"][feature_name]
            num_actions = random.randint(1, min(5, len(available_actions)))
            selected_actions = random.sample(available_actions, num_actions)
            
            for action_name in selected_actions:
                start_date = self.generate_random_date(0, 7)
                end_date = start_date + timedelta(days=random.randint(1, 14))
                
                recommended_actions.append({
                    "name": action_name,
                    "importance": random.choice(self.importance_levels),
                    "startDate": start_date.isoformat(),
                    "endDate": end_date.isoformat()
                })
            
            dummy_data.append({
                "workspaceId": workspace_id,
                "categoryId": category_id,
                "featureId": feature_id,
                "recommendedActions": recommended_actions
            })
        
        return dummy_data
    
    def save_to_files(self):
        """생성된 데이터를 파일로 저장"""
        # 백엔드 → MLOps 데이터
        backend_to_mlops = self.generate_backend_to_mlops_data(100)
        with open('backend_to_mlops_dummy_data.json', 'w', encoding='utf-8') as f:
            json.dump(backend_to_mlops, f, ensure_ascii=False, indent=2)
        
        # MLOps → 백엔드 데이터
        mlops_to_backend = self.generate_mlops_to_backend_data(100)
        with open('mlops_to_backend_dummy_data.json', 'w', encoding='utf-8') as f:
            json.dump(mlops_to_backend, f, ensure_ascii=False, indent=2)
        
        print("✅ 더미 데이터 생성 완료!")
        print(f"📁 backend_to_mlops_dummy_data.json: {len(backend_to_mlops)}개 데이터")
        print(f"📁 mlops_to_backend_dummy_data.json: {len(mlops_to_backend)}개 데이터")
        
        return backend_to_mlops, mlops_to_backend

if __name__ == "__main__":
    generator = ProjectDummyDataGenerator()
    generator.save_to_files()