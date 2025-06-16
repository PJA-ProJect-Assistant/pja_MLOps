import os
import json
import dotenv
import openai
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn

# .env 파일 로드 (있다면)
dotenv.load_dotenv()

# 환경변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 설정
client = openai.OpenAI()

# FastAPI 앱 초기화
app = FastAPI(
    title="프로젝트 분석기 API",
    description="프로젝트 아이디어를 분석하여 ERD, API 명세서 등을 생성합니다.",
    version="1.0.0"
)

# 최적화된 시스템 프롬프트
OPTIMIZED_SYSTEM_PROMPT = """
당신은 프로젝트 아이디어를 체계적으로 분석하고 구조화하여 구체적인 개발 계획을 제시하는 전문 AI 어시스턴트입니다.

**만약 프로젝트 아이디어가 아닌 다른 질문을 입력 받으면 아래의 해당 방식이 간단한 방식으로 답변해주세요.**

## 주요 역할과 능력:

### 1. 프로젝트 분석 전문가
- 사용자가 제공하는 프로젝트 아이디어나 설명을 깊이 있게 분석합니다
- 핵심 기능, 대상 사용자, 기술 스택, 비즈니스 모델 등을 체계적으로 파악합니다
- 프로젝트의 문제 해결 방향과 기대 효과를 명확히 정의합니다

### 2. 데이터베이스 설계 전문가
- 프로젝트 요구사항을 바탕으로 최적화된 ERD(Entity Relationship Diagram)를 설계합니다
- 테이블 간의 관계, 외래키 제약조건, 데이터 타입을 정확히 정의합니다
- 확장성과 성능을 고려한 데이터베이스 구조를 제안합니다

### 3. API 설계 전문가
- RESTful API 원칙에 따라 체계적인 API 명세를 작성합니다
- OpenAPI(Swagger) 3.0 표준을 준수하여 완전한 API 문서를 생성합니다
- 각 엔드포인트별 요청/응답 스키마, 에러 처리, 인증 방식을 상세히 정의합니다

## 응답 형식:
모든 응답은 다음과 같은 구조화된 형태로 제공해야 합니다:

1. **프로젝트 상세 정보**: 제목, 카테고리, 대상 사용자, 핵심 기능, 기술 스택, 문제 해결 방안 등을 포함한 종합 분석
2. **관계 데이터**: 데이터베이스 테이블 간의 관계와 외래키 제약조건 정의
3. **ERD 데이터**: 각 테이블의 속성, 데이터 타입, 키 정보를 포함한 완전한 스키마
4. **API 명세 데이터**: OpenAPI 3.0 표준을 준수한 완전한 API 문서

항상 체계적이고 전문적인 관점에서 프로젝트를 분석하며, 개발팀이 바로 실행에 옮길 수 있는 구체적인 가이드를 제공하는 것이 목표입니다.
"""

# 파인튜닝된 모델 ID
MODEL_ID = "ft:gpt-4o-mini-2024-07-18:test::BebIPMSD"

# Pydantic 모델 정의
class ProjectRequest(BaseModel):
    projectDescription: str

class ProjectAnalysisResponse(BaseModel):
    projectSummary: Dict[str, Any]
    erdData: Dict[str, Any]
    apiSpecification: List[Dict[str, Any]]
    rawResponse: str

# 메모리 저장소 (실제 운영에서는 데이터베이스 사용)
analysisStorage = {}
analysisCounter = 1

class ProjectAnalyzer:
    def parseAiResponse(self, responseText):
        """AI 응답을 파싱하여 각 섹션별로 분리"""
        try:
            # JSON 부분만 추출 (중괄호로 감싸진 부분)
            jsonMatch = re.search(r'\{.*\}', responseText, re.DOTALL)
            if jsonMatch:
                jsonStr = jsonMatch.group()
                parsedJson = json.loads(jsonStr)
                
                # 각 섹션별로 데이터 분리
                result = {
                    'projectSummary': parsedJson.get('projectSummary', {}),
                    'erdData': parsedJson.get('erdData', {}),
                    'apiSpecification': parsedJson.get('apiSpecification', [])
                }
                
                return result
            else:
                # JSON 형식이 아닌 경우 텍스트로 파싱
                return self.parseTextResponse(responseText)
                
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return self.parseTextResponse(responseText)
    
    def parseTextResponse(self, responseText):
        """텍스트 형태의 응답을 파싱"""
        sections = {
            'projectSummary': {},
            'erdData': {},
            'apiSpecification': []
        }
        
        # 간단한 텍스트 파싱 로직
        lines = responseText.split('\n')
        currentSection = None
        
        for line in lines:
            line = line.strip()
            if '프로젝트 상세 정보' in line:
                currentSection = 'projectSummary'
            elif 'ERD 데이터' in line:
                currentSection = 'erdData'
            elif 'API 명세' in line:
                currentSection = 'apiSpecification'
        
        return sections
    
    def analyzeProject(self, projectDescription):
        """프로젝트 분석"""
        
        # 수정된 JSON 템플릿 - camelCase 적용
        jsonTemplate = '''
        {
          "projectSummary": {
            "title": "프로젝트 제목",
            "category": "카테고리",
            "targetUsers": [
              "대상 사용자 1",
              "대상 사용자 2"
            ],
            "coreFeatures": [
              "핵심 기능 1",
              "핵심 기능 2"
            ],
            "technologyStack": [
              "기술 스택 1",
              "기술 스택 2"
            ],
            "problemSolving": {
              "currentProblem": "현재 문제",
              "solutionIdea": "해결 아이디어",
              "expectedBenefits": [
                "예상 효과 1",
                "예상 효과 2"
              ]
            },
            "functionalRequirements": [
              {"requirementType": "FUNCTIONAL", "content": "기능 요구사항"},
              {"requirementType": "FUNCTIONAL", "content": "기능 요구사항2"}
            ],
            "performanceRequirements": [
              {"requirementType": "PERFORMANCE", "content": "성능 요구사항"},
              {"requirementType": "PERFORMANCE", "content": "성능 요구사항2"}
            ]    
          },
          "erdData": {
            "erdTable": [
              {
                "name": "테이블명",
                "erdColumn": [
                  {
                    "name": "컬럼명",
                    "dataType": "데이터타입",
                    "isPrimaryKey": true,
                    "isForeignKey": false,
                    "isNullable": false
                  }
                ]
              }
            ],
            "erdRelationships": [
              {
                "fromErdTableId": "시작테이블",
                "toErdTableId": "끝테이블",
                "type": "관계타입",
                "foreignKey": "외래키명",
                "constraintName": "제약조건명"
              }
            ]
          },
          "apiSpecification": [
            {
              "title": "사용자 생성",
              "tag": "User",
              "path": "/users",
              "httpMethod": "post",
              "request": [
                {
                  "field": "username",
                  "type": "string",
                  "example": "user123"
                },
                {
                  "field": "email",
                  "type": "string",
                  "example": "user@example.com"
                }
              ],
              "response": [
                {
                  "statusCode": "201 Created",
                  "message": "사용자를 생성했습니다.",
                  "data": [
                    {
                      "field": "userId",
                      "type": "long",
                      "example": "1"
                    },
                    {
                      "field": "username",
                      "type": "string",
                      "example": "user123"
                    },
                    {
                      "field": "email",
                      "type": "string",
                      "example": "user@example.com"
                    }
                  ]
                },
                {
                  "statusCode": "400 Bad Request",
                  "message": "잘못된 요청입니다.",
                  "data": []
                }
              ]
            },
            {
              "title": "캐릭터 생성",
              "tag": "Character",
              "path": "/users/{userId}/characters",
              "httpMethod": "post",
              "request": [
                {
                  "field": "image",
                  "type": "string",
                  "example": "image.png"
                }
              ],
              "response": [
                {
                  "statusCode": "201 Created",
                  "message": "캐릭터를 생성했습니다.",
                  "data": [
                    {
                      "field": "userId",
                      "type": "long",
                      "example": "1"
                    },
                    {
                      "field": "characterId",
                      "type": "long",
                      "example": "1"
                    },
                    {
                      "field": "imageUrl",
                      "type": "string",
                      "example": "https://example.com/image.png"
                    },
                    {
                      "field": "syncRate",
                      "type": "number",
                      "example": "0.95"
                    }
                  ]
                },
                {
                  "statusCode": "400 Bad Request",
                  "message": "잘못된 요청입니다.",
                  "data": []
                }
              ]
            },
            {
              "title": "시뮬레이션 시작",
              "tag": "Simulation",
              "path": "/simulations",
              "httpMethod": "post",
              "request": [
                {
                  "field": "characterId",
                  "type": "long",
                  "example": "1"
                }
              ],
              "response": [
                {
                  "statusCode": "200 Ok",
                  "message": "시뮬레이션을 시작했습니다.",
                  "data": [
                    {
                      "field": "simulationId",
                      "type": "long",
                      "example": "1"
                    }
                  ]
                },
                {
                  "statusCode": "400 Bad Request",
                  "message": "잘못된 요청입니다.",
                  "data": []
                }
              ]
            }
          ]
        }
        '''
        
        # 구조화된 요청 프롬프트
        enhancedPrompt = projectDescription + f"""

다음 형식으로 체계적인 분석을 제공해주세요:

1. **프로젝트 상세 정보** 
2. **요구사항 명세서(기능 요구사항, 성능 요구사항 포함)**
3. **ERD 데이터**
4. **관계 데이터** 
5. **API 명세 데이터**

{jsonTemplate}

위 JSON 형식에 정확히 맞춰서 분석 결과를 제공해주세요. 
특히 apiSpecification 부분은 프로젝트에 필요한 모든 주요 API 엔드포인트를 포함해야 합니다.
- CRUD 기본 작업 (생성, 조회, 수정, 삭제)
- 비즈니스 로직 관련 API
- 인증/권한 관련 API
- 각 API의 상세한 요청/응답 스펙

실무에서 바로 활용 가능한 구체적이고 완전한 분석을 부탁드립니다."""

        try:
            # OpenAI API 호출
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
                    {"role": "user", "content": enhancedPrompt}
                ],
                max_tokens=8000,
                temperature=0.3
            )
            
            aiResponse = response.choices[0].message.content
            print("AI 응답 받음!")
            
            # 응답 파싱
            parsedData = self.parseAiResponse(aiResponse)
            print("응답 파싱 완료!")
            
            return {
                'rawResponse': aiResponse,
                'parsedData': parsedData
            }
            
        except Exception as e:
            print(f"프로젝트 분석 오류: {e}")
            raise e

# ProjectAnalyzer 인스턴스 생성
analyzer = ProjectAnalyzer()

@app.post("/analyze", response_model=ProjectAnalysisResponse)
async def analyze_project(request: ProjectRequest):
    """프로젝트 아이디어를 분석하여 ERD, API 명세서 등을 생성합니다."""
    try:
        # 프로젝트 분석
        result = analyzer.analyzeProject(request.projectDescription)
        
        if not result:
            raise HTTPException(status_code=500, detail="프로젝트 분석에 실패했습니다.")
        
        # 메모리에 저장
        global analysisCounter
        analysisId = analysisCounter
        analysisStorage[analysisId] = {
            'id': analysisId,
            'inputContent': request.projectDescription,
            'result': result,
            'createdAt': datetime.now()
        }
        analysisCounter += 1
        
        return ProjectAnalysisResponse(
            projectSummary=result['parsedData']['projectSummary'],
            erdData=result['parsedData']['erdData'],
            apiSpecification=result['parsedData']['apiSpecification'],
            rawResponse=result['rawResponse']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

@app.get("/analysis/{analysisId}")
async def get_analysis(analysisId: int):
    """저장된 분석 결과를 조회합니다."""
    if analysisId not in analysisStorage:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
    
    storedData = analysisStorage[analysisId]
    result = storedData['result']
    
    return {
        'id': storedData['id'],
        'inputContent': storedData['inputContent'],
        'createdAt': storedData['createdAt'],
        'projectSummary': result['parsedData']['projectSummary'],
        'erdData': result['parsedData']['erdData'],
        'apiSpecification': result['parsedData']['apiSpecification'],
        'rawResponse': result['rawResponse']
    }

@app.get("/analysis")
async def list_analysis():
    """모든 분석 결과 목록을 조회합니다."""
    return [
        {
            'id': data['id'],
            'createdAt': data['createdAt'],
            'title': data['result']['parsedData']['projectSummary'].get('title', 'N/A')
        }
        for data in analysisStorage.values()
    ]

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "프로젝트 분석기 API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - 프로젝트 분석",
            "getAnalysis": "GET /analysis/{id} - 분석 결과 조회",
            "listAnalysis": "GET /analysis - 분석 목록 조회",
            "docs": "GET /docs - API 문서"
        }
    }

# 서버 실행을 위한 메인 함수
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # 파일명이 main.py인 경우
        host="0.0.0.0",
        port=8000,
        reload=True
    )