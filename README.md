# 🚀 PJA MLOps Project

![Language](https://img.shields.io/badge/Stack-Python%20%7C%20FastAPI-blue) ![Contributors](https://img.shields.io/badge/Contributors-3-brightgreen)

> **프로젝트 어시스턴트(Project Assistant)를 위한 AI 기반 MLOps 플랫폼**  
> OpenAI API를 활용한 지능형 요구사항 분석 및 프로젝트 관리 시스템

## 📋 목차

- [📖 프로젝트 개요](#-프로젝트-개요)
- [🏗️ 아키텍처](#️-아키텍처)
- [📁 폴더 구조](#-폴더-구조)
- [⚡ 주요 기능](#-주요-기능)
- [🛠️ 기술 스택](#️-기술-스택)
- [🚀 빠른 시작](#-빠른-시작)
- [📚 API 문서](#-api-문서)
- [🔄 CI/CD 파이프라인](#-cicd-파이프라인)
- [🐳 Docker 배포](#-docker-배포)
- [🧪 테스트](#-테스트)
- [📊 모니터링](#-모니터링)
- [🤝 기여 가이드](#-기여-가이드)
- [📝 라이선스](#-라이선스)

## 📖 프로젝트 개요

PJA MLOps Project는 AI 기반의 프로젝트 관리 및 요구사항 분석 플랫폼입니다.
OpenAI API를 활용하여 프로젝트 요구사항을 자동으로 분석하고, 카테고리와 기능을 추천하며, 실행 가능한 액션 아이템을 생성합니다.

### 🎯 주요 목표

- **지능형 요구사항 분석**: AI를 통한 자동 요구사항 분석 및 분류
- **프로젝트 관리 자동화**: 카테고리, 기능, 액션 추천을 통한 프로젝트 계획 수립
- **실시간 API 서비스**: FastAPI 기반의 고성능 웹 서비스

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │────│   FastAPI       │────│   OpenAI API    │
│                 │    │   Server        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                        ┌─────────────────┐
                        │   Database      │
                        │   (JSON Files)  │
                        └─────────────────┘
```

## 📁 폴더 구조

```
pja_MLOps/
├── 📁 .github/                    # GitHub Actions 워크플로우
│   └── workflows/
│       └── deploy.yml             # CI/CD 파이프라인
├── 📁 FastAPI/                    # 웹 API 서비스
│   ├── models/                    # 데이터 모델
│   │   ├── requests.py            # 요청 모델
│   │   └── response.py            # 응답 모델
│   ├── routers/                   # API 라우터
│   │   ├── requirements.py        # 요구사항 분석 API
│   │   ├── recommendation.py      # 추천 시스템 API
│   │   ├── json_API.py            # JSON 처리 API
│   │   └── stats.py               # 통계 API
│   ├── utils/                     # 유틸리티 함수
│   │   ├── json_parsing.py        # JSON 파싱 유틸
│   │   └── slack_alarm.py         # 슬랙 알림
│   ├── DB/                        # 데이터베이스 관련
│   ├── main.py                    # FastAPI 메인 애플리케이션
│   ├── Dockerfile                 # Docker 이미지 빌드
│   ├── requirements.txt           # Python 의존성
│   └── startup.sh                 # 서버 시작 스크립트
├── 📁 Test                        # 테스트 및 실험
├── 📋 requirements.txt            # 전역 의존성
├── 📋 pyproject.toml              # 프로젝트 설정
├── 📋 uv.lock                     # 의존성 잠금 파일
└── 📋 README.md                   # 프로젝트 문서
```

## ⚡ 주요 기능

### 🤖 AI 기반 요구사항 분석
- **우선순위 결정**: AI 기반 요구사항 우선순위 자동 설정
- **요구사항 검증**: 일관성 및 완전성 자동 검사

### 📊 지능형 추천 시스템
- **카테고리 추천**: 프로젝트 성격에 맞는 카테고리 제안
- **기능 추천**: 요구사항 기반 핵심 기능 추천
- **액션 추천**: 구체적인 실행 계획 및 태스크 생성

### 🔄 실시간 API 서비스
- **RESTful API**: 표준화된 API 인터페이스
- **실시간 처리**: 비동기 처리를 통한 빠른 응답

### 📈 모니터링 및 로깅
- **헬스체크**: 서비스 상태 실시간 모니터링
- **성능 메트릭**: API 성능 및 사용량 추적

## 🛠️ 기술 스택

### Backend
- **Python 3.9+**: 메인 프로그래밍 언어
- **FastAPI**: 고성능 웹 프레임워크
- **Pydantic**: 데이터 검증 및 직렬화
- **Uvicorn**: ASGI 서버

### AI/ML
- **OpenAI API**: GPT 모델 활용
- **scikit-learn**: 머신러닝 알고리즘
- **KoNLPy**: 한국어 자연어 처리

### DevOps
- **Docker**: 컨테이너화
- **GitHub Actions**: CI/CD 파이프라인
- **AWS EC2**: 클라우드 배포
- **Docker Hub**: 컨테이너 레지스트리

### Development Tools
- **uv**: Python 패키지 관리자
- **Jupyter**: 데이터 분석 및 실험
- **Git**: 버전 관리

## 🚀 빠른 시작

### 사전 요구사항
- Python 3.9 이상
- uv 패키지 매니저
- OpenAI API 키
- Docker (선택사항)

### 1. 저장소 클론
```bash
git clone https://github.com/PJA-ProJect-Assistant/pja_MLOps.git
cd pja_MLOps
```

### 2. 환경 설정
```bash
# FastAPI 디렉토리로 이동
cd FastAPI

# 가상환경 생성
uv venv .venv

# 가상환경 활성화
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows

# 의존성 설치
uv pip install -r requirements.txt
```

### 3. 환경변수 설정
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 4. 서버 실행
```bash
# 개발 서버 실행
uvicorn main:app --reload

# 또는 시작 스크립트 사용
./startup.sh
```

### 5. API 테스트
브라우저에서 다음 URL로 접속:
- **API 문서**: http://127.0.0.1:8000/docs
- **헬스체크**: http://127.0.0.1:8000/health

## 📚 API 문서

### 🔍 주요 엔드포인트

#### 요구사항 분석
```http
POST /requirements/generate
Content-Type: application/json

{
  "project_overview": "프로젝트 개요",
  "existing_requirements": "기존 요구사항"
}
```

#### 추천 시스템
```http
POST /recommendation/generate
Content-Type: application/json

{
  "project_type": "웹 애플리케이션",
  "requirements": ["기능1", "기능2"]
}
```

#### 헬스체크
```http
GET /health
```

### 📖 상세 API 문서
서버 실행 후 http://127.0.0.1:8000/docs 에서 전체 API 문서를 확인할 수 있습니다.

## 🔄 CI/CD 파이프라인

### GitHub Actions 워크플로우

프로젝트는 자동화된 CI/CD 파이프라인을 사용합니다:

1. **테스트 단계**: 코드 문법 검사 및 기본 테스트
2. **버전 관리**: 커밋 메시지 기반 자동 버전 업데이트
3. **빌드 및 푸시**: Docker 이미지 빌드 및 Docker Hub 푸시
4. **배포**: AWS EC2 자동 배포

### 🏷️ 태그 및 릴리스
- **major**: 주요 기능 추가 (v1.0.0 → v2.0.0)
- **minor**: 새로운 기능 추가 (v1.0.0 → v1.1.0)
- **patch**: 버그 수정 (v1.0.0 → v1.0.1)

## 🐳 Docker 배포

### Docker 이미지 빌드
```bash
cd FastAPI
docker build -t pja-project:latest .
```

### Docker 컨테이너 실행
```bash
docker run -d \
  --name pja-fastapi \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your_api_key" \
  pja-project:latest
```

### Docker Compose 사용
```bash
docker-compose up -d
```

## 🧪 테스트
pja_MLOps/
└── 📁 Test/                      # 테스트 및 실험
    ├── Fine-tuning/              # 모델 파인튜닝
    ├── LLM_test/                 # LLM 테스트
    └── QLoRA_test/               # QLoRA 실험

## 📊 모니터링

### 📈 메트릭 수집
- API 응답 시간
- 요청 수 및 성공률
- 에러 발생률
- 리소스 사용량

### 🚨 알림 시스템
- Slack 통합 알림
- 에러 발생 시 자동 알림
- 배포 상태 알림

### 👥 기여자
- **JMinkyu**   - 프로젝트 리드
- **listgreen** - 백엔드 개발 DevOps 및 인프라
- **onxtorlo**  - 모델 개발

## 📞 문의 및 지원

- **이슈 트래커**: [GitHub Issues](https://github.com/PJA-ProJect-Assistant/pja_MLOps/issues)
- **버그 신고**: Issue 생성을 통해 버그를 신고해주세요
- **기능 요청**: Enhancement 라벨로 새로운 기능을 제안해주세요

## 📋 체크리스트

### ✅ 개발 환경 설정
- [ ] Python 3.9+ 설치
- [ ] uv 패키지 매니저 설치
- [ ] OpenAI API 키 발급
- [ ] 가상환경 설정
- [ ] 의존성 설치

### ✅ 배포 준비
- [ ] Docker 설치
- [ ] AWS 계정 및 EC2 설정
- [ ] GitHub Secrets 설정
- [ ] 환경변수 구성

## 🎉 마무리

PJA MLOps Project는 AI 기반 프로젝트 관리의 새로운 패러다임을 제시합니다. 지속적인 개선과 확장을 통해 더 나은 프로젝트 관리 경험을 제공하겠습니다.

**Happy Coding! 🚀**

---
*마지막 업데이트: 2025년 6월*