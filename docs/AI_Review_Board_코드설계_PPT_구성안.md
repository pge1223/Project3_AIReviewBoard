# AI Review Board 코드 설계 — PPT 구성안

> 기준: 2026-07-29 현재 저장소의 실제 코드와 최신 서비스 방향 문서를 함께 확인해 정리  
> 권장 발표 시간: 12~15분 / 본문 12장 + 부록 3장

## 발표 핵심 메시지

AI Review Board는 공모전 공고·수상작·사용자 문서를 하나의 근거 데이터층으로 연결하고,  
작성 전에는 두 AI 위원이 아이디어를 함께 구체화하며, 작성 후에는 기준 기반 병렬 심사와 문서 위 피드백을 제공하는 워크벤치다.

---

## 1. 표지

### 제목

**AI Review Board 코드 설계와 서비스 흐름**

### 부제

근거 기반 공모전 분석부터 아이디어 회의, 문서 심사, 개선 추적까지

### 발표자 한마디

“단순한 생성형 AI 첨삭이 아니라, 공모전의 공식 기준과 실제 자료를 근거로 사용자의 기획 과정을 전후로 지원하는 구조입니다.”

---

## 2. 서비스는 ‘작성 전’과 ‘작성 후’를 하나의 데이터 흐름으로 연결한다

### 화면에 넣을 내용

```text
공모전 정보 입력
        ↓
공고문·신청서·수상작 수집 및 RAG 색인
        ↓
┌────────────────────┬────────────────────┐
│ 작성 전             │ 작성 후             │
│ 아이디어 발굴        │ 문서 피드백          │
│ AI 위원 토론         │ AI 위원 병렬 심사    │
│ 신청서 초안 생성     │ 점수·근거·수정 제안  │
└────────────────────┴────────────────────┘
        ↓
사용자 수정·선택·재평가
        ↓
버전별 개선 이력 축적
```

### 핵심 문구

- 공통 입력: 공고문, 신청 양식, 수상작, 사용자 프로필, 사용자 기획서
- 공통 기반: 문서 파싱·청킹·임베딩·검색·근거 연결
- 두 사용자 여정이 프로젝트 ID와 문서 ID를 중심으로 이어짐

### 발표 포인트

작성 전과 작성 후는 별도 서비스처럼 보이지만, 코드에서는 같은 프로젝트·문서·RAG 자산을 재사용한다.

---

## 3. 전체 시스템은 UI·API·AI 오케스트레이션·데이터 계층으로 분리된다

### 화면에 넣을 아키텍처

```text
[React / Vite Frontend]
프로젝트·문서 업로드·아이디어 회의·워크벤치·리포트
                 │ REST / NDJSON Stream / WebSocket
                 ▼
[FastAPI Backend]
인증·프로젝트·문서·회의·리포트·미디어 API
                 │
        ┌────────┴────────┐
        ▼                 ▼
[AI Orchestration]   [Persistence]
LangGraph 회의        MongoDB
RAG 파이프라인         Chroma Vector DB
점수 규칙 엔진         업로드 파일 저장소
LLM 호출 추적
```

### 파트별 책임

| 파트 | 핵심 책임 |
| --- | --- |
| Frontend | 사용자 여정, 상태 표시, 스트리밍 회의 출력, 문서 위 피드백 |
| FastAPI | 인증·권한, 입력 검증, 비동기 처리, AI 모듈 조합, 응답 계약 |
| RAG | 문서 변환·파싱·청킹·임베딩·검색·출처 보존 |
| Meeting | AI 위원 역할 분리, 토론 순서, 상태 전이, 결과 종합 |
| Scoring | LLM 의견과 분리된 결정론적 점수 계산 |
| Persistence | 사용자·프로젝트·문서·세션·결과·근거의 영속화 |

### 코드 근거

- 앱 조립: `backend/app/main.py`
- 화면 라우팅: `frontend/src/App.jsx`
- AI 회의 그래프: `ai/meeting/graph/`
- RAG 모듈: `ai/rag/`

---

## 4. 사용자 요청은 Frontend API 계층을 거쳐 기능별 FastAPI 라우터로 전달된다

### 화면에 넣을 흐름

```text
React Page / Component
        ↓
frontend/src/api/*
        ↓ Authorization Header
FastAPI Router
        ↓
Repository + AI Service
        ↓
MongoDB / Chroma / LLM
        ↓
JSON 또는 NDJSON Streaming 응답
```

### 주요 API 묶음

| 기능 | Frontend | Backend |
| --- | --- | --- |
| 인증 | `authApi.js` | `/auth` |
| 프로젝트 | `projectApi.js` | `/projects` |
| 문서 | `documentApi.js` | `/documents` |
| 아이디어 회의 | `ideationConversationApi.js` | `/ideation-conversation` |
| 신청서 코치 | `ideationFormCoachApi.js` | `/ideation-form-coach` |
| 문서 피드백 | `workbenchApi.js` | `/workbench` |
| 미디어 | `mediaApi.js` | `/media` |

### 설계 의미

화면은 AI 구현을 직접 알지 않고 API 계약만 사용하므로, 프롬프트·RAG·그래프 변경이 UI 전체로 번지는 것을 줄인다.

---

## 5. 문서는 ‘수집 → 정규화 → 파싱 → 청킹 → 임베딩 → 색인’ 순서로 지식화된다

### 화면에 넣을 흐름

```text
파일 업로드 또는 URL 입력
        ↓
형식·권한·소유 프로젝트 검증
        ↓
HWP 변환 / 웹 콘텐츠 정제
        ↓
PDF·DOCX·PPTX·HTML 통합 파싱
        ↓
페이지·슬라이드·섹션 메타데이터 보존
        ↓
문맥 단위 청킹
        ↓
KURE 임베딩
        ↓
Chroma 저장 + MongoDB 처리 상태 갱신
        ↓
ready / failed 상태 제공
```

### 핵심 설계

- 파일과 URL을 하나의 문서 처리 계약으로 수렴
- 원문 위치 정보를 남겨 최종 피드백에서 출처로 역추적
- 무거운 색인은 백그라운드로 처리하고 상태 조회·재시도 API 제공
- HWP 변환 가능 여부를 서버 시작 시 진단해 `/health`에 반영
- 임베딩 모델 초기화는 싱글턴과 락으로 중복 로딩 방지

### 코드 근거

- 문서 API·백그라운드 색인: `backend/app/api/routes/documents.py`
- 변환: `ai/rag/converters/`
- 파싱: `ai/rag/parsers/`
- 청킹: `ai/rag/chunking/`
- 임베딩·저장: `ai/rag/embedding/`, `ai/rag/retrieval/`

---

## 6. RAG는 ‘무엇을 찾았는가’뿐 아니라 ‘어디서 찾았는가’를 결과까지 전달한다

### 화면에 넣을 흐름

```text
현재 과업·위원 역할·평가 항목
              ↓
역할별 검색 질의와 필터 구성
              ↓
Chroma Top-K 검색
              ↓
문서 유형·프로젝트·출처별 범위 제한
              ↓
관련성·충분성·최신성 검토
              ↓
근거 묶음 → AI 위원 프롬프트
              ↓
주장 ↔ 인용 문장 ↔ 원문 위치 연결
```

### 검색 대상

- 공고문과 공식 심사기준
- 신청 양식과 필수 작성 항목
- 과거 수상작·유사 사례
- 사용자의 아이디어와 답변
- 사용자가 제출한 기획서

### 품질 안전장치

- 도메인 및 문서 유형 분류
- 평가 기준 추출과 정규화
- 근거 충분성 판정
- 주장 근거 연결과 인용 추출
- 외부 자료의 출처 검증·최신성 검사
- 검색·생성 품질 평가 스크립트와 회귀 테스트

### 발표 포인트

일반 챗봇과의 차이는 답을 잘 쓰는 것보다, 답의 판단 근거를 원문까지 다시 따라갈 수 있게 만든 데 있다.

---

## 7. 작성 전 흐름은 문제 발견부터 신청서 초안까지 상태 기반으로 진행된다

### 화면에 넣을 흐름

```text
공고 분석
  ↓
문제 영역 발굴
  ↓ 사용자 초점 선택
문제 정의
  ↓
아이디어 발산
  ↓
기획·개발 위원 충돌 조정 및 결합
  ↓
기획 관점 검증 → 기술 관점 검증
  ↓ 사용자 콘셉트 확정
전문가 라운드테이블
  ↓
아이디어 캔버스 갱신
  ↓
결과 정리 → 신청서 항목별 초안
```

### LangGraph가 관리하는 것

- 현재 단계와 다음 발언자
- 사용자 결정을 기다리는 중단 지점
- 기획 위원·개발 위원·진행자의 역할
- 반복 횟수와 실패 상태
- 회의 메시지, 근거, 아이디어 캔버스
- 신청서 필드별 `empty / draft / confirmed` 상태

### 실시간 UX

- 서버는 NDJSON 이벤트 스트림으로 발언·상태·완료 이벤트를 전송
- 프론트는 reducer로 중복 제거, 부분 문자열 표시, 재연결 상태를 관리
- 취소, 실패 노드 재시도, 다음 전문가 발언 이어가기를 별도 API로 제공

### 코드 근거

- 그래프: `ai/meeting/graph/ideation_conv_build.py`
- 실행·상태: `ai/meeting/graph/ideation_conv_run.py`, `ideation_conv_state.py`
- API: `backend/app/api/routes/ideation_conversation_preview.py`
- 스트림 해석: `frontend/src/pages/board/ideationStreamReducer.js`

---

## 8. 작성 후 흐름은 독립 심사와 규칙 기반 채점으로 설명 가능성을 확보한다

### 화면에 넣을 흐름

```text
사용자 제출 문서 + 공고 기준 + 검색 근거
                   ↓
       ┌───────────┴───────────┐
       ▼                       ▼
기획 위원 독립 심사       개발 위원 독립 심사
문제·가치·차별성          구현·데이터·보안·운영
       └───────────┬───────────┘
                   ↓
           Python 점수 규칙 엔진
                   ↓
             위원장 종합 의견
                   ↓
점수·강점·리스크·수정 우선순위·근거
```

### 중요한 설계 원칙

- 위원 노드는 LangGraph에서 병렬 실행
- 각 위원은 다른 위원의 결과를 보기 전에 독립 판단
- 최종 점수는 LLM이 임의로 정하지 않고 Python 규칙으로 계산
- 기준별 만점, 필수 항목, 근거 부족, 점수 상한을 계산에 반영
- 위원장 종합은 리뷰·채점 이후 실행해 결과를 설명 가능한 서사로 구성

### 코드 근거

- 병렬 그래프: `ai/meeting/graph/build.py`
- 회의 실행: `ai/meeting/graph/run.py`
- 리뷰·채점·위원장 노드: `ai/meeting/graph/nodes/`
- 점수 계산: `ai/meeting/scoring/calculator.py`
- 분석 API: `backend/app/api/routes/meetings.py`

---

## 9. 워크벤치는 AI 결과를 문서의 실제 위치와 연결해 행동 가능한 피드백으로 바꾼다

### 화면에 넣을 구성

```text
┌───────────────────────────────┬─────────────────────┐
│ 사용자 문서                   │ 선택 피드백          │
│                               │                     │
│ 하이라이트된 문장 ────────────→│ 왜 문제인가          │
│ 메모 핀                       │ 연결 평가 기준       │
│ 페이지/문단 위치              │ 근거 인용            │
│                               │ 수정 제안            │
└───────────────────────────────┴─────────────────────┘
```

### 제공 기능

- 인용 문구를 원문 문단과 매칭
- 문맥 이상·논리 비약 탐지
- 오탈자 검사
- 공고문 기준의 페이지 수·형식 검사
- AI 피드백 결과 스냅샷 저장
- PDF·HTML 미리보기

### 코드 근거

- 화면: `frontend/src/pages/board/WorkbenchScreen.jsx`
- API: `backend/app/api/routes/workbench.py`
- 문서 보기: `frontend/src/pages/board/PdfDocumentView.jsx`

---

## 10. 데이터 모델은 프로젝트를 중심으로 모든 결과의 추적 가능성을 만든다

### 화면에 넣을 관계

```text
User
 └─ Project
     ├─ Documents
     │   ├─ criteria / announcement
     │   ├─ application_form
     │   └─ target submission versions
     ├─ RAG Chunks
     ├─ Ideation Sessions
     │   ├─ messages
     │   ├─ evidence traces
     │   └─ application form draft
     └─ Meetings
         ├─ reviewer results
         ├─ score result
         ├─ chair summary
         └─ evidence links
```

### 저장소별 역할

| 저장소 | 저장 내용 |
| --- | --- |
| MongoDB | 사용자, 프로젝트, 문서 메타데이터, 세션, 심사 결과, 처리 상태 |
| Chroma | 검색 가능한 문서 청크 벡터와 출처 메타데이터 |
| 파일 저장소 | 업로드 원본, 변환본, 미리보기 자료 |

### 추적 키

`user → project_id → document_id / session_id / meeting_id`

### 발표 포인트

결과만 저장하는 것이 아니라 입력 문서, 검색 근거, 위원 의견, 계산 결과를 분리해 저장하므로 재평가와 버전 비교가 가능하다.

---

## 11. 신뢰성은 상태 관리·실패 격리·재현 가능한 계산으로 확보한다

### 화면에 넣을 내용

| 위험 | 코드의 대응 |
| --- | --- |
| 긴 문서 처리 | 백그라운드 색인, 진행 상태 조회 |
| 중복 색인 요청 | 싱글턴 서비스와 락 |
| 스트리밍 중단 | 요청 ID, 취소, 이어가기, 재시도 |
| 일부 AI 노드 실패 | 단계별 상태와 실패 노드 재실행 |
| 근거 없는 답변 | 근거 충분성·claim grounding |
| 점수 변동 | Python 규칙 엔진과 계산 버전 |
| 문서 형식 차이 | 통합 파서와 변환 진단 |
| 기능 장애 전파 | 미디어·외부 조사 등 부가 기능의 fallback |

### 테스트 범위

- 문서 업로드·재색인·HWP/OCR
- RAG 청킹·검색·근거 연결·품질 평가
- 아이디어 회의 상태 전이·스트리밍·취소
- 위원 병렬 심사·점수 일관성·재평가
- 신청서 초안과 행정 필드 제외

### 발표 포인트

AI 품질만 테스트하는 것이 아니라 상태 전이, API 계약, 점수 재현성, 장애 복구까지 테스트 대상으로 둔다.

---

## 12. 현재 구조는 도메인과 AI 위원을 추가할 수 있는 확장형 설계다

### 화면에 넣을 내용

```text
현재 MVP
IT 공모전
기획 위원 + 개발 위원
        ↓
확장
정책·공공성 / 디자인 / 재무·사업성 위원
        ↓
도메인 패키지
공고문 + 수상작 + 루브릭 + 검색 프로필 + 프롬프트
```

### 확장 시 바뀌는 부분

- 도메인 분류 규칙
- 평가 루브릭과 배점
- 위원 persona와 담당 기준
- 검색 프로필과 근거 데이터
- 프롬프트와 결과 표현

### 유지되는 부분

- React 사용자 여정
- FastAPI 기능 경계
- 문서 수집·파싱·색인 파이프라인
- LangGraph 상태 오케스트레이션
- 점수 계산 계약
- MongoDB·Chroma 추적 구조

### 마무리 문구

**AI Review Board의 핵심은 여러 AI를 보여주는 것이 아니라, 서로 다른 전문 판단을 공식 기준과 원문 근거에 연결하고 사용자의 수정 행동까지 이어 주는 구조다.**

---

# 부록

## A. 폴더 구조와 담당 역할

```text
frontend/
  src/pages/board/     핵심 워크벤치·아이디어 회의 화면
  src/api/             Backend API 어댑터

backend/
  app/main.py          FastAPI 앱 조립과 수명주기
  app/api/routes/      기능별 HTTP 경계
  app/repositories/    MongoDB 접근 계층
  app/schemas/         요청·응답 계약

ai/
  rag/                 문서 수집·파싱·검색·근거 연결
  meeting/graph/       LangGraph 회의와 상태 전이
  meeting/scoring/     결정론적 점수 계산
  meeting/prompts/     역할·단계별 프롬프트
  media/               TTS·립싱크 관련 실험 및 구성

contracts/
  schemas/             모듈 간 결과 JSON 계약
  mocks/               통합 전 독립 개발용 예시 데이터
```

## B. 대표 API 흐름

### 작성 전

```text
POST /documents/fetch-url 또는 /documents/{project_id}
→ POST /documents/{project_id}/announcement-analysis
→ POST /ideation-conversation/start/stream
→ POST /ideation-conversation/{session_id}/reply/stream
→ POST /ideation-conversation/{session_id}/finalize
→ POST /ideation-conversation/{session_id}/form-draft
```

### 작성 후

```text
POST /documents/{project_id}
→ GET /documents/{project_id}/{document_id}/status
→ POST /projects/{project_id}/mentor-candidates
→ POST /projects/{project_id}/analyze
→ GET /projects/{project_id}/analyze/progress
→ GET /projects/{project_id}/report
→ POST /workbench/{project_id}/quotes
→ GET /projects/{project_id}/comparison
```

## C. 발표에서 구분해서 말할 현재 구현 상태

- `작성 후 심사`, 문서 처리, RAG, 워크벤치 API는 기본 앱 라우터에 등록되어 있다.
- 대화형 아이디어 회의 라우터는 현재 `ENABLE_IDEATION_PREVIEW` 설정이 켜질 때 등록되는 개발용 프리뷰 경로다.
- 신청서 코치 라우터 구현은 존재하지만, `backend/app/main.py`의 기본 라우터 등록 목록에는 포함되어 있지 않다.
- 미디어는 사용 가능한 화자 조회와 WebSocket 중계 API가 구현되어 있으나, TTS·립싱크 생성 전체가 주 분석 흐름의 필수 단계로 묶여 있지는 않다.
- 최신 서비스 방향 문서의 목표와 초기 아키텍처 문서 사이에는 차이가 있으므로, 발표 시 “현재 구현”과 “확장 목표”를 분리해 설명하는 것이 정확하다.

