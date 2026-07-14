# apps/server — FastAPI + MongoDB (NCP)

담당: 민경(CRUD, 페르소나) / 이재인(CHECK, PLAN 로직) / 김윤한(NCP 서버 설정)

## 이 폴더 컨텍스트
- FIND는 `services/rag`에서 검색 결과를 받아오는 쪽이고, 이 폴더는 그 결과를
  받아 CHECK/PLAN 로직과 CRUD API를 제공
- 공용 시스템 프롬프트(페르소나)는 한 곳에서 관리 —
  `services/rag/persona/system_prompt.md` (민경 담당) 를 import해서 쓸 것,
  각 엔드포인트마다 프롬프트 복붙 금지

## CHECK 단계 규칙 (엄격 적용)
- 신뢰도 4단계 🟢🟡🟠🔴 중 하나를 반드시 반환
- 근거 문서(공고 원문 청크)를 인용 없이 자격 판정하지 말 것 —
  할루시네이션 방지 규칙 최우선
- 애매한 경우에만 clarifying question 트리거 (전체 대화형 슬롯필링 아님)

## API 엔드포인트 (예시 — 실제 계약 확정되면 갱신)
| Method | Path | 설명 | 담당 |
|---|---|---|---|
| POST | /find | 하이브리드 검색 | 김용준 연동 |
| POST | /check | 자격 판정 | 이재인 |
| POST | /plan | 타임라인/체크리스트 추출 | 이재인 |
| CRUD | /profile | 사용자 프로필 | 민경 |

## 환경변수 (server 전용, `.env.example` 참고)
- `MONGODB_URI`, `PERSO_API_KEY`, `NCP_*` 등은 절대 코드에 하드코딩 금지
