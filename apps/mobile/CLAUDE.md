# apps/mobile — React Native (Expo) 프론트엔드

담당: 가은

## 이 폴더 컨텍스트
- 루트 CLAUDE.md의 FIND/CHECK/PLAN 구조를 화면 단위로 구현
- 백엔드는 `apps/server` (FastAPI) — API 계약은 `apps/server/CLAUDE.md`의
  엔드포인트 표 참고, 변경 필요 시 민경/이재인과 협의 후 이 파일에도 반영

## 화면 ↔ 단계 매핑 (예시 — 실제 화면 구조에 맞게 채워넣기)
- 홈/검색 화면 → FIND
- 지원사업 상세/자격확인 화면 → CHECK (신뢰도 배지 🟢🟡🟠🔴 UI 표시 필수)
- 신청 플랜/체크리스트 화면 → PLAN
- 회원가입/프로필 → FIND의 기반 데이터 수집 (FR-PROF-002 계열)

## 컨벤션
- Expo 관리형 워크플로우 유지 (bare workflow 전환 논의 없이 임의 변경 금지)
- 상태관리: (팀 결정 사항 기입)
- API 호출 base URL은 `.env`의 `EXPO_PUBLIC_API_BASE_URL` 사용, 하드코딩 금지

## 하지 말 것
- CHECK 결과를 UI에서 신뢰도 없이 이분법(자격 O/X)으로 표시하지 말 것
