# E2E QA — 개인 맞춤형 피드백 루프 (버전 추적형 User RAG)

> 작성: 경이 / 2026-07-22
> 범례: `[x]` 검증 완료 · `[ ]` 확인 대기(브라우저 실행 필요) · 🔎 확인 방법
> 환경: 로컬 프론트(`localhost:5173`) → 배포 백엔드(`http://101.79.25.179:8000`)

---

## 0. 사전 검증 — 로직·계약 (코드로 검증 완료, 서버 불필요)

- [x] **개인화 로직 동작**: 비전공자 → `HARD/detailed`, 전공자 → `EASY/brief`
  🔎 `scratchpad/qa_personalization.py` 실행 결과로 확인 (실제 `classify_impl_difficulty`/`attach_impl_guides` 호출)
- [x] **프로필 계약 정합**: 윤한 `backend/app/schemas/user.py` = 내 분류기 필드 동일
  (`education.is_technical_major/degree/graduation_status`, `experience.internship_months/competition_count/award_count`,
  `github.public_repos/followers/total_stars/primary_languages`)
- [x] **degree 영문 enum 강제**: 한글로 오면 FastAPI가 422 거부 (윤한 `Literal["bachelor","master","phd","other"]`)
- [x] **리포트 개인화 훅 존재**: `is_technical_persona`로 개발 위원(`technical_feasibility`) 지적만 골라
  `attach_impl_guides(dev_feedback, user_profile, llm_call)` → 응답에 `impl_guides` 포함 (`backend/app/api/routes/meetings.py`)
- [x] **안전 폴백**: 프로필 미제출/필드 누락 → `hard` (테스트 `test_no_profile_returns_none`, `test_missing_github_and_experience_defaults_to_hard`)
- [x] **버전 비교 로직**: `build_revision_comparison` + `GET /projects/{id}/comparison` 엔드포인트
- [x] **자동화 테스트 154개 통과** (`ai/meeting/tests/`)

---

## 1. 프로필 저장 — 개인화의 입력 (브라우저)

- [x] 마이페이지에서 **🅐 비전공자** 프로필 입력 → 저장 ✅ (2026-07-22 오전 10:10 저장, sophia@test.com)
  (전공=비전공 · 학위=학사 · 졸업=졸업 · 인턴 0 · 공모전 0 · 수상 0 · GitHub 비움)
- [ ] 저장값 재확인 — `degree`가 **영문 `bachelor`**로 저장됐는지
  🔎 F12 → Network → `profile` 요청 → Response 확인 (한글이면 422 나야 정상)

## 2. 문서 분석 → 리포트 개인화 (핵심 기능)

- [x] `QA_기획서_샘플.pdf` 업로드 → 분석 시작 → 회의 완료 ✅ (2026-07-22, 로컬 백엔드+실제 OpenAI로 회의 성공, domain=competition, 위원 4명에 technical_feasibility 포함)
- [x] **백엔드 `/report` 응답에 개인화된 `impl_guides` 실측 확인** ✅
  🔎 실제 응답: `feedback_id:"feasibility"` / `level:"hard"` / `verbosity:"detailed"` /
  `label:"구현 난이도 · 어려울 수 있음"` + OpenAI가 생성한 비전공자용 단계별(①②③) prose
  (sophia@test.com 비전공자 프로필 → HARD/detailed 정상). project=6a6036f9958e01e455f803e0
- [x] **🅑 전공자로 바꿔 저장 → 리포트 다시** → 같은 지적이 **`level:"easy"` / `verbosity:"brief"` / 짧은 `prose`** ✅
  ← ⭐ **"개인 맞춤형" E2E 증명 완료** (2026-07-22). **동일 프로젝트 `6a6036f9`, 동일 지적 `feasibility`**:
  비전공자=`hard/detailed`(긴 ①②③ 입문) → 전공자=`easy/brief`(짧은 전문가 요약 "API 엔드포인트 설계·DB
  스키마…캐싱/로드밸런싱"). 재분석 없이 프로필(전공 True·인턴6·공모전1·수상1·GitHub cobong16 Python)만
  바꿔 /report 재호출 → attach_impl_guides가 프로필로 즉시 재계산됨을 실측.
  - [x] 부수 검증: 개발 위원 판단이 좋은(acceptable) 프로젝트들은 `impl_guides=0`(resolved라 가이드 생략) — 의도된 동작
  - 화면(embedded VersionTrackerTestPage)도 프로필 "컴퓨터공학 전공자" 잠김 + 구현 난이도 "쉬움" 표시 확인
- [ ] ⚠️ **UI 렌더링 갭**: 가은님 `WorkbenchScreen.jsx`는 `/report`의 `impl_guides`를 안 읽음
  (issues/suggestions만 렌더) → "완성된 리포트"가 화면에 안 나오는 원인. 데이터는 옴, 그림만 없음.

## 3. 버전 추적 — 상승세 (브라우저)

- [ ] 문서 수정 → 재업로드 → 재분석 (2번째 회의)
- [ ] 항목별 증감 + 해결/신규 지적 표시
  🔎 `GET /projects/{id}/comparison?before=&after=`

## 4. 엣지 — 안전 폴백 (브라우저)

- [ ] 프로필 **미제출** 계정으로 리포트 → `impl_guides`가 `hard` 폴백 (무음실패 없이)

---

## 진행 로그

- 2026-07-22: 0번(로직·계약) 전부 검증 완료. 1~4번은 로컬 프론트→배포 백엔드로 진행 예정.
- 2026-07-22: **핵심 E2E(2번) 화면+데이터 양쪽 완결.** 비전공자(hard/detailed) → 전공자(easy/brief)
  뒤집힘을 동일 프로젝트 `/report`로 실증. 완성 리포트 단계(가은 /board 흐름에 embedded)에서
  버전추적형 리포트 디자인 + 프로필 잠금 + 구현 난이도 개인화 렌더 확인.
- 발견한 후속 이슈(내 영역 아님 — 담당자 확인 필요):
  - (가은 MyPage) 제출 정보 요약 줄 "이력·교육 수준: **undefined undefined** · 전공 …" — 표시 버그.
    판정엔 무영향(is_technical_major boolean 사용). 요약 렌더가 수집 안 하는 필드(전공분야/학교?)를 참조.
  - (가은 폼 or 윤한 스키마) 마이페이지에서 학위 "학사" 선택했는데 저장된 `profile.education.degree=None`.
    지금 테스트는 bachelor=0점이라 easy 판정에 영향 없지만, 석사/박사 선택 시 +1이 안 붙을 위험 → 폼-스키마
    degree 매핑 확인 필요.
