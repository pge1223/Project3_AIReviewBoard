# mky Devlog

## 2026-07-26 (이어서 — 종합 리포트 UI 정리 + 해결됨 근거 검증 + 인용 원문 게이트)

- 한 일(UI 정리 — 경이 시안 스크린샷 반영, 전부 공용 컴포넌트라 전 버전·전 위원 일괄 적용):
  - **히어로 카드**: "버전 추적형 USER RAG" 배지·"IT 공모전·개인 맞춤형 피드백 루프" 제목 삭제
    → 가운데 큰 "종합 리포트" 제목. 설명 줄 순서(AI 피드백 → 기획·개발 위원) 교체, 본문 글씨
    검정 통일, 경고 문구 "위원" 삭제.
  - **점수 체계표**: 상단 "총점은 참고용" 배너를 헤더 ⚠️ 클릭 팝업으로 이동(부제 "측정 가능
    항목만 채점"만 남김, "심사위원의"·"(예: 안전성·윤리성)" 문구 삭제).
  - **상세 화면**: 뒤로가기 "점수 추적 그래프"로 개명, 정렬 안내·하단 API 각주·위원 부제·카드
    우상단 판정 배지(JudgmentChange) 제거, "이 위원 점수"→"점수", 상태 필터 칩 무채색(회색
    배경+검정 글씨) 통일, 프로필 카드 "내" 삭제·"개인별 맞춤"·비전공자 검정 굵게.
  - **Why 플로우**: 버튼 "왜 이 점수·피드백인가요?"→"Why?"(검정), STEP N 라벨 검정, STEP 1
    "이 항목 채점에 반영" 라벨 삭제, 플로우 프레임 색을 위원 accent가 아니라 **지적 상태색**
    (보완 필요=황토/신규=주황)으로 통일. STEP 1 인용 부재 문구를 "내용 부재: …"로 교체.
    보조 자료는 기본 접힘 탭(클릭 시 세부 인용 펼침).
  - **막대 숫자 가독성**: 소수점 점수(6.67)가 고정폭을 넘쳐 "/10"과 붙어 보이던 것 분리.
  - **점수 변화 팝업**: 항목 카드 +N점 배지 클릭 → 이전→현재 점수, 해결/신규/잔여 건수,
    판정·상한, 점수 체계 한 줄 설명(전부 그 항목 실데이터로만 구성).
- 한 일(근거 기반 강화 — 할루시네이션 2건 실측 차단):
  - **해결됨 근거 검증+중복 제거**(buildVersionsFromHistory): resolved_issues가 직전 버전의
    실제 지적과 매칭될 때만 표시(2-gram 유사도 ≥0.66/포함), 직전 지적 1건당 해결됨 1건으로
    합침(해결 6→2 실측), 문구는 재표현본이 아니라 **직전 버전 지적 원문**을 취소선("v1.0
    보완 필요 …" + 끝에 해결됨). 직전 버전에 없던 해결됨(예: "구체적인 확산 계획 부족")은
    근거 없음으로 미표시.
  - **인용 원문 게이트**(realCitations): evidence의 quote는 위원 LLM 자기보고라, **색인된 청크
    원문(text)에 실제 존재할 때만** 표시하고 아니면 청크 원문으로 대체, 청크 미상은 제외 —
    붙임2 p.9에 없는 문장(rubric 설명문)이 실제 파일명·페이지에 붙어 보이던 사고(경이 직접
    발견) 차단. 저장 데이터 불변·표시 계층 게이트라 기존 회의에도 즉시 적용.
  - **중심·보조 자료 전 버전 노출**: 공고 자료 인용은 프로젝트 공통이라 v1.0~에도 STEP 2에
    표시(이전엔 최신 버전만). 제출 문서 인용은 버전마다 파일이 달라 최신 버전에만(오귀속 방지).
  - **calibration S4 사유 축별 분기**(백엔드): 실현 가능성 축 검사는 원래 프로토타입·인프라·
    일정 용어인데 사유 문구가 "모델·알고리즘…"으로 고정돼 기술축 표현이 잘못 노출(경이 지적)
    → feasibility면 "구현 계획의 실체(프로토타입·인프라·개발 일정 등 실행 방법)가 명시되지
    않음"으로 분기. ai/meeting 테스트 366개 통과.
- 결정/이유:
  - **표시 계층 근거 게이트 우선**: 저장된 회의 데이터를 고치지 않고 프론트에서 원문 검증 —
    기존 회의 결과에도 소급 적용되고, 백엔드(EvidencePool) 저장 시점 검증은 후속 검토.
  - 해결됨 유사도 임계 0.66: 같은 지적의 어미 변형(실측 0.82·0.64…)은 흡수하되 다른 지적
    ("확산 계획" vs "성과 지표", 실측 0.55)은 구분되는 지점으로 캘리브레이션.

## 2026-07-26 (Why 플로우 마감 다듬기 + Comprehensive_Report 발표 자료 세트)

- 한 일(UI — 전부 규칙 기반이라 모든 항목·지적에 자동 적용):
  - **우선순위 팝업 가림 수정**: 각 항목 카드가 vt-fade 애니메이션으로 자체 스태킹 컨텍스트를
    가져 팝업이 다음 카드에 가려지던 문제 — 팝업 열린 카드를 zIndex로 형제 위로 올림.
  - **"이렇게 고치세요" 박스 완전 제거**: STEP 4와 중복 — 개선 제안은 Why 플로우 STEP 4에서만.
  - **STEP 1 재구성**: 제목 '제출 문서 근거' + 라벨 '이 항목 채점에 반영' 하나로 정리. 인용
    통합 후 **부분 문장이 더 긴 인용에 포함되면 가장 완전한 인용만** 남김(같은 문장 3회 노출
    해소의 일반 규칙). 헤더의 지적 문구에서도 "(p.N) '…'" 인용부를 잘라 요지만(검증된 지적만).
  - **STEP 2 다듬기**: 중심/보조 라벨 검정·굵게. 보조 자료 인용을 "파일명·페이지 번호 헤더 +
    섹션 제목 + ㅇ 불릿 줄" 구조로 가독성 개선(표 잔재 잡음 "※<작성 요령>" 제거 — 원문 문장은
    불변, 줄 나눔·잡음 제거만). 인용 선별은 **KURE 의미 유사도(score) 내림차순 상위 4개**
    (단어 매칭이 아니라 의미 기반 — evidence에 저장된 유사도 활용).
- 한 일(발표 자료 — `docs/prompts/Comprehensive_Report/`, 폴더명 변경: 트러블슈팅_채점엔진_버전추적 →):
  - **종합리포트_평가지표_설계가이드.md** — 4대 품질 축(재현성·변별력·근거 기반성·기준 충실성)
    정의/측정 방법/실측값 + 가이드북 9장 지표(Faithfulness·G-Eval 등) 매핑 + 발표 핵심 메시지.
  - **종합리포트_개선실험_Before_After.md** — 가이드북 9.5-④ 대응 실험 5건 표.
  - **종합리포트_설계검증_시각화.ipynb** + PNG 4종(2026_07_25_시각화자료/) — fig1 재현성
    (±32.5→±1.5), fig2 변별력(21.5vs77·55.5vs81), fig3 기준 충실성(배점 원문 강제·만점 85),
    fig4 근거 기반성(항목별 근거 연결 실측 + **지적 인용 14건 중 검증 통과 6건만 표시·8건 차단**
    — 게이트 실작동 실증). 노트북 실행 검증 완료(matplotlib 설치, Malgun Gothic).
  - 원칙: 전 수치 실측값(재현 경로 devlog 명시), Ragas 배치 측정 등 검색 품질은 용준(RAG) 영역으로 구분 표기.

## 2026-07-25 (이어서 — Why 플로우 중복 제거 + 우선순위 팝업 + 중심/보조 자료 구분)

- 한 일(경이 UI 검수 반영):
  - **중복 제거**: ① STEP 2의 "이 기준 대비 지적: …" 라인 삭제(아코디언 헤더와 중복)
    ② Why 플로우가 열리면 "이렇게 고치세요" 박스 숨김(STEP 4와 중복) ③ STEP 1 같은 인용
    2회 표시 제거 — 위원 여러 명이 같은 청크를 인용해도 1번만(정규화 dedupe), 지적 전용
    인용(f.ref)과 겹치는 문장도 "함께 쓰인 원문"에서 제외.
  - **우선순위 팝업**: 카드에 늘어놓던 "근거 기반 점수 상한 적용" 배너를 제거하고,
    [우선순위 N ⓘ] 배지 클릭 팝업으로 이동 — 판정·감점(배점 대비)·"판정 심각도→감점
    비율 순" 설명 + 상한 상세(위원 제안→상한·사유).
  - **중심/보조 자료 구분(경이 확정 기준)**: 중심 자료 = **파일명에 '공고문'이 포함된 파일**
    (URL 스크랩·신청서식 아님). 없을 때만 rubric 소스→첫 공고 자료 폴백. STEP 2를 순차
    애니메이션 2단으로 재구성: 첫째 중심 자료[파일명·배점·세부 기준·인용(p.N)] → 둘째
    보조 자료[파일명·인용] — "중심 자료(공고문 기준)를 보완하는 세부 근거"로 명시.
  - 검증: esbuild 문법 + vite 빌드 성공.

## 2026-07-25 (이어서 — 종합 리포트 개편: 할루시네이션 3중 방어 + 지적별 인용 1:1 + 화면 전환 UI)

- 한 일:
  - **이름 변경**: "완성 리포트" → "종합 리포트" (사이드바·이동 버튼·배지·주석 전체).
  - **부동소수점 표시 수정**: `+24.700000000000003점` — DeltaPill·추이 차트 증감 라벨·위원 소계
    합산 3곳 모두 소수 1자리 반올림.
  - **배점 열 혼용 결정론 보정(v6)**: 부문별 배점 열이 2개인 표(실증·PoC/우수사례)에서 LLM이
    확산성·효과성만 우수사례 열(25)을 집는 사고 재발 → 공고문 원문에서 "항목명 다음 첫 번째
    숫자"(=첫 부문 열)를 찾아 다르면 강제 교체(`_align_scores_with_notice_first_column`).
    본문 서술에 항목명이 먼저 등장하는 경우·숫자 사이 빈 줄 함정 처리. 실제 파싱 텍스트로
    검증: 확산성 25→10, 안전윤리 15→10, 측정 가능 합 85. + "합은 반드시 100" 프롬프트 규칙이
    혼용의 원인이었음을 확인하고 제거(공고문 배점 그대로, 합 95여도 정상). 추출 LLM도
    temperature=0+seed 고정.
  - **할루시네이션 3중 방어(v5)**: 실측 사고 — 잘못된 URL 스크랩(shutterstock 43자)이 공고문로
    들어가자 LLM이 기본 4범주를 지어내 "추출 성공"으로 기록되고 UI가 "공고문에서 자동 추출"로
    표시. ① 사전 가드(내용 없으면 LLM 호출 안 함) ② 사후 검증(추출 항목명이 공고문 원문에
    실제 등장해야 함 — 없으면 폴백) ③ 정직 라벨(rubric.extracted_from_notice — 폴백이면 점수
    체계표에 빨간 경고 "공고문 기준이 아님, 참고하지 마세요"). 프롬프트에도 [지어내기 금지] 명문화.
  - **지적↔문장 1:1 매핑(issue_refs)**: 지적 텍스트의 "(p.N) '인용'"을 transform에서 결정론
    파싱하고 **제출 문서 원문 존재를 검증**(verified) — 지어낸 인용은 verified=false로 화면에
    안 나감. rubricScore.issue_refs 선택 필드(스키마 v1.1 규칙), comparison 버전 히스토리에도 전달.
  - **종합 리포트 UI 전면 개편**(경이 요청): ① 버전 점 클릭 시 개요(차트)를 대체하는 **별도
    상세 화면**(뒤로가기+버전 전환 칩) ② 점수 체계표 접힘 탭 ③ "다음 수정본 제출" 카드 밖으로
    ④ 탭 순서 AI 피드백→기획→개발 ⑤ 항목 우선순위(심각도·감점률) 정렬+배지 ⑥ 상태 필터
    칩(전체/신규/보완필요/해결) ⑦ 지적 아코디언(기본 접힘) ⑧ 지적별 "왜 이 점수·피드백인가요?"
    4단계 애니메이션: 제출 문서 원문(그 지적이 근거한 문장 강조) → 공고문 기준(파일명)+보조
    자료 근거 → 판정·점수 논리 → 피드백(개발×비전공자는 난이도+구체 해결방안 토글). 인용은
    문서 역할(제출/공고문/보조)별로 분류해 스텝에 배치 — 전부 색인된 실제 원문만.
  - 검증: ai/meeting 테스트 366개 통과 · vite 빌드 성공 · issue_refs 파서/검증 단위 테스트
    (검증 인용 true / 지어낸 인용 false) · 배점 보정 실측(85점).
- 계약 변경(경이 담당 영역, 선택 필드 추가 — v1.1 규칙): review_output.schema.json에
  rubric.extracted_from_notice / rubric.excluded_criteria / rubricScore.issue_refs.
  (excluded_criteria는 스키마에 없어 실 동적 분석에서 저장 검증이 터질 뻔한 잠재 버그도 발견·수정)
- **(추가) 균등 배분 지어내기 차단(v7)**: 실측 — 붙임2 신청 서식만 공고 자료로 올라간 프로젝트에서
  LLM이 서식의 "작성 항목 제목" 4개(문제 정의/AI 도입의 필요성…)를 평가항목으로 뽑고 "배점
  없으면 100점 균등 배분" 규칙에 따라 25점씩 지어냄(합 100, 이름이 원문에 있어 이름 검증도 통과).
  ① 프롬프트에서 균등 배분 규칙 삭제 — 배점 숫자가 명시된 표의 항목만 추출, 신청서 양식
  제목 금지 ② 결정론 검증 — 원문에서 배점 숫자를 확인 못 한 항목이 과반이면 추출 거부(정직
  폴백) + 배점은 항상 원문 첫 부문 열 값으로 강제 ③ 공고문 텍스트 예산 6000→12000자(공고문+
  신청서식 동시 업로드 시 배점표 잘림 방지). 검증: 신청서식 4항목 → 거부, NIA 공고문 → 85점 유지.

## 2026-07-25 (도메인 전환 대응 — 측정 가능 항목만 채점하는 동적 점수 체계 + 점수 체계표)

- 배경: 대상 공모전이 NIA 「2026 공공기관 AI 혁신 챌린지」(실증·PoC 수행보고서)로 바뀜.
  공고문 배점표(목표부합성15/기술성·혁신성30/실현가능성30/확산성·효과성10/안전성·윤리성10
  +가점5)를 근거로 채점하되, ① 안전성·윤리성처럼 정성 판단이 필요한 주관 항목과 ② 공모전마다
  변동이 큰 가점은 자동 채점에서 배제해야 하고, 다른 공모전 배점표가 와도 하드코딩 없이
  동작해야 한다는 요구.
- 한 일:
  - **measurable 분류(동적)**: rubric 추출 프롬프트(meetings.py)가 항목마다
    measurable(true/false)+measurability_reason을 분류. 측정 불가 항목은 채점 rubric에서 빼고
    `excluded_criteria`(배점·사유)로 보존(ai/meeting/graph/rubric.py). **만점 = 측정 가능 항목
    배점 합**(NIA 실증·PoC면 85점 만점). 가점은 기존 bonus_rules 분리 유지(채점 제외).
    `_RUBRIC_EXTRACTION_VERSION=3`으로 기존 캐시 무효화.
  - **근거 인용 강화**: reviewer_prompt에 [지적 작성 규칙 — 문장 단위 인용] 추가 — 지적마다
    문제 문장을 그대로 인용+(p.N) 표기, 문서에 없는 문장 인용 금지(할루시네이션 차단). 리포트
    항목 카드에 "근거 인용 — 채점에 사용된 원문"(RAG evidence quote·페이지·출처) 블록 렌더.
  - **완성 리포트 화면**(VersionTrackerTestPage): ⚠️"제시된 총점은 참고용" 배너 + **점수
    체계표**(항목/배점/채점·제외/기준·사유 — 공고문에서 자동 추출된 그대로) + 히어로 총점
    동적 만점(/85 등) 표시. `/report`가 rubric(criteria·excluded_criteria·bonus)을 내려주도록 추가.
  - 검증: ai/meeting 테스트 366개 전부 통과(Codex calibration 테스트 포함 — #159 코드도 이때
    검증됨). NIA 배점표 시나리오 단위 검증: 채점 4항목 합 85점, 안전성·윤리성 10점 제외+사유,
    가점 5점 별도 확인.
- 다음 할 일: 배포 후 NIA 공고문(HWPX)+수행보고서 실제 업로드 E2E — measurable 분류가 LLM
  추출에서도 기대대로 나오는지(안전성·윤리성 제외) 확인.

## 2026-07-24 (채점 흔들림 해소(seed) + 엄정 채점 2단계 강제)

- 배경: "완성 리포트에서 나쁜 문서 데이터 점수가 너무 후하다"는 지적을 파고드니, 후함이 아니라
  **점수 흔들림**이 진짜 원인이었다. 같은 50점대 테스트 문서를 같은 프로젝트·같은 위원·같은
  rubric으로 돌렸는데 **총점이 31.5 ↔ 64로(±32.5)** 튀었다(data_utilization 5/20 ↔ 15/20).
  temperature=0인데도 gpt-4o-mini가 **경계(애매한) 문서**에서 판정을 flip한다. 후하게 걸린 실행에서는
  위원이 "출처 부족"이라 지적하면서도 acceptable(15/20)을 줘 [엄정 채점] 규칙을 무시했다.
- 한 일:
  - **채점 재현성 — seed 고정**(`meetings.py` llm_call, `_SCORING_SEED=7`): temperature=0에 더해
    OpenAI seed를 고정. best-effort지만 경계 문서 판정 흔들림을 크게 줄인다. 실측: 같은 문서 3회
    **54.5 / 56 / 55.5(±1.5)** 로 안정 — 이전 ±32.5에서 거의 결정론으로.
  - **[엄정 채점]을 2단계 강제 규칙으로**(`ai/meeting/prompts/reviewer_prompt.txt`):
    · (하한) 구체 출처(데이터셋·기관명)·방법·수치가 사실상 전무 → critical_risk, max_score 40% 이하
      (2개 이상 해당·근거 전무면 25% 이하). 특히 데이터 활용 항목은 실제 데이터셋을 이름으로
      명시 안 하고 "다양한 공공데이터 활용" 수준이면 critical_risk.
    · (중간) 방향·요소는 있으나 구체성·정량 근거가 부분적 → needs_improvement, 64% 이하.
    "주제·방향은 맞다"는 이유로 올리지 못하게 못 박음.
  - **검증(변별력 안전점검)**: 같은 rubric에서 —
    · 나쁜 문서(50점대, 출처 없음): 총점 **55.5**, data_utilization 11/20(55%, needs_improvement)
    · 좋은 문서(A유형, 출처 풍부): 총점 **81**, data_utilization 18/20(90%) — 강화 규칙이 좋은 문서를
      깎지 않음(출처를 이름으로 명시하므로 조건 미해당). data 55% vs 90%, 총점 55.5 vs 81로 확실히 갈림.
  - **의사결정**: data를 5/20까지 강제로 내리는 결정론적 페널티는 **하지 않음** — 데이터를 정상적으로
    폭넓게 언급한 좋은 문서까지 깎을 위험이 크고, 이미 55% vs 90%로 변별되기 때문. LLM-판정의 바닥
    효과(주제 관련 있으면 ~50% 밑으로 잘 안 감)는 감수.
  - (참고) 앞서 넣었던 [변별 우선] 규칙은 "A·B 본문이 완전 동일 → 위원 채점 A≈B는 정상"으로 확인돼
    이미 되돌린 상태였고, 이번 강화는 그와 별개로 "근거 전무 문서를 확실히 낮추는" 방향이다.
- 다음 할 일:
  - 65점대 테스트 문서도 목표 구간(adequate 하단)에 드는지 확인
  - 채점 흔들림이 남는지 여러 문서로 추가 관찰(seed가 완벽 결정론은 아님)

## 2026-07-23 (이어서 — 버전 히스토리 누적(v1.0→v1.1→v1.2) + 분량·밀도를 AI 피드백에 추가)

- 한 일:
  - **버전 누적 버그 수정**(구조적): "다음 수정본 제출"을 3번째 하면 v1.2가 안 생기고 v1.1이
    덮어써지며 v1.0 기록이 사라지던 문제. 원인은 백엔드·프론트 **양쪽 모두** 버전을 2개로
    제한한 것 — 백엔드 `GET /comparison`은 최근 2개 meeting만 비교, 프론트
    `buildVersionsFromComparison`은 `[v1.0, v1.1]` 하드코딩.
    · **백엔드**: `ai/meeting/scoring/comparison.py`에 `build_version_history(documents)` 신설 —
      프로젝트의 모든 meeting을 오래된→최신 순으로 v1.0, v1.1, v1.2 …로 누적 변환(항목별
      점수·판정·소속위원·지적 + 직전 대비 new/resolved). `/comparison`이 이 `versions` 배열 +
      최신 pairwise `comparison`을 함께 반환하도록 교체.
    · **프론트**(`VersionTrackerTestPage.jsx`): `buildVersionsFromHistory(versions)`로 히스토리
      전체를 렌더. 점수 추이 그래프에 점이 제출한 만큼 누적(v1.0·v1.1·v1.2…).
    · 검증: 기존 프로젝트 `/comparison`이 v1.0(82)·v1.1(77)·v1.2(21.5) 3버전 반환 확인.
  - **"A유형 ≈ B유형" 재해석 → 위원 프롬프트 변별 강화 되돌림**: 사용자 확인 결과 A·B는 겹치는
    본문이 **토시 하나 안 틀리고 동일**(내용은 같고, A는 페이지에 여백이 많고 B는 꽉 참). 즉
    **위원 채점(내용)에서 A≈B는 정상**이고, "B가 A보다 낫다"는 건 **분량·밀도**(문서 피드백) 축의
    차이였음. 전날 넣었던 `reviewer_prompt.txt` [변별 우선] 규칙은 잘못된 전제라 **삭제**(검증된
    캘리브레이션 밴드만 유지).
  - **분량·밀도를 완성 리포트 "AI 피드백" 탭에 추가**(`VersionTrackerTestPage.jsx`): 기존 AI 피드백은
    오탈자·맥락 2축만 가져왔는데, A/B를 가르는 **분량(요구 페이지 수 충족)·밀도(페이지 채움률·빈
    페이지)** 축(`getFormatCheck`, 재인 워크벤치)을 추가해 탭 상단 "분량·밀도" 요약 카드로 표시.
    같은 내용이라도 여백 많은 문서(A=밀도 부족)와 꽉 찬 문서(B=밀도 양호)가 여기서 갈림(점수 미반영).
  - **히어로 문구 교체**: "기획·개발 위원=평가기준·배점 근거 채점 / AI 피드백=오탈자·분량·밀도·맥락 /
    ⚠️위원 총점이 높아도 AI 피드백 미반영 시 서류 심사 불통과"를 2줄 + 경고 콜아웃으로 한눈에 정리.
- 막힌 점 / 후속:
  - **분량·밀도는 PDF 입력이어야 작동**: DOCX는 페이지/밀도 계산에 LibreOffice(DOCX→PDF)가 필요한데
    이 환경은 LibreOffice 다운(`HWP_CONVERTER_UNAVAILABLE`) → DOCX는 `None`. PDF는 변환 없이
    PyMuPDF로 직접 분석되고 실측 검증됨(예: 67% 양호 vs 40% 부족). 여백은 레이아웃 속성이라 PDF가
    더 정확한 입력이므로, A/B는 **PDF로 저장해 업로드**하면 밀도가 갈림. (위원 채점은 DOCX/PDF 무관)
  - 이 공고문은 요구 페이지 수 추출이 `None`(분량 기준 없음) → 분량 카드는 "기준 없음", 밀도가 변별 축.

## 2026-07-23 (이어서 — 채점 캘리브레이션(점수 밴드)로 변별력 확보)

- 한 일:
  - **reviewer 프롬프트에 점수 구간 기준 추가**(`ai/meeting/prompts/reviewer_prompt.txt` [점수 제안 규칙]):
    judgment→배점 대비 비율 밴드(critical_risk 0~40% / needs_improvement 40~64% / adequate 65~84% /
    strong 85~100%) + 엄정 채점 규칙(근거 없음·"~할 예정" 추상 서술·수치 없는 일반론이면 adequate
    이상 금지). 기존엔 밴드가 없어 LLM이 무난하게 중상위권만 줘서 나쁜 문서도 ~80점(변별력 부족).
  - **검증(실측)**: 같은 공고문·rubric 기준으로 —
    · 잘 쓴 문서(A유형 v1.0): 82 → 77 (적정)
    · 나쁜 문서(추상적·근거 없음): **21.5** (needs_improvement 5 + critical_risk 3)
    → temperature=0(결정론) + 캘리브레이션으로 "품질=점수" 변별력 확보. "나쁜 문서인데 82점" 해소.
- 다음 할 일:
  - v1.0→v1.1 재분석으로 상향선 재현(이제 품질 반영이 제대로 됨)
  - 캘리브레이션이 위원장 종합/최종 계산과도 일관되는지 추가 관찰

## 2026-07-23 (이어서 — 채점 결정론화(temperature=0) + 채점 과정 트레이스 로그)

- 한 일:
  - **채점 LLM temperature=0**(meetings.py `_build_real_llm_call`): 기본값 1.0에서는 같은 문서를
    채점해도 매번 ±수 점씩 흔들려(노이즈가 품질을 가림) "더 좋은/긴 문서가 더 낮게" 나오거나
    개선본(v1.1)이 하향되는 문제가 실측됨(원본 81 vs A유형 82 역전, v1.1 하향). 채점·위원장 종합은
    재현성이 중요한 판단 작업이라 0으로 고정 → 같은 문서 재채점 시 동일 점수(결정론).
  - **채점 과정 트레이스 로그**(팀장 제출/검증용, `_SCORING_TRACE_FILE`): 각 위원 LLM 호출의
    프롬프트(위원에게 준 공고문 평가기준·배점 + 제출 문서 + RAG 근거 + 심사 지시)와 응답(항목별
    점수·판정·강점/지적/제안) 원본을 파일에 append. 터미널 `tail -f`로 실시간 관찰 가능. 위원이
    실제로 "주어진 rubric·문서·근거에 기반해 채점"(User RAG)하는지 증거로 남긴다.
  - (레포 외) 트레이스 → 팀장 제출용 발췌 text 생성 스크립트 별도 작성(Mongo 접속정보 포함이라
    커밋 안 함): 공통 심사 지시 발췌 + 배점 + 위원별 점수·근거 인용 + 위원장 + 최종점수. 파일명에
    생성 시각(분 단위) 포함.
- 막힌 점 / 후속:
  - **채점 관대함**: 캘리브레이션(점수 밴드)이 없어 나쁜 문서도 ~80%에 몰림(변별력 부족) — reviewer
    프롬프트에 "배점 구간별 기준(근거 없으면 후하게 주지 말 것)"을 넣는 것을 다음으로 제안.
  - temperature=0 반영했으니 v1.0→v1.1 재분석으로 상향선 재현 필요.

## 2026-07-23 (완성 리포트 실데이터 배선 + 동적 rubric 수정 + 버전 비교(C) + AI 피드백 탭)

- 한 일:
  - **A. 동적 rubric 추출 버그 수정**(가은 위임, `backend/app/api/routes/meetings.py`
    `_build_rubric_extraction_prompt`): LLM이 `criterion_id`에 `persona_id`를 복사해 여러 항목이
    같은 id를 갖고(중복) → `build_dynamic_rubric_mapping`이 `ValueError`로 거부 → 정적 4항목으로
    폴백되던 문제. 프롬프트를 "criterion_id는 항목마다 고유(persona 식별자 아님) + 배점 0 항목
    제외"로 고쳐 **공고문 실제 5항목·배점(AI혁신성25/데이터활용성20/실현가능성20/창의성차별성15/
    기대효과성20=100)** 이 그대로 채점에 반영됨. 실 파이프라인 재분석으로 검증
  - **B. 완성 리포트를 실제 `/report` 데이터로 배선**(`VersionTrackerTestPage.jsx`): mock
    `ALL_VERSIONS` 대신 `reportToVersions(report)`로 실 점수·위원 피드백·개인화 impl_guides 렌더.
    항목별 max_score(동적 rubric 대비)·위원 탭 매핑 실데이터화. embedded일 때만 실데이터, 단독
    `/version-test`는 mock 데모 보존
  - **위원 배분 fix**: 종합 위원(presentation_completeness)이 전 항목을 겹쳐 채점해 "첫 채점자
    우선"이면 개발 항목이 기획으로 잘못 감 → **채점자 우선순위(기술>전문>종합)로 담당 결정**
  - **C. 버전 비교(RPT-004)**: `GET /projects/{id}/comparison` 엔드포인트 신설(최근 2 meeting을
    `build_revision_comparison`으로 비교) + 프론트 "다음 수정본 제출"을 **실제 파일 업로드→기존
    target 삭제→재분석→진행률**로 구현 + `/comparison`으로 **[v1.0,v1.1] 2버전 렌더**(이전/현재
    막대·해결/신규/잔존 뱃지·점수 추이). 개인화 가이드는 최신 버전에서만 표시
  - **자세히 보기 파서 수정**: 실 LLM 산문이 `1. 2. 3.` 형식인데 파서가 `①②③`만 인식→밋밋한
    문단 폴백. 두 형식 다 인식(문장 중 숫자 오탐 방지)해 단계 카드+애니메이션 복원
  - **AI 피드백 탭 신설**(기획/개발 위원 옆 3번째): 문자서식·오탈자를 위원 채점과 별개로 표시
    (`getTypoCheck`/`getContextCheck`), **점수 미반영**, "수정 필요"로 추적. LLM 호출이라 메인
    리포트와 분리(비차단) 로딩
  - **환경 구성**: HWP 대비 LibreOffice 26.2 + H2Orestart 0.7.13(unopkg add) 설치, `olefile`
    누락(HWP 바이너리 파서 필수, requirements.txt엔 있으나 로컬 env 미설치) 설치. 단, 실제 테스트
    파일은 DOCX라 텍스트 추출은 LibreOffice 불필요(HWPParser/DOCXParser 직접 파싱)임을 확인
- 결정/이유:
  - **위원 탭(2개)은 4-persona를 접은 추상화** — 개발=technical_feasibility, 나머지=기획. 동적
    rubric에서 여러 위원이 겹쳐 채점하므로 "첫 채점자"가 아니라 우선순위로 담당을 정해야 정확
  - **AI 피드백은 점수와 분리** — 오탈자·서식은 배점 대상이 아니라 교정 항목이라 별도 탭 +
    "수정 필요"만 추적(사용자 요청)
  - **수정본은 기존 target 삭제 후 업로드** — analyze가 target "첫 문서"를 쓰므로. 이전 버전
    데이터는 meeting 스냅샷에 보존돼 `/comparison`이 그대로 비교 가능(문서 삭제 무영향)
  - **동적 rubric 캐시**(project.dynamic_rubric_mapping)로 재분석 시 같은 rubric 재사용 →
    v1.0/v1.1이 동일 기준으로 비교됨
- 막힌 점:
  - H2Orestart HWPX 변환이 `0xC0000409` 크래시(LibreOffice Java 설정) — 그러나 텍스트 추출 경로는
    LibreOffice를 안 쓰는 걸 확인해 우회(변환은 재인 담당 미리보기 영역). 테스트 파일도 DOCX라 무관
  - 개선 수정본이 원본보다 짧아(10,246→5,728자) 재채점에서 점수가 오히려 하락 — 비교 메커니즘
    (해결/신규/잔존·델타)은 정확히 동작. LLM 채점 노이즈 + "교체"라 손해. additive 수정본이 상승에 유리
  - AI 피드백 버전 간 "해결" 뱃지(v0→v1 오탈자 diff)는 버전별 검사 결과 저장이 필요 → 다음 단계
- 다음 할 일:
  - A(rubric)·B·C·AI 피드백 탭 브라우저 E2E 최종 확인
  - AI 피드백 버전 diff(해결 추적) — 검사 결과 per-version 저장
  - 개발 위원 개인화 뱃지 실데이터 확인(개발 항목이 needs_improvement로 낮게 나오는 문서 필요)

## 2026-07-21 (이어서 — 개인 맞춤형 피드백 루프 백엔드 로직·계약 정합)

- 한 일:
  - **개발 위원 피드백 개인화 로직** `ai/meeting/scoring/personalization.py` 신설:
    `classify_impl_difficulty(profile)`(전공/학위/경력/GitHub 신호로 구현 난이도 hard/moderate/easy
    **결정론 판정** — LLM 아님) + `build_impl_guide`/`attach_impl_guides`(지적별 난이도+가이드,
    산문은 주입 `llm_call`로 생성, 없으면 판정만). **회의 파이프라인 무손상 후처리(B안)** — 프로필
    없으면 자연 폴백. 프론트 `IMPL_GUIDE` mock을 대체할 실로직
  - `is_technical_persona()` 헬퍼 — 회의엔 `committee:'dev'` 플래그가 없어(도메인별 4인 persona)
    개발 위원을 persona_id 화이트리스트(`technical_feasibility`/`dev_expert`)로 고정.
    윤한 리포트 훅이 이걸 그대로 import해 개발 위원 지적만 골라 `attach_impl_guides`에 넘김
  - version-test 프론트를 백엔드 `attach_impl_guides` 출력(`{level,verbosity,label,prose}`)에 정렬 —
    `personalizeGuide()` 하나가 E2E 교체 지점(mock→fetch)
  - **프로필 계약을 마이페이지/GitHub API 실제 필드에 정합**(무음실패 방지, 아래 막힌 점 참고):
    `github{public_repos,followers,total_stars,primary_languages}` / `degree` 영문 enum /
    `experience{internship_months,competition_count,award_count}`. fixture `0.2.0-draft`
  - 윤한 3종(프로필 CRUD `GET/PUT /users/me/profile` · 비교 API `GET /comparison` · 리포트
    개인화 훅 `impl_guides`) dev 반영 확인 — 전부 내 `is_technical_persona`+`attach_impl_guides`
    계약 그대로 배선됨. 테스트 154개 통과
- 결정/이유:
  - **개인화도 점수엔진과 같은 철학**: 난이도 "판정"은 결정론(재현 가능), "산문"만 LLM(DI로 분리).
    프로필 미제출/일부 누락은 없는 키 0 처리 → hard 폴백(안전)
  - **GitHub 신호는 공개 API(api.github.com)로 얻는 값만** 사용 — 커밋 총수·백엔드 이력 같은
    조회 불가/파생 필드는 계약에서 배제해 현실에 맞춤. `degree`는 영문 enum으로 통일(한글은 표시용)
  - 프로필 저장은 윤한이 이미 한 `users.profile` 임베드 방식 유지(별도 컬렉션 분리 강제 안 함 —
    rework 회피, DB는 윤한 담당)
- 막힌 점:
  - **개인화 무음실패 직전 발견**: 내 초기 분류기가 GitHub 공개 API로 조회 불가한 필드
    (`has_backend_experience`/`relevant_projects`/`total_commits`)를 읽어, E2E로 붙였으면 GitHub
    신호가 항상 0 → 전원 "어려움"으로 무음 오작동할 뻔. 실제 조회 가능 필드로 재설계해 해결
  - squash-merge로 커밋 SHA가 바뀌어(내 로컬 브랜치는 ahead지만 content는 dev 반영) 반영 여부가
    헷갈림 — SHA 아닌 파일 content 기준으로 확인하는 습관 필요
  - 가은 MyPage 프로필 폼이 아직 dev 미반영(다른 브랜치/PR 대기 추정) — dev엔 여전히 shell
- 다음 할 일:
  - 가은: MyPage 프로필 폼 dev 반영(**계약 필드 준수** — `degree` 영문 enum, `github` 4필드) +
    `/board` 리포트에서 `impl_guides`/`comparison` 렌더
  - 프론트 실API 배선: version-test mock → `GET /report`(impl_guides) · `GET /comparison` · `GET /meetings`
  - E2E QA 패스: 업로드→분석→리포트 개인화→수정 재제출→비교 상승세 확인

## 2026-07-21

- 한 일:
  - **버전 추적형 User RAG = 개인 맞춤형 피드백 루프** 프론트 실험 화면(`/version-test`) 신설
    (`frontend/src/pages/VersionTrackerTestPage.jsx` + `VersionTrackerTest.css`). 내 백엔드
    산출물을 화면으로 검증하는 용도 — 각 버전 위원 피드백은 `review_output.reviewer_results`,
    버전 간 증감·해결/잔존/신규는 **내 RPT-004 `build_revision_comparison()`** 출력 구조를
    그대로 mock으로 넣음(백엔드 연동 시 mock만 실데이터로 교체)
  - 기능: v1.0→v1.3 **버전 누적 제출**(하나씩 reveal) + **위원 탭 분리**(기획/개발) +
    **이전 vs 현재 막대 비교** + **점수 추이 라인차트**(SVG) + 카운트업/막대/라인 애니메이션
  - **개인화 입력**: 수정본(기본) + GitHub + 이력·교육수준 제출, **TEST 프로필 토글**(비전공자/전공자).
    개발 위원 피드백을 프로필에 따라 다르게 — 비전공자=`구현 난이도 어려울 수 있음`+`자세히 보기`
    (단계별 상세), 전공자=`쉬움`+간결한 한 줄
  - dev 최신화(가은 서비스 방향전환 리디자인 `/board` 프로토타입 + 용준 ideation 병합) 반영,
    `feat(frontend) TEST 섹션` 커밋(`f2f3694`) 푸시
- 결정/이유:
  - **격리 원칙 유지**: `/board`·StepSidebar 등 가은 프론트 코드 미수정, `App.jsx` 라우트 1줄만
    추가. 실험 검증 후 정식 플로우(`/board` 프로젝트 리포트) 이어붙일 때 가은과 배치 협의 예정
  - **디자인 톤을 가은 새 `.rb-root`(웜 화이트/글래스, 퍼플·코랄·그린·앰버, mono, lucide)에 1:1**
    맞춤 — 나중에 이어붙일 때 이질감 0
  - **RPT-004 재활용이 핵심 메시지**: "1회성 챗봇과 달리 수정 이력을 기억해 점수 상승세·지적 해결을
    추적"을 시각적으로 보여줘, 내 비교 로직이 제품 차별점으로 직결됨을 시연
- 막힌 점:
  - 새 디자인이 `lucide-react`를 쓰는데 node_modules 미설치 상태여서 `/board`도 안 뜸 →
    `npm install`로 해결. `Github` 아이콘은 lucide 1.x에서 브랜드 아이콘 삭제로 없어서 `GitBranch`로 교체
  - 막대가 점수 비율과 무관하게 꽉 차 보이던 버그 — 채워지는 div에 `height`가 없어 0px(빈 트랙만
    보임)이었음. `height:100%` 지정으로 수정
- 다음 할 일:
  - 실제 연동: 프로젝트당 회의(버전) 다건 저장·목록 조회 API(윤한) → mock을 실데이터로 교체
  - 개인화 입력(GitHub/이력) 실제 파싱·프로필화는 별도 논의(현재는 프론트 mock 프로필 2종)
  - 가은 새 디자인 정식 플로우에 버전 히스토리 탭으로 이어붙이기 협의

## 2026-07-16

- 한 일:
  - **RAG-003/004/005 회의 연동**(용준 어댑터): `run_meeting`/state/build에 `evidence_context`(persona·criterion별 근거+사전 sufficiency) + `evidence_callback`(backend 주입) optional 추가. reviewer 노드가 ①사전 prompt_guard 삽입 → ②의견 생성 → ③criterion별 콜백(RAG-004 링크+RAG-005 최종판정) → ④A안(근거를 RAG-004로 교체) + `allow_numeric_score=False` 게이팅. EvidencePool에 `(document_id, chunk_id)→evidence_id` 역조회(`register_linked`) 추가. `<<EVIDENCE_GUARD>>` 토큰 신설. **전부 backward-compatible**(인자 없으면 기존 flat 경로 그대로)
  - 용준 실제 어댑터 출력 샘플(`rag_adapter_samples.json`)로 통합 테스트 — `MeetingLinkedEvidenceRef`가 v2 evidence로 정확 매핑(text 없는 건 retrieved에서 보강) + run_meeting E2E 검증
  - **review_output v2.1.0 계약 개정**(팀 동의 후): 선택 필드 `similar_success_cases`(RAG-006 유사사례, reference_only) 추가, `schema_version` const→enum `["2.0.0","2.1.0"]`(하위호환). `run_meeting`/`assemble_document` pass-through, 신규 문서 "2.1.0"
  - **TST-002 위원 일관성 하네스**(`ai/meeting/quality/consistency.py`): 반복 평가 편차(총점/항목 점수/judgment 일치율/핵심 지적 Jaccard) 측정 + 허용범위(`ConsistencyTolerance`) 위반 판정. v2 문서만 읽어 실행 방식 비의존(DI)
  - 테스트 40개 통과(scoring 5 + explanation 4 + comparison 3 + graph 9 + rerun 2 + reevaluate 2 + evidence_integration 8 + consistency 7). **내 담당 요구사항(MTG-001~004/006/007, RPT-004/006, TST-002) 코드 전부 완료**
- 결정/이유:
  - **회의 파이프라인 ↔ RAG decoupling 유지**: 그래프가 `ai.rag`를 직접 import하지 않고, backend가 판정 결과·근거를 plain data(evidence_context) + Callable(evidence_callback)로 주입. RAG 스키마가 바뀌어도 회의 코드가 안 깨짐
  - **RAG-004 A안(위원 자기보고 근거 폐기, RAG-004 링크만 사용)** + 게이팅은 `(persona, criterion)` 단위 — 용준과 계약 확정
  - **similar_success_cases는 permissive(내부 재검증 안 함) + 최상위 + schema_version enum**: RAG-006이 진행 중이라 계약을 용준 스키마에 안 묶고, 기존 "2.0.0" 문서도 유효하게 유지
  - **일관성은 '완전 일치'가 아니라 '허용 편차 정의+측정'**(생성 모델 특성상 완전 동일 요구 금지). 실제 모델 붙기 전엔 stub으로 파이프라인 결정론(편차 0) baseline 확인
  - 공용 계약 변경(v2.1.0)은 팀 절차대로: 제안서(`review_output.v2.1.proposal.md`) → 재인/윤한/가은/용준 동의 → 적용
- 막힌 점:
  - 가은이 실제 OpenAI 호출로 `assemble_document`의 persona_id 버그를 잡아줌(LLM이 지어낸 persona_id를 못 믿어 딕셔너리 키로 덮어쓰게 수정) — 내 stub 테스트로는 안 잡혔던 것. 회귀 가드 테스트 추가 예정
  - 용준 어댑터 출력은 **persona별 flat**인데 내 `evidence_context`는 `(persona,criterion)별+sufficiency` 묶음이라, 그 사이 조립 헬퍼(`build_evidence_context`)가 필요 → RAG-005 사전 sufficiency granularity 확인 후 추가 예정
  - dev가 하루에 #46~#57까지 빠르게 머지돼(로깅·HWP·RAG-006·backend 실연결 등) push 전마다 재싱크 반복
- 다음 할 일:
  - `build_evidence_context` 헬퍼(용준 회신 대기) / 가은 모델명 확정 후 실제 LLM E2E / 실제 모델로 일관성 편차 실측
  - `assemble_document` persona_id 회귀 가드 테스트
  - RPT-004/006 화면(가은)·evidence_context 조립·API(윤한) 연동 지원

## 2026-07-15

- 한 일:
  - **rubric_mapping_government_support 4인 확장**: 회의 진행 위원회를 government_support.json과 동일한 4인(policy_fit·business_strategy·technical_feasibility·budget_execution)으로 맞추고, policy_alignment→policy_fit·execution_plan→budget_execution 재배정, `required:true`(source:notice) 반영 (#27)
  - **M3 LangGraph State**: 회의 1회 공유 상태(TypedDict) 정의. reviewer_results/evidence에 병렬 fan-in 병합 리듀서를 둬 위원 결과 유실 방지 (#30)
  - **M4 노드+그래프 조립**: reviewer(위원별 독립 병렬) → score(M2 계산 엔진 연결) → chair(종합+top_revisions) 노드, rubric_mapping→v2 rubric 변환기, 위원 raw(judgment 6종)→v2 reviewerResult(4종) 변환기, EvidencePool. LLM은 인터페이스만 분리하고 stub으로 테스트 (머지됨)
  - **실제 LLM 연동 + 엔트리포인트**: `make_openai_llm_call`(모델명 필수 인자로 강제), `run_meeting`(rubric_mapping+문서→v2 결과 조립). backend가 analyze() 내부만 교체하면 되게 시그니처 맞춤
  - **MTG-006 완성**: `run_meeting(on_progress=...)` 진행률 통지 + `assemble_meeting_graph(checkpointer=...)`로 실패 노드부터 재시도
  - **RPT-006**: `build_score_explanation` 점수 설명 카드 로직(계산값에서만 설명 생성)
  - **MTG-007**: `rerun_reviewer` 특정 위원만 재평가+재종합, 나머지 위원 결과 유지
  - **RPT-004**: `build_revision_comparison` 수정 전후 비교(항목별 증감·해결/신규 지적, 평가기준 변경 시 직접 비교 제한)
  - `contracts/mocks/final_meeting_result.v2.json` 추가(가은 프론트 스텁 API용, v2 검증 통과). 기존 `final_meeting_resault.json`은 v2 이전(17개 위반)이라 쓰지 말 것으로 가은에게 전달
  - 테스트 총 23개(scoring 5 + explanation 4 + comparison 3 + graph 9 + rerun 2) 통과
- 결정/이유:
  - **required 필드 v2 표준은 경이가 확정**(가은 위임): source:notice 항목은 전부 required:true, default_supplementary_perspectives는 채점 제외. 관련 파트(윤한·용준·가은)에 공유
  - **government_support 회의 과정=4인 / 영상 MVP=2인**: 4인으로 평가·종합하되 media_script는 2인분만(테스트 후 4인 확장 예정)
  - **점수/설명/비교 리포트는 LLM이 아니라 계산값에서만 생성**: RPT-006 예외("LLM 자연어와 계산값 불일치 방지")를 구조적으로 차단 — 카드의 어떤 수치도 M2 출력과 어긋날 수 없음
  - **위원 raw 출력 스키마는 가은 초안 그대로 유지, v2 변환은 경이 노드가 전담**(담당 경계). judgment 6종 중 insufficient_evidence/not_applicable은 rubric_scores에서 제외 → M2 누락 감점 로직이 자연 처리
  - **chair_prompt.txt의 final_priority_actions에 title/target/reason/evidence_ids 보강**: 기존 스키마로는 MTG-004 검수 기준("이유·대상 문단 제시")을 못 지켜서 실행 프롬프트 자체를 수정
- 막힌 점:
  - LangGraph 노드 이름에 `:` 예약문자 불가 → `reviewer__{persona_id}`로 변경
  - openai 패키지 미설치 → requirements.txt에 추가. 실제 사용 모델은 가은이 비용/품질 검토 중이라 `make_openai_llm_call` model을 기본값 없는 필수 인자로 둠
- 다음 할 일:
  - **TST-002 위원 일관성 테스트**(내 담당 마지막 요구사항): 반복 실행 편차 측정 하네스. 실제 편차는 LLM 붙어야 의미 있으므로 stub으로 파이프라인 결정론 baseline부터
  - 윤한: 진행률/재시도/재실행 API 연결, RPT-004 두 회의 조회, meetings.py 스텁을 run_meeting 호출로 교체
  - 가은: RPT-006 점수 설명 카드·RPT-004 비교 리포트 React 화면
  - RPT-004/006은 로직 완료, 화면·DB 연결만 남음

## 2026-07-14

- 한 일:
  - CLAUDE.md / `docs/*.md` 정독 후 경이 담당(LangGraph·점수 엔진·평가 결과 구조) 마일스톤 스케줄 수립 (M0 환경 → M1 Mock → M2 점수 엔진 …)
  - `review-board` conda env 생성 (Python 3.12 + langgraph/jsonschema/pytest), import 검증 완료
  - **M1**: `ai/meeting/tests/fixtures/`에 회의 결과 Mock(reviewer_result / final_meeting_result / rag_response) 작성 + jsonschema 검증 통과 → PR #13으로 dev 머지
  - **review_output v2.0.0 계약 개정 제안**: 가은 `sample_review_result` 설계를 통합(rubric 배점표 / judgment / cross_reviews / 풍부한 chair_summary / criterion_owner 채점), `media_script` 유지(재인), MTG-003 필수항목 누락감점(penalties) 반영 → #13에 포함, 팀(가은·윤한·재인) 승인
  - 가은 `docs/prompts` 초안 → `ai/meeting/prompts/` 실행 파일 3종 변환: `reviewer_prompt.txt`, `chair_prompt.txt`, `prompt_loader.py` (+ `__init__.py`), 스모크 테스트 통과
  - **M2 점수 엔진** `ai/meeting/scoring/`: `calculator.py`(criterion_owner·Decimal 결정론), `weights.py`, `deductions.py`(누락감점) + `tests/test_scoring.py` **pytest 5개 통과**(mock 재현 총점 61·동일입력=동일출력·누락감점)
- 결정/이유:
  - 점수 모델은 **criterion_owner**(배점=가중치) 채택 — 실제 심사 방식에 가깝고 MTG-003 "가중합" 요건 충족. 점수는 LLM이 아니라 Python 규칙으로만 계산해 재현성 보장
  - reviewer/chair 실행 프롬프트는 **가은 초안 출력 스키마 그대로 유지**하고, v2 계약으로의 변환은 경이 LangGraph 노드(M4)에서 처리하기로 함 — 가은 설계 보존 + 담당 경계 유지(위원 원본→v2 매핑은 경이 코드 몫)
  - 페르소나 프롬프트 prose는 중복 저장하지 않고 `persona_cards.json`에서 렌더링 → 카드만 고치면 프롬프트 자동 반영(drift 방지)
  - 공통 계약(`review_output.schema.json` v1→v2 교체)은 프롬프트·M2와 **분리해 별도 PR**로 — 계약 변경은 재인·윤한·가은 리뷰가 필요하기 때문
- 막힌 점:
  - dev가 반나절에 #13~#20까지 빠르게 머지돼 push 직전마다 재싱크 필요 — fetch → `git merge origin/dev` 반복으로 대응(내 담당 영역과 충돌은 없었음)
  - `conda run -n review-board`가 한글 stdout에서 cp949 인코딩 에러 → env python 직접 호출 + `PYTHONUTF8=1`로 우회
  - JSON은 주석 불가 + 스키마 `additionalProperties:false`라 Mock 파일 헤더를 `_meta` 블록/README로 대체
- 다음 할 일:
  - v1→v2 스키마 교체 + 드래프트 스키마 제거 + `contracts/mocks` 승격(가은 페르소나/rubric 반영)을 계약 PR로 올리기
  - M3~M5: LangGraph State·노드(reviewer_a/b·score·chair·media_script)·그래프 조립
  - 위원 원본(raw) → v2 `reviewerResult` 매핑 노드 구현
  - TST-002 위원 일관성 테스트 본격화
