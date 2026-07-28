# [요청] 종합 리포트 채점 위원을 기획·개발 위원 2명 체계로 — 선행 조건 2건

> **[완결, 2026-07-28]** 요청 2건 모두 용준 PR #188로 dev 머지 완료
> (domain_tags에 competition 추가 + planning_expert→planning / dev_expert→technology).
> 이를 받아 경이 후속 PR에서 rubric_mapping_competition.json committee를
> `[planning_expert, dev_expert]`로 교체하고 추출 캐시 버전을 v9로 올렸다.
> 아래 본문은 요청 당시 기록.

> 작성: 경이 → 용준 / 2026-07-27
> 배경 결정(경이): 종합 리포트(심사형 회의)의 실제 채점 주체를 구 4위원(창의·기술·
> 사업전략·완성도)에서 **아이디어 회의의 기획 전문가(planning_expert)·개발 전문가
> (dev_expert) 2명**으로 통일한다. 화면은 이미 기획/개발 위원 2그룹으로 보이는데 내부만
> 4위원이라, 사용자가 만나는 인물과 실제 채점 주체가 어긋나 있었다(아이디어 회의 →
> 종합 리포트로 이어지는 "같은 전문가에게 계속 검토받는" 서사 복원 목적).

## 경이 쪽에서 이미 준비된 것 (이번 PR)

- **배정된 위원만 집계**: 항목마다 배정(주 담당 1 + 보조 0~1)된 위원의 점수·지적만
  채점에 반영 (ai/meeting/graph/transform.py — 이전엔 비담당 위원 점수도 평균에 섞여
  항목당 3명 평균이 발생). 비담당 항목은 RAG-004/005 콜백 호출도 건너뛰어 검색 비용 절약.
- **추출 배정 규칙 강화**: rubric 추출 시 "항목 세부 평가내용과 실제로 관련 있는 위원만
  1~2명 배정, 무관 위원 금지" (meetings.py 프롬프트). 캐시 버전 v8로 재추출 유도.
- **RAG 인터페이스 불변**: 항목당 1~2명은 기존 primary/secondary_persona_id 스키마
  그대로라 `iter_persona_criteria` 등 ai/rag 쪽 수정이 **필요 없습니다** (보조 위원
  리스트(3명) 확장안은 RAG 변경이 필요해 철회했습니다).

## 요청 2건 (용준님 영역 — 이게 반영돼야 위원회 교체 가능)

1. **persona_cards.json** — `planning_expert` / `dev_expert`의 `domain_tags`에
   `"competition"` 추가. 지금은 `["ideation"]`뿐이라 심사형 회의에서 카드 로드가
   막힙니다(prompt_loader가 domain_tags로 필터). 두 카드가 아이디어 회의용 어조로
   작성돼 있어서, 심사형(채점) 관점 문구 보강까지 해주시면 더 좋습니다 — 범위는
   용준님 판단에 맡깁니다.
2. **role_mapping.py** — `_DOMAIN_PERSONA_ROLE_MAPPING["competition"]`에 두 위원 추가.
   가안: `planning_expert → planning`, `dev_expert → technology` (역할 정의를 아시는
   용준님이 최종 확정해 주세요). 매핑이 없으면 `PersonaRoleMappingError`로 분석이
   500이 나서, 이 두 줄이 위원회 교체의 하드 선행 조건입니다.

## 반영 후 순서

1. 용준님 위 2건 머지 → 2. 경이가 rubric_mapping_competition.json의 committee를
   `["planning_expert", "dev_expert"]`로 교체 + 기본 4축 배정 재작성(후속 PR, 준비돼 있음)
   → 3. 배포 후 새 프로젝트로 실측 1회(카드 어조가 심사형에서 자연스러운지 포함).

프론트(기획/개발 그룹핑)는 이미 dev_expert 기준으로 준비돼 있어 추가 작업이 없습니다.
