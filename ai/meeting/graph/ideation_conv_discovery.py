# 작성자: 용준/Claude(2026-07-21) / pge/Claude(2026-07-27, 주제 브레인스토밍 재설계 — 키워드
#         선택 방식)
# 목적: 대화형 아이디어 발전 회의(ideation-conversation)의 discovery(아이디어 발굴) 모드
#       전용 LangGraph 노드.
#
#       재설계 흐름: 트렌드+공모전 근거로 키워드를 추천하고(keyword_recommendation), 사용자가
#       관심 키워드를 다중 선택하면(keyword_selection, LLM 호출 없이 결정적으로 파싱 또는
#       재추천 판단) 그 키워드 조합으로 가벼운 주제 목록을 만든다(topic_generation). 이후
#       사용자의 선택/결합/재추천/전문가추천 요청 처리(candidate_selection)와 최종적으로
#       refinement의 첫 phase("planning_question")로 합류시키는 부분은 옛 구조(기획 위원이
#       완성된 후보 5개를 곧바로 만들던 candidate_planning/candidate_feasibility) 그대로다 —
#       idea_candidates/selected_idea 등 필드명을 그대로 재사용해 가벼운 "주제"를 담도록
#       바꿨을 뿐이다. 실현 가능성 검토(옛 candidate_feasibility)는 후보 전체가 아니라 선택
#       확정된 주제 1개에만 수행하도록 candidate_selection 쪽으로 옮겼다(_apply_feasibility_review
#       참고) — 선택되지 않은 나머지에 LLM 비용을 쓰지 않는다.
#
#       refinement 전용 노드(ideation_conv_nodes.py)는 이 파일이 건드리지 않고 그대로
#       재사용한다(_build_message/_safe_call_structured_json/_blank/_last_user_answer).
# import: prompts.build_ideation_conv_*(형제 패키지), 같은 패키지의
#         ideation_conv_state/ideation_conv_nodes/llm.

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Callable

from prompts import (
    build_ideation_conv_candidate_feasibility_prompt,
    build_ideation_conv_candidate_selection_prompt,
    build_ideation_conv_keyword_recommendation_prompt,
    build_ideation_conv_topic_generation_prompt,
)

from .ideation_conv_nodes import (
    _blank,
    _bullets,
    _build_message,
    _last_user_answer,
    _safe_call_structured_json,
)
from .ideation_conv_state import IdeationConvState, build_roundtable_opening_message
from .ideation_nodes import EvidenceLookup, call_evidence_lookup
from .ideation_trace import sanitize_preview, trace_event
from .llm import LLMCall


def _runtime_scope_for(state: IdeationConvState) -> dict[str, Any]:
    """ideation_conv_nodes.py::_runtime_scope_for와 동일한 목적 — 이 파일의 두 검색 호출
    시점(candidate_planning/candidate_feasibility)은 아직 후보를 선택하기 전이라
    selected_idea_document_id는 항상 None이지만, session_id는 여기서도 최신 state 기준으로
    넘겨야 사용자 답변 target(user_session_answer) 스코프가 일관되게 적용된다."""
    return {
        "session_id": state.get("session_id"),
        "selected_candidate_document_id": state.get("selected_idea_document_id"),
    }

# 용준/Claude(2026-07-22, 요청: 선택된 아이디어를 target 문서로 생성) — ai/meeting/graph는
# ai.rag를 직접 import하지 않는다(기존 evidence_lookup/ground_claims와 동일한 경계). 실제
# 색인 구현(ai.rag.orchestration.ideation_target_indexing_service)은 backend가 만들어
# 주입한다. kind="candidate"면 payload={"session_id","candidate_id","candidate"}, 반환값은
# {"document_id": str} 이상(색인 실패 시 호출부가 예외를 던지거나 status="failed"를 반환할
# 수 있다 — 이 노드는 실패해도 selected_idea_document_id=None으로 안전하게 진행한다).
IndexTargetEvidenceFn = Callable[[str, dict], dict]

# 요청: "후보 재생성이 무한 반복되거나 LLM 호출 제한을 우회하지 못하도록 상한을 두세요."
# 재질문 상한(_MAX_ANSWER_RETRY)과 같은 원칙 — 상한 도달 시 LLM을 아예 호출하지 않고
# 코드가 즉시 안내 메시지로 막는다(무한 루프뿐 아니라 LLM 호출 자체를 원천 차단).
MAX_CANDIDATE_REGENERATIONS = 2

# 용준/Claude(2026-07-27, RAG-007 연결) — ai/meeting/graph는 ai.rag를 직접 import하지 않는다
# (evidence_lookup/index_target_evidence와 동일한 경계). 실제 구현
# (ai.rag.orchestration.ideation_external_evidence_service)은 backend가 만들어 주입한다.
# RAG-006 evidence_lookup(EvidenceLookup, list[dict] 반환)과 반환 타입이 다르다 — RAG-007은
# used_dataset_search/warnings 같은 응답 단위 메타데이터도 함께 돌려줘야 하므로(요청 11번)
# {"external_evidence": list[dict], "used_dataset_search": bool, "used_public_api_search": bool,
# "warnings": list[str]} 형태의 dict를 반환한다 — call_evidence_lookup을 재사용하지 않는다.
ExternalEvidenceLookupFn = Callable[[str, str], dict]

_EMPTY_EXTERNAL_EVIDENCE_RESULT: dict[str, Any] = {
    "external_evidence": [],
    "used_dataset_search": False,
    "used_public_api_search": False,
    "warnings": [],
}


def _call_external_evidence_lookup(
    external_evidence_lookup: ExternalEvidenceLookupFn | None, persona_id: str, query: str
) -> dict[str, Any]:
    """external_evidence_lookup(persona_id, query) -> dict를 안전하게 호출한다. 콜러블이
    없거나(use_rag=False 등) 예외를 던지면 빈 결과로 진행한다(요청 10번 — 외부자료 검색
    실패/부재가 후보 생성 자체를 막지 않는다)."""
    if external_evidence_lookup is None:
        return dict(_EMPTY_EXTERNAL_EVIDENCE_RESULT)
    try:
        result = external_evidence_lookup(persona_id, query)
    except Exception:
        return dict(_EMPTY_EXTERNAL_EVIDENCE_RESULT)
    if not isinstance(result, dict):
        return dict(_EMPTY_EXTERNAL_EVIDENCE_RESULT)
    return result


def _external_evidence_query(state: IdeationConvState, candidates: list[dict] | None = None) -> str:
    """RAG-007(외부 통계·시장·정책) 검색어 — 공모전명, 평가 기준(공고문 원문), 문제 상황,
    후보 아이디어 내용을 포함한다(요청 6번). RAG-006 project evidence_lookup이 쓰는
    _contest_query(공모전명+공고문만)와는 별도로 관리한다 — 두 RAG은 책임이 다르므로
    검색어 조합 로직도 독립적으로 둔다."""
    parts = [_contest_query(state)]
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        fragment = " ".join(str(candidate.get(field) or "") for field in ("title", "problem", "solution")).strip()
        if fragment:
            parts.append(fragment)
    return "\n".join(p for p in parts if p)


# pge/Claude(2026-07-27, 주제 브레인스토밍 — 네이버 트렌드 검색 연동) — ai/meeting/graph는
# ai.rag를 직접 import하지 않는다(evidence_lookup/external_evidence_lookup과 동일한 경계).
# 실제 구현(ai.rag.orchestration.ideation_trend_search_service)은 backend가 만들어 주입한다.
# persona_id를 받지 않는다(트렌드 검색은 위원 역할과 무관하게 candidate_planning 진입 시
# 한 번만 실행된다) — external_evidence_lookup과 시그니처가 다른 이유다.
TrendSearchLookupFn = Callable[[str], dict]

_EMPTY_TREND_RESULT: dict[str, Any] = {
    "trend_evidence": [],
    "used_naver_search": False,
    "warnings": [],
}


def _call_trend_search_lookup(trend_search_lookup: TrendSearchLookupFn | None, query: str) -> dict[str, Any]:
    """trend_search_lookup(query) -> dict를 안전하게 호출한다. 콜러블이 없거나
    (use_trend_search=False 등) 예외를 던지면 빈 결과로 진행한다 — 실시간 이슈 검색
    실패/부재가 후보 생성 자체를 막지 않는다(_call_external_evidence_lookup과 동일 원칙)."""
    if trend_search_lookup is None:
        return dict(_EMPTY_TREND_RESULT)
    try:
        result = trend_search_lookup(query)
    except Exception:
        return dict(_EMPTY_TREND_RESULT)
    if not isinstance(result, dict):
        return dict(_EMPTY_TREND_RESULT)
    return result


# pge/Claude(2026-07-28, 실측: NCP API Hub 431 "Request Header Fields Too Large") —
# _contest_query(state)는 공고문 요약 전체(최대 2000자, notice_document)까지 그대로 이어붙여
# RAG-006 벡터 검색어로는 적합하지만, 네이버 뉴스 검색은 짧은 키워드성 질의를 기대하는 API라
# 그대로 재사용하면 안 됐다 — 실측에서 query_len=1083까지 커져 NCP API Gateway가 요청 자체를
# 거부했다(구 openapi.naver.com에서는 나지 않던 증상). 공모전명(+선택 입력 initial_issue)만
# 쓰고 길이도 상한을 둔다.
_MAX_TREND_QUERY_CHARS = 100


def _trend_search_query(state: IdeationConvState) -> str:
    """트렌드(실시간 이슈) 검색어 — 공모전명에, 사용자가 브레인스토밍 화면에서 입력한
    이슈/관심사(initial_issue)가 있으면 더한다(공고문 전문은 쓰지 않는다 — 위 주석 참고).
    initial_issue가 없어도(선택 입력이라 비어 있을 수 있음) 공모전명만으로 검색을 시도한다."""
    notice = state.get("notice_and_criteria") or {}
    parts = [str(notice.get("competition_name") or "").strip()]
    initial_issue = state.get("initial_issue")
    if initial_issue:
        parts.append(str(initial_issue))
    query = " ".join(p for p in parts if p)
    return query[:_MAX_TREND_QUERY_CHARS]


def _merge_external_evidence_results(previous: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """candidate_planning과 candidate_feasibility가 같은 요청 안에서 각각 검색한
    external_evidence를 하나의 state 필드로 합친다(둘 다 있어야 요청 11번 응답 노출 요건을
    만족한다 — 뒤에 실행되는 feasibility가 planning의 검색 결과를 덮어쓰지 않는다).
    (source_id, document_id, chunk_id) 기준으로 중복 항목만 제거한다."""
    previous = previous or {}
    existing_items = previous.get("external_evidence") or []
    seen = {(item.get("source_id"), item.get("document_id"), item.get("chunk_id")) for item in existing_items}
    merged_items = list(existing_items)
    for item in new.get("external_evidence") or []:
        key = (item.get("source_id"), item.get("document_id"), item.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        merged_items.append(item)

    existing_warnings = previous.get("warnings") or []
    merged_warnings = list(existing_warnings)
    for warning in new.get("warnings") or []:
        if warning not in merged_warnings:
            merged_warnings.append(warning)

    return {
        "external_evidence": merged_items,
        "used_dataset_search": bool(previous.get("used_dataset_search")) or bool(new.get("used_dataset_search")),
        "used_public_api_search": bool(previous.get("used_public_api_search")) or bool(new.get("used_public_api_search")),
        "warnings": merged_warnings,
    }

_VALID_FEASIBILITY = {"high", "medium", "low"}
_VALID_RESOLUTIONS = {"select", "combine", "recommend", "unclear"}
# 결합(combine) 해석 시 merge_analysis.fit이 가질 수 있는 값 — feasibility와 값 집합은
# 같지만(high/medium/low) 의미가 다르므로(실현 가능성이 아니라 "결합 적합도") 별도 상수로
# 분리한다.
_VALID_MERGE_FIT = {"high", "medium", "low"}
# pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): topic_generation이 만드는 "주제"는 옛
# candidate_planning의 완성된 후보(solution/main_features/differentiation/success_metrics
# 포함)보다 훨씬 가볍다 — 실제 설계(어떻게 만들지)는 AI 아이디어 회의에서 하므로, 여기서는
# "무엇을 다룰지"를 특정하는 데 필요한 필드만 요구한다. one_liner는 core_value 필드명
# 그대로 재사용한다(CandidateCard가 이미 "핵심 가치" 라벨로 표시하는 필드라 프론트 코드
# 변경 없이 재사용 가능).
_REQUIRED_TOPIC_FIELDS = ("title", "problem", "target_user", "core_value", "contest_fit")
MIN_TOPICS = 2
MAX_TOPICS = 5

_REQUIRED_IDEA_FIELDS = ("title", "problem", "target_user", "solution")

# pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): 키워드 출처 3묶음 — 트렌드(실시간 검색),
# 공모전 연관(RAG-006 근거), 사용자 이슈(브레인스토밍 화면에서 입력한 initial_issue에서
# 추출). 세 출처가 반드시 다 있어야 하는 건 아니다(트렌드 검색 비활성/실패 시 trend가 없거나,
# initial_issue를 안 썼으면 user_issue가 없을 수 있음 — 기존 fail-open 원칙과 동일).
_VALID_KEYWORD_SOURCES = {"trend", "contest", "user_issue"}
_REQUIRED_KEYWORD_FIELDS = ("keyword_id", "keyword", "rationale")
# pge/Claude(2026-07-27, 실측 버그 수정): 원래 3~10이었으나, 트렌드 검색이 비활성/실패하고
# 사용자 이슈 입력도 없는(현재 프론트 기본 경로) 세션은 "contest" 출처 하나만 남는다 — 이때
# LLM이 정직하게 2개 이하만 뽑아도(지어내지 않으려고) 무조건 재시도 2회 실패 후 phase="failed"
# 로 세션 전체가 막혔다(실측: keywords_count_invalid, 완전히 같은 응답으로 재시도도 실패).
# fail-open 원칙(트렌드/사용자 이슈가 없어도 회의는 진행돼야 함)에 맞춰 최소값을 1로 낮춘다 —
# 근거가 얇으면 키워드도 적게 나오는 게 정상이지, 세션이 막히면 안 된다.
MIN_KEYWORDS = 1
MAX_KEYWORDS = 10

# pge/Claude(2026-07-28, 실측 요청): 화면에 항상 같은 순서로 그룹을 보여주기 위한 고정
# 출처 순서 — 프론트가 keyword_options 배열 안 원소 순서(=이 리스트로 정렬)에 의존해도
# 매번 같은 순서로 그룹이 나오게 한다(전에는 배치마다 LLM이 어떤 출처를 먼저 나열하는지가
# 들쭉날쭉해 화면에서 트렌드/공모전 그룹 순서가 그때그때 바뀌었다).
KEYWORD_SOURCE_ORDER = ("trend", "contest", "user_issue")

# pge/Claude(2026-07-28, 실측 요청: "키워드 누적 갯수는 트렌드 최대 8개, 공모전 누적 최대
# 5개") — "다른 키워드 추천받기"를 여러 번 눌러도 무한정 쌓이지 않도록 출처별 누적 상한을
# 둔다. user_issue는 사용자가 직접 남긴 이슈에서만 나오는 소량 출처라 별도 상한을 요청받지
# 않았으므로 여기 없으면 상한을 적용하지 않는다.
MAX_ACCUMULATED_KEYWORDS_BY_SOURCE = {"trend": 8, "contest": 5}


def _apply_keyword_accumulation_caps(keywords: list[dict]) -> list[dict]:
    """누적된 키워드 목록에서 출처별 상한을 초과하는 만큼 가장 오래된 것부터 버린다(최신
    키워드를 우선 유지한다) — keywords는 누적 순서(오래된 것이 앞)로 들어온다고 가정한다."""
    counts = {}
    for kw in keywords:
        source = kw.get("source")
        counts[source] = counts.get(source, 0) + 1
    drop_counts = {
        source: max(0, counts.get(source, 0) - cap)
        for source, cap in MAX_ACCUMULATED_KEYWORDS_BY_SOURCE.items()
    }
    dropped_so_far = {source: 0 for source in drop_counts}
    result = []
    for kw in keywords:
        source = kw.get("source")
        if source in drop_counts and dropped_so_far[source] < drop_counts[source]:
            dropped_so_far[source] += 1
            continue
        result.append(kw)
    return result


_SELECTION_QUESTION = (
    "제안된 후보 중 발전시키고 싶은 아이디어를 선택해 주세요. 번호나 제목을 입력하거나, "
    "'다시 추천', '전문가 추천'처럼 답할 수 있습니다."
)

_REGENERATE_KEYWORDS = (
    "다시 추천",
    "다른 후보",
    "재추천",
    "다시 만들어",
    "다시 제안",
    "새로운 후보",
    # 후보를 선택한 뒤 refinement 질문에 들어간 상태에서도 사용자가
    # 자연스럽게 쓰는 재생성 표현을 결정적으로 인식한다. "다시 설명"과 같은
    # 일반 명확화 요청을 재생성으로 오판하지 않도록 아이디어/기획 대상 표현만 두었다.
    "아이디어 다시 짜",
    "아이디어를 다시 짜",
    "아이디어 다시 만들",
    "아이디어를 다시 만들",
    "아이디어 새로",
    "새 아이디어",
    "기획 다시 짜",
    "처음부터 다시 짜",
)

# "1", "1번", "1번째", "candidate_1", "candidate 1" 처럼 순수하게 번호만 가리키는 경우만
# 코드가 결정적으로 처리한다 — 문장이 더 길거나 다른 말이 섞여 있으면(예: "1번인데 2번
# 기능도 넣고 싶어요") LLM 해석으로 넘긴다(요청: 단순 선택은 코드로, 자연어 결합/수정 요청은
# LLM으로). pge/Claude(2026-07-28, 실측 요청: 주제 후보 누적) — 후보가 더 이상 최대 5개로
# 고정되지 않고(재추천마다 이어붙임) 계속 쌓일 수 있어 자릿수를 고정하지 않는다. 범위 확인은
# 정규식이 아니라 _match_single_candidate의 인덱스 경계 검사(0 <= index < len(candidates))가
# 담당한다.
_NUMERIC_SELECT_RE = re.compile(r"^(candidate[_\s]?)?(\d+)\s*(번|번째)?$", re.IGNORECASE)


def _keyword_recommendation_retry_note(reason: str) -> str:
    """pge/Claude(2026-07-27, 실측 버그 수정): 재시도 프롬프트가 최초 시도와 완전히 동일하면
    같은 응답이 그대로 반복된다(실측: keywords_count_invalid로 2회 모두 동일 completion_tokens
    실패). 실패 사유별로 짧은 보정 지시를 덧붙여 재시도가 실제로 다른 결과를 내도록 한다."""
    guidance = {
        "keywords_count_invalid": (
            "keywords가 1~10개 범위를 벗어났습니다. trend_evidence/initial_issue가 비어 있어도 "
            "contest 출처만으로 평가 기준·목적·요구 기술을 서로 다른 각도로 나눠 최소 3개 이상 "
            "만들어 보세요(무리라면 확실한 근거가 있는 만큼만 만들어도 됩니다)."
        ),
        "keyword_id_missing_or_duplicate": "keyword_id를 kw_1, kw_2 순서로 모두 고유하게 채우세요.",
        "invalid_keyword_source": 'source는 "trend"/"contest"/"user_issue" 중 하나만 쓰세요.',
    }.get(reason, "이전 시도의 스키마 문제를 고쳐 다시 출력하세요.")
    return f"\n\n[이전 시도 보정 지시]\n{guidance}"


def _validate_keyword_recommendation_response(raw: dict) -> str | None:
    keywords = raw.get("keywords")
    if not isinstance(keywords, list) or not (MIN_KEYWORDS <= len(keywords) <= MAX_KEYWORDS):
        return "keywords_count_invalid"
    seen_ids: set[str] = set()
    for keyword in keywords:
        if not isinstance(keyword, dict):
            return "keyword_not_object"
        keyword_id = keyword.get("keyword_id")
        if _blank(keyword_id) or keyword_id in seen_ids:
            return "keyword_id_missing_or_duplicate"
        seen_ids.add(keyword_id)
        if keyword.get("source") not in _VALID_KEYWORD_SOURCES:
            return "invalid_keyword_source"
        for field in _REQUIRED_KEYWORD_FIELDS:
            if _blank(keyword.get(field)):
                return f"missing_or_empty_field:{field}"
    return None


def _validate_topic_generation_response(raw: dict, valid_keyword_ids: set[str] | None = None) -> str | None:
    """valid_keyword_ids(호출부가 실제로 프롬프트에 넣은 selected_keywords의 id 집합)를
    주면, 각 주제의 keyword_ids가 그 집합의 부분집합인지도 확인한다(pge/Claude(2026-07-28,
    실측 요청: "LLM이 엉뚱한 id를 지어내도 코드가 못 잡는다") — 프롬프트 규칙 3번("실제로
    그 주제를 구성하는 데 쓴 selected_keywords의 keyword_id만 담습니다")을 코드로 한 번 더
    확인하는 것뿐, 주제 내용 자체(title/problem/target_user 등)는 건드리지 않는다). None이면
    (호출부가 안 넘기면) 이 검사를 건너뛴다 — 기존 동작·테스트와 하위 호환."""
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not (MIN_TOPICS <= len(candidates) <= MAX_TOPICS):
        return "candidates_count_invalid"
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return "candidate_not_object"
        candidate_id = candidate.get("candidate_id")
        if _blank(candidate_id) or candidate_id in seen_ids:
            return "candidate_id_missing_or_duplicate"
        seen_ids.add(candidate_id)
        for field in _REQUIRED_TOPIC_FIELDS:
            if _blank(candidate.get(field)):
                return f"missing_or_empty_field:{field}"
        keyword_ids = candidate.get("keyword_ids")
        if not isinstance(keyword_ids, list) or not keyword_ids:
            return "missing_or_empty_field:keyword_ids"
        if valid_keyword_ids is not None and not set(keyword_ids) <= valid_keyword_ids:
            return "keyword_ids_not_in_selected_keywords"
    return None


def _validate_candidate_feasibility_response(raw: dict) -> str | None:
    reviews = raw.get("candidate_reviews")
    if not isinstance(reviews, list) or not reviews:
        return "candidate_reviews_missing"
    for review in reviews:
        if not isinstance(review, dict):
            return "candidate_review_not_object"
        if _blank(review.get("candidate_id")):
            return "missing_or_empty_field:candidate_id"
        if _blank(review.get("technical_approach")):
            return "missing_or_empty_field:technical_approach"
        if review.get("feasibility") not in _VALID_FEASIBILITY:
            return "invalid_feasibility_value"
    return None


def _validate_merge_analysis(merge_analysis: Any) -> str | None:
    """combine 해석 시 함께 요구되는 결합 분석 결과를 검증한다(요청: 공통 문제/공통 가치/
    결합 적합도/주 기능/보조 기능/충돌 지점/미확정 사항을 구조화된 필드로 강제)."""
    if not isinstance(merge_analysis, dict):
        return "merge_analysis_missing"
    if merge_analysis.get("fit") not in _VALID_MERGE_FIT:
        return "invalid_merge_analysis_fit"
    if _blank(merge_analysis.get("common_problem")) or _blank(merge_analysis.get("common_value")):
        return "missing_or_empty_field:merge_analysis_common_problem_or_value"
    for field in ("primary_features", "secondary_features", "conflicts", "open_questions"):
        if not isinstance(merge_analysis.get(field), list):
            return f"merge_analysis_{field}_not_list"
    return None


def _validate_candidate_selection_response(raw: dict) -> str | None:
    resolution = raw.get("resolution")
    if resolution not in _VALID_RESOLUTIONS:
        return "invalid_resolution"
    if resolution in ("combine", "recommend"):
        idea = raw.get("combined_idea")
        if not isinstance(idea, dict):
            return "combined_idea_missing"
        for field in _REQUIRED_IDEA_FIELDS:
            if _blank(idea.get(field)):
                return f"missing_or_empty_field:{field}"
    if resolution == "combine":
        problem = _validate_merge_analysis(raw.get("merge_analysis"))
        if problem is not None:
            return problem
    if resolution == "unclear" and _blank(raw.get("clarifying_question")):
        return "missing_or_empty_field:clarifying_question"
    if resolution == "select":
        ids = raw.get("selected_candidate_ids")
        if not isinstance(ids, list) or not ids or _blank(ids[0]):
            return "selected_candidate_ids_missing"
    return None


def _contest_query(state: IdeationConvState) -> str:
    notice = state.get("notice_and_criteria")
    if isinstance(notice, dict):
        return " ".join(str(v) for v in notice.values() if v)
    return str(notice or "")


def _merge_candidate_reviews(candidates: list[dict], reviews: list[dict]) -> list[dict]:
    """기획 전문가 후보(problem/target_user/solution 등)와 개발 전문가 검토 결과
    (required_data/technical_approach/feasibility 등)를 candidate_id 기준으로 병합한다.
    review가 없는 후보(개발 전문가가 누락했을 때의 방어적 처리)는 feasibility="medium",
    빈 배열/문자열로 채운다 — 병합 자체가 실패하지는 않는다(개별 리뷰 결측은 전체 재시도
    사유로 삼지 않는다, 최소 1개 이상 존재하면 되도록 검증 단계에서 이미 확인했다)."""
    review_by_id = {r.get("candidate_id"): r for r in reviews if isinstance(r, dict)}
    merged = []
    for candidate in candidates:
        review = review_by_id.get(candidate.get("candidate_id"), {})
        merged.append(
            {
                **candidate,
                "required_data": review.get("required_data") or [],
                "technical_approach": review.get("technical_approach") or "",
                "mvp_scope": review.get("mvp_scope") or "",
                "feasibility": review.get("feasibility") or "medium",
                "risks": review.get("risks") or [],
                "dev_notes": review.get("dev_notes"),
                "novelty_preservation": review.get("novelty_preservation") or "",
            }
        )
    return merged


def _normalize(text: str) -> str:
    return (text or "").strip()


def is_regenerate_request(text: str) -> bool:
    normalized = _normalize(text)
    return any(keyword in normalized for keyword in _REGENERATE_KEYWORDS)


# pge/Claude(2026-07-27, 실측 요청: "주제 후보 카드에서 키워드 다시 선택 가능하게") — 주제
# 후보 화면의 "다시 추천"(같은 키워드로 주제만 재생성)과는 다른, "키워드 자체를 다시 고르고
# 싶다"는 별도 의도. 자연어로 느슨하게 인식하면 위 REGENERATE_KEYWORDS와 겹쳐 오해될 수
# 있어(예: "다시" 포함) 버튼 전용 고정 문구만 정확히 일치시킨다 — 번호 선택/재추천과 같은
# "단순 요청은 코드로 결정적으로" 원칙.
KEYWORD_RESELECT_MESSAGE = "키워드 다시 선택"


def is_keyword_reselect_request(text: str) -> bool:
    return _normalize(text) == KEYWORD_RESELECT_MESSAGE


def _match_single_candidate(text: str, candidates: list[dict]) -> dict | None:
    """번호/후보 id/제목 정확 일치만 결정적으로 처리한다(요청: 단순 번호 선택은 코드로).
    그 외(복수 선택, 결합 의도, 수정 요청이 섞인 문장 등)는 None을 반환해 LLM 해석으로
    넘긴다."""
    normalized = _normalize(text)
    if not normalized or not candidates:
        return None
    match = _NUMERIC_SELECT_RE.match(normalized)
    if match:
        index = int(match.group(2)) - 1
        return candidates[index] if 0 <= index < len(candidates) else None
    for candidate in candidates:
        if normalized == str(candidate.get("candidate_id", "")):
            return candidate
    lowered = normalized.lower()
    for candidate in candidates:
        title = str(candidate.get("title", "")).strip().lower()
        if title and lowered == title:
            return candidate
    return None


def _find_candidate(candidates: list[dict], candidate_id: str | None) -> dict | None:
    if not candidate_id:
        return None
    for candidate in candidates:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    return None


def _candidate_document_id_key(idea: dict, source_ids: list[str]) -> str:
    """선택/결합된 아이디어를 target 문서로 색인할 때 쓸 candidate_id를 결정한다. 단일
    선택(select)은 원본 candidate_id를 그대로 쓰고, 결합(combine)/추천(recommend)은
    candidate_id가 없을 수 있으므로 source_ids를 정렬해 이어붙인 결정적 키를 만든다 —
    같은 두 후보를 같은 순서로 다시 결합하면 항상 같은 document_id가 나와야 재선택 시
    중복 색인이 아니라 upsert가 되기 때문이다(요청 3번)."""
    candidate_id = idea.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id.strip():
        return candidate_id.strip()
    if source_ids:
        return "combined-" + "-".join(sorted(str(sid) for sid in source_ids if sid))
    return "selection-" + hashlib.sha256((idea.get("title") or "").encode("utf-8")).hexdigest()[:10]


def _index_selected_candidate(
    *,
    state: IdeationConvState,
    idea: dict,
    source_ids: list[str],
    index_target_evidence: IndexTargetEvidenceFn | None,
) -> str | None:
    """선택된 후보를 target evidence로 색인하고(요청 2번), 실패해도 회의 state를 손상시키지
    않는다(요청 17-4번) — 실패하면 로그만 남기고 selected_idea_document_id=None으로 진행한다
    (이후 검색에서 그 후보의 target 근거가 아직 없는 것으로만 취급된다)."""
    if index_target_evidence is None:
        return None
    candidate_id = _candidate_document_id_key(idea, source_ids)
    started_event_fields = {
        "session_id": state["session_id"],
        "candidate_id": candidate_id,
        "title": sanitize_preview(idea.get("title") or "", limit=100),
    }
    try:
        result = index_target_evidence(
            "candidate",
            {"session_id": state["session_id"], "candidate_id": candidate_id, "candidate": idea},
        )
    except Exception as exc:  # noqa: BLE001 — 색인 실패는 회의를 막지 않는다.
        trace_event(
            "IDEATION_TARGET_EVIDENCE_UPSERT_FAILED",
            level=30,
            source_type="ideation_candidate",
            error=sanitize_preview(str(exc), limit=100),
            **started_event_fields,
        )
        return None
    return result.get("document_id") if isinstance(result, dict) else None


def _resolve_selection(
    state: IdeationConvState,
    *,
    idea: dict,
    reason: str,
    source: str,
    source_ids: list[str],
    user_selection_message: str | None = None,
    source_candidates: list[dict] | None = None,
    merge_analysis: dict | None = None,
    index_target_evidence: IndexTargetEvidenceFn | None = None,
) -> dict[str, Any]:
    """선택/결합/추천이 확정된 아이디어를 refinement로 넘길 형태로 변환한다. 이후
    refinement 흐름(질문/의견/재질문/최종 종합)은 전혀 수정하지 않고 그대로 재사용한다
    (요청 4번 "discovery에서 선택된 아이디어도 동일한 refinement 흐름으로 발전").

    user_selection_message/source_candidates/merge_analysis(요청: 후보 결합 컨텍스트
    보존)는 state에 그대로 저장되어, 이후 refinement 질문 프롬프트가 "1번과 2번"이
    실제로 무엇이었는지를 conversation_context의 최근 메시지에 우연히 남아있는 것에
    기대지 않고 명시적으로 참조할 수 있게 한다(ideation_conv_nodes.py::
    _selection_context_for 참고). source는 이미 "select"/"combine"/"recommend" 중
    하나이므로 selection_intent 값으로 그대로 재사용한다."""
    idea = dict(idea)
    idea["source"] = source
    idea["source_candidate_ids"] = source_ids

    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환): 이전에는 candidate_selection
    # 직후 planning_question(인터뷰) 노드가 require_combine_structure=True로 실행되어
    # "1번과 2번을 결합..."을 원본 후보 제목·핵심 내용까지 재진술했다. 이제는 그 노드를
    # 거치지 않고 곧바로 planning_expert_discussion(라운드테이블)으로 들어가므로, 원본
    # 후보 정보가 사라지지 않도록 이 요약 메시지에 "결합/선택 대상 후보" 섹션을 추가한다 —
    # planning_expert_discussion이 conversation_context.recent_messages로 이 메시지를 그대로
    # 보므로 별도 LLM 호출 없이 컨텍스트가 보존된다.
    source_lines = "\n".join(
        f"- {c.get('title', '')}: {c.get('problem', '')}" for c in (source_candidates or []) if isinstance(c, dict)
    )
    source_section = f"\n\n[선택/결합 대상 후보]\n{source_lines}" if source_lines else ""

    summary_message = _build_message(
        persona_id="ideation_facilitator",
        round_number=state["round"],
        message_type="summary",
        content=(
            f"선택된 아이디어: {idea.get('title', '')}\n"
            f"문제: {idea.get('problem', '')}\n"
            f"선택 이유: {reason}"
            f"{source_section}"
        ),
        referenced_message_ids=[],
        evidence=[],
    )
    title = idea.get("title", "") or ""
    problem = idea.get("problem", "") or ""
    initial_idea_text = f"{title} — {problem}".strip(" —") or None

    # 회의 안건 메시지 — LLM 호출 없이 현재 후보 데이터만 사용한다.
    opening_message = build_roundtable_opening_message(
        initial_idea_text or title,
        round_number=state["round"],
    )

    # 용준/Claude(2026-07-22, 요청: 선택된 아이디어를 target 문서로 생성) — 후보가 확정되는
    # 이 시점(사용자 API 호출이 끝나기 전, state에 selected_idea가 저장되는 것과 같은 노드
    # 실행 안)에 target evidence 색인을 동기적으로 완료한다. 다음 전문가 턴(planning_expert_
    # discussion)이 이 함수가 반환한 selected_idea_document_id로 RAG 검색을 수행하므로,
    # 색인이 끝나기 전에 다음 턴이 시작되는 race condition이 구조적으로 없다(그래프는 노드를
    # 순차 실행한다).
    selected_idea_document_id = _index_selected_candidate(
        state=state, idea=idea, source_ids=source_ids, index_target_evidence=index_target_evidence
    )

    return {
        "messages": [summary_message, opening_message],
        # pge/Claude(2026-07-28, 실측 요청: "브레인스토밍 이전 내역이 라운드테이블에 남아있음")
        # — 이 함수가 방금 만든 두 메시지가 라운드테이블의 첫 메시지가 되도록, 그 직전까지의
        # 길이를 기록한다(discovery 단계의 키워드/주제 문답을 프론트가 걸러낼 수 있도록).
        "refinement_message_offset": len(state["messages"]),
        "selected_idea": idea,
        "selected_idea_document_id": selected_idea_document_id,
        "selection_reason": reason,
        "user_idea": idea,
        "initial_idea": initial_idea_text,
        # 용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) — phase는 항상
        # 실제 canonical 상태("expert_discussion", 라운드테이블 진입점)로 유지하고, "같은
        # 요청 안에서 곧바로 planning_expert_discussion까지 이어간다"는 그래프 내부 라우팅
        # 신호는 next_route로 분리한다(ideation_conv_build.py::_route_after_candidate_selection
        # 참고) — 이전에는 phase="planning_question"을 그 신호로 재사용했는데, candidate_
        # selection 직후 곧바로 실행되는 다음 노드 도중 취소되면 이 "잠깐"의 phase가 그대로
        # 세션에 저장돼 이후 재개 시점에 재개 불가능한 phase로 오인해 거부했다.
        "phase": "expert_discussion",
        "next_route": "to_refinement",
        "selection_intent": source,
        "user_selection_message": user_selection_message,
        "source_candidates": source_candidates if source_candidates is not None else [],
        "merge_analysis": merge_analysis,
    }


_KEYWORD_SELECTION_QUESTION = (
    "요즘 이슈와 공모전 관련 키워드를 준비했습니다. 관심 있는 키워드를 자유롭게 선택해 "
    "주세요. 마음에 드는 키워드가 없으면 '다시 추천'이라고 답해 주세요."
)

_KEYWORD_SELECTION_MESSAGE_PREFIX = "선택한 키워드:"


def make_keyword_recommendation_node(
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
    trend_search_lookup: TrendSearchLookupFn | None = None,
) -> Callable[[IdeationConvState], dict]:
    """discovery(아이디어 발굴) 모드 진입 시 실행되는 첫 노드. 공모전 근거(RAG-006,
    evidence_lookup)와 실시간 이슈(네이버 검색, trend_search_lookup)를 각각 검색한 뒤, LLM
    한 번으로 트렌드 기준/공모전 연관/사용자 이슈 세 출처의 키워드를 함께 만든다 — "무엇을
    다룰지"만 정하는 단계이므로 실제 아이디어(문제/해결책 등)는 만들지 않는다(그건
    topic_generation 이후의 몫). 사용자가 관심 키워드를 다중 선택하면 keyword_selection
    노드가 이어받는다.

    pge/Claude(2026-07-27, 주제 브레인스토밍 재설계) — 옛 candidate_planning의 역할(트렌드+
    RAG 근거로 곧바로 완성된 후보를 만듦)을 대체한다. RAG-007(external_evidence_lookup)은
    쓰지 않는다 — 공모전 연관성 근거는 notice_and_criteria/retrieved_evidence만으로 충분하고,
    각 키워드에 이미 rationale로 근거가 남아 topic_generation 단계에서 다시 검색할 필요가
    없다."""

    def node(state: IdeationConvState) -> dict:
        retrieved = call_evidence_lookup(
            evidence_lookup, "planning_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        previous_keywords = state.get("keyword_options") or []
        regeneration_reason = None
        if previous_keywords:
            last_answer = _last_user_answer(state["messages"])
            regeneration_reason = (last_answer or {}).get("content")

        trend_query = _trend_search_query(state)
        trend_result = _call_trend_search_lookup(trend_search_lookup, trend_query)
        if trend_search_lookup is None:
            trend_status = "skipped"
        elif trend_result.get("used_naver_search"):
            trend_status = "ok"
        else:
            trend_status = "failed"
        trend_searched_at = (
            datetime.now(timezone.utc).strftime("%Y-%m-%d") if trend_search_lookup is not None else None
        )

        prompt = build_ideation_conv_keyword_recommendation_prompt(
            state["notice_and_criteria"],
            retrieved,
            trend_evidence=trend_result.get("trend_evidence"),
            initial_issue=state.get("initial_issue"),
            previous_keywords=previous_keywords,
            regeneration_reason=regeneration_reason,
        )
        raw, ok, attempts = _safe_call_structured_json(
            llm_call,
            prompt,
            _validate_keyword_recommendation_response,
            "keyword_recommendation",
            retry_note_for=_keyword_recommendation_retry_note,
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": "keyword_recommendation", "llm_calls_used": used}

        # pge/Claude(2026-07-28, 실측 요청: "다른 키워드 추천받기 눌러도 키워드가 안 사라지고
        # 쌓이게") — 이전에는 새 배치가 keyword_options를 통째로 대체했다. 이제 이전 배치에
        # 이어붙인다. 프롬프트가 매번 "kw_1"부터 다시 번호를 매기므로(build_ideation_conv_
        # keyword_recommendation_prompt의 출력 규칙), 그대로 이어붙이면 keyword_id가 겹친다 —
        # 화면에 보이는 keyword 텍스트가 아니라 내부 식별자일 뿐이므로, LLM 출력을 신뢰하지
        # 않고 누적 개수 기준으로 여기서 다시 번호를 매긴다(선택/렌더링 키의 유일성 보장).
        new_keywords = raw["keywords"]
        offset = len(previous_keywords)
        for i, keyword in enumerate(new_keywords, start=1):
            keyword["keyword_id"] = f"kw_{offset + i}"
        # pge/Claude(2026-07-28, 실측 요청: "키워드 누적 갯수는 트렌드 최대 8개, 공모전 누적
        # 최대 5개") — 무한정 쌓이지 않도록 출처별 상한을 적용한다(오래된 것부터 버림).
        accumulated_keywords = _apply_keyword_accumulation_caps(previous_keywords + new_keywords)

        question_message = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="question",
            content=_KEYWORD_SELECTION_QUESTION,
            referenced_message_ids=[],
            evidence=[],
        )
        return {
            "keyword_options": accumulated_keywords,
            "selected_keyword_ids": [],
            "contest_analysis": raw.get("contest_analysis"),
            "messages": [question_message],
            "phase": "awaiting_keyword_selection",
            "llm_calls_used": used,
            "trend_query": trend_query,
            "trend_evidence": trend_result.get("trend_evidence") or [],
            "trend_search_status": trend_status,
            "trend_searched_at": trend_searched_at,
        }

    return node


def _parse_selected_keyword_ids(text: str, options: list[dict]) -> list[str]:
    """"선택한 키워드: A, B, C" 형식(프론트 keywordSelectMessage와 약속된 형식)에서
    keyword_options의 keyword 텍스트와 정확히 일치하는 항목만 골라 keyword_id를 반환한다.
    정확 일치만 받아들인다(번호 선택과 같은 "단순 선택은 코드로 결정적으로" 원칙 — 유사/부분
    일치는 받아들이지 않아 잘못된 키워드가 섞이지 않는다)."""
    if _KEYWORD_SELECTION_MESSAGE_PREFIX not in text:
        return []
    _, _, rest = text.partition(_KEYWORD_SELECTION_MESSAGE_PREFIX)
    requested = {piece.strip() for piece in rest.split(",") if piece.strip()}
    if not requested:
        return []
    return [
        opt["keyword_id"]
        for opt in options
        if isinstance(opt, dict) and opt.get("keyword") in requested
    ]


def make_keyword_selection_node() -> Callable[[IdeationConvState], dict]:
    """사용자가 키워드 화면에서 보낸 답을 처리한다. LLM을 호출하지 않는다 — 고정 형식
    메시지를 코드가 결정적으로 파싱하거나, 재추천 요청을 감지한다(candidate_selection의
    재추천 처리와 동일한 원칙: 단순 선택/재요청은 코드로, 자연어 해석이 필요한 것만 LLM으로
    — 이 노드에는 애초에 해석이 필요한 자연어 케이스가 없다)."""

    def node(state: IdeationConvState) -> dict:
        options = state.get("keyword_options") or []
        last_answer = _last_user_answer(state["messages"])
        text = _normalize((last_answer or {}).get("content", ""))

        if is_regenerate_request(text):
            regen_count = state.get("candidate_regeneration_count", 0)
            if regen_count >= MAX_CANDIDATE_REGENERATIONS:
                notice = _build_message(
                    persona_id="ideation_facilitator",
                    round_number=state["round"],
                    message_type="summary",
                    content=(
                        f"키워드 재추천은 최대 {MAX_CANDIDATE_REGENERATIONS}회까지 가능합니다. "
                        "현재 제시된 키워드 중에서 골라 주세요."
                    ),
                    referenced_message_ids=[],
                    evidence=[],
                )
                return {"messages": [notice], "phase": "awaiting_keyword_selection"}
            return {
                "phase": "keyword_generation",
                "candidate_regeneration_count": regen_count + 1,
            }

        selected_ids = _parse_selected_keyword_ids(text, options)
        if not selected_ids:
            notice = _build_message(
                persona_id="ideation_facilitator",
                round_number=state["round"],
                message_type="question",
                content="키워드를 최소 1개 선택해 주세요.",
                referenced_message_ids=[],
                evidence=[],
            )
            return {"messages": [notice], "phase": "awaiting_keyword_selection"}

        return {"selected_keyword_ids": selected_ids, "phase": "topic_generation"}

    return node


def make_topic_generation_node(llm_call: LLMCall) -> Callable[[IdeationConvState], dict]:
    """선택된 키워드 조합으로 가벼운 주제 목록(2~5개)을 만드는 노드. 옛 candidate_planning의
    자리를 대신하지만, 실제 설계(solution/main_features/differentiation 등)는 만들지 않는다
    — 그건 AI 아이디어 회의(refinement 라운드테이블)에서 한다. RAG/트렌드 재검색도 하지
    않는다 — 근거는 이미 keyword_options의 rationale에 있다.

    pge/Claude(2026-07-27, 주제 브레인스토밍 재설계) — keyword_selection 노드가 "다시 추천"
    요청을 감지하면 phase를 "keyword_generation"으로 돌리므로 이 노드까지 오지 않는다. 이
    노드가 받는 "다시 추천"은 오직 하나 — 사용자가 이미 awaiting_candidate_selection(주제
    목록)에서 재추천을 요청한 경우로, candidate_selection 노드가 phase를 다시 이 노드로
    돌린다(선택한 키워드는 그대로 유지, 주제만 다시 만든다)."""

    def node(state: IdeationConvState) -> dict:
        options = state.get("keyword_options") or []
        selected_ids = set(state.get("selected_keyword_ids") or [])
        selected_keywords = [
            opt for opt in options if isinstance(opt, dict) and opt.get("keyword_id") in selected_ids
        ]

        previous_topics = state.get("idea_candidates") or []
        regeneration_reason = None
        if previous_topics:
            last_answer = _last_user_answer(state["messages"])
            regeneration_reason = (last_answer or {}).get("content")

        prompt = build_ideation_conv_topic_generation_prompt(
            state["notice_and_criteria"],
            selected_keywords,
            previous_topics,
            regeneration_reason,
        )
        raw, ok, attempts = _safe_call_structured_json(
            llm_call,
            prompt,
            lambda raw: _validate_topic_generation_response(raw, valid_keyword_ids=selected_ids),
            "topic_generation",
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": "topic_generation", "llm_calls_used": used}

        question_message = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="question",
            content=_SELECTION_QUESTION,
            referenced_message_ids=[],
            evidence=[],
        )
        # pge/Claude(2026-07-28, 실측 요청: "이미 생성된 후보도 그냥 두는 게 좋을 것 같다 —
        # 기억X 유지O") — 이전에는 새 배치가 idea_candidates를 통째로 대체했다(previous_topics는
        # dedup 근거로만 프롬프트에 넘기고 화면에서는 사라졌다). 이제는 화면에도 계속 쌓인다.
        # 프롬프트가 매번 "candidate_1"부터 다시 번호를 매기므로(키워드와 동일한 이유),
        # candidate_id 충돌을 막기 위해 누적 개수 기준으로 다시 번호를 매긴다 — 번호 선택은
        # candidate_id 문자열이 아니라 리스트 위치(인덱스)로 판단하므로(_match_single_candidate)
        # 여기서 id만 바꿔도 번호 선택 동작에는 영향이 없다.
        new_candidates = raw["candidates"]
        offset = len(previous_topics)
        for i, candidate in enumerate(new_candidates, start=1):
            candidate["candidate_id"] = f"candidate_{offset + i}"
        accumulated_candidates = previous_topics + new_candidates

        update: dict[str, Any] = {
            "idea_candidates": accumulated_candidates,
            "messages": [question_message],
            "phase": "awaiting_candidate_selection",
            "llm_calls_used": used,
        }
        if not state.get("original_idea_candidates"):
            # 최초 생성일 때만 캡처한다 — 재추천으로 idea_candidates가 갱신돼도 이 값은
            # 그대로 남아 최종 결과의 "최초 생성 주제" 이력이 된다(요청 8번과 동일 원칙).
            update["original_idea_candidates"] = new_candidates
        return update

    return node


# pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): 이 노드는 더 이상 그래프에 배선되지
# 않는다(ideation_conv_build.py 참고) — 후보 전체(옛 방식으로는 5개)를 검토하던 방식은
# "선택되지 않은 나머지에 LLM 비용을 쓴다"는 문제가 있었다. 대신 아래 _apply_feasibility_review
# 가 candidate_selection 확정 시점에 선택된 주제 1개에만 이 프롬프트/검증을 재사용한다.
# 파일은 지우지 않는다(참고용 + 하위 호환) — 다만 이 함수 자체를 다시 그래프에 연결하지는
# 말 것(사용자에게 정지 지점 없이 후보 전체를 검토하던 옛 동작으로 되돌아간다).
def make_candidate_feasibility_node(
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
    external_evidence_lookup: ExternalEvidenceLookupFn | None = None,
) -> Callable[[IdeationConvState], dict]:
    """개발 전문가가 기획 전문가의 후보들을 실현 가능성 관점에서 검토하고 병합해, 사용자에게
    선택 질문 하나를 던지고 멈춘다(awaiting_candidate_selection).

    용준/Claude(2026-07-27, RAG-007 연결) — external_evidence_lookup은 candidate_planning과
    별개로 dev_expert 역할로 한 번 더 검색한다(요청 5번 역할 매핑: dev_expert -> technology).
    두 노드의 검색 결과는 _merge_external_evidence_results로 합쳐 state에 누적한다 — 뒤에
    실행되는 이 노드가 planning의 검색 결과를 덮어쓰지 않는다."""

    def node(state: IdeationConvState) -> dict:
        candidates = state.get("idea_candidates") or []
        retrieved = call_evidence_lookup(
            evidence_lookup, "dev_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        external_result = _call_external_evidence_lookup(
            external_evidence_lookup, "dev_expert", _external_evidence_query(state, candidates)
        )
        prompt = build_ideation_conv_candidate_feasibility_prompt(
            state["notice_and_criteria"],
            candidates,
            retrieved,
            external_research=external_result.get("external_evidence"),
        )
        raw, ok, attempts = _safe_call_structured_json(
            llm_call, prompt, _validate_candidate_feasibility_response, "candidate_feasibility"
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": "candidate_feasibility", "llm_calls_used": used}

        merged = _merge_candidate_reviews(candidates, raw.get("candidate_reviews") or [])
        question_message = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="question",
            content=_SELECTION_QUESTION,
            referenced_message_ids=[],
            evidence=[],
        )
        merged_external = _merge_external_evidence_results(
            {"external_evidence": state.get("external_evidence", []), **(state.get("external_evidence_meta") or {})},
            external_result,
        )
        update: dict[str, Any] = {
            "idea_candidates": merged,
            "messages": [question_message],
            "phase": "awaiting_candidate_selection",
            "llm_calls_used": used,
            "external_evidence": merged_external["external_evidence"],
            "external_evidence_meta": {
                "used_dataset_search": merged_external["used_dataset_search"],
                "used_public_api_search": merged_external["used_public_api_search"],
                "warnings": merged_external["warnings"],
            },
        }
        if not state.get("original_idea_candidates"):
            # 최초 생성일 때만 캡처한다 — 재추천으로 idea_candidates가 갱신돼도 이 값은
            # 그대로 남아 최종 결과의 "최초 생성 후보" 이력이 된다(요청 8번).
            update["original_idea_candidates"] = merged
        return update

    return node


def _apply_feasibility_review(
    llm_call: LLMCall,
    state: IdeationConvState,
    evidence_lookup: EvidenceLookup | None,
    result: dict[str, Any],
    used: int,
) -> tuple[dict[str, Any], int]:
    """pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): 후보(주제) 전체가 아니라 방금 확정된
    주제 1개에 대해서만 개발 전문가 실현 가능성 검토를 수행한다 — 옛 candidate_feasibility가
    발굴 단계에서 후보 전부(예전엔 5개)를 검토하던 것과 달리, 선택되지 않은 나머지에는 LLM
    비용을 쓰지 않는다. 결과는 화면에 보이는 새 정지 지점 없이 조용히 selected_idea에
    병합된다 — 사용자는 추가 클릭 없이 그대로 라운드테이블(AI 아이디어 회의)로 넘어가고,
    그 오프닝 컨텍스트에 실현 가능성 정보가 이미 실려 있게 된다.

    실패해도(LLM 오류 등) 선택 확정 자체는 막지 않는다(기존 fail-open 원칙과 동일) — 그냥
    실현 가능성 필드 없이 진행한다. build_ideation_conv_candidate_feasibility_prompt/
    _validate_candidate_feasibility_response/_merge_candidate_reviews를 그대로 재사용한다
    (그 함수들은 "후보 목록"을 받을 뿐 개수를 가정하지 않으므로 1개짜리 목록도 그대로
    동작한다)."""
    selected_idea = result.get("selected_idea")
    if not isinstance(selected_idea, dict):
        return result, used

    # result["selected_idea_document_id"]는 _resolve_selection이 이 호출 직전에 방금 색인해
    # 채운 값이다 — state(선택 확정 이전의 원래 state)의 값(대부분 None)을 쓰면 stale
    # runtime_scope가 되므로, _runtime_scope_for(state) 대신 이 시점의 최신 값으로 직접
    # 구성한다(같은 요청 안에서 뒤이어 실행되는 라운드테이블 첫 전문가 검색과 동일한 원칙 —
    # ideation_conv_nodes.py의 stale closure 수정 사례 참고).
    runtime_scope = {
        "session_id": state.get("session_id"),
        "selected_candidate_document_id": result.get("selected_idea_document_id"),
    }
    retrieved = call_evidence_lookup(
        evidence_lookup, "dev_expert", _contest_query(state), runtime_scope=runtime_scope
    )
    prompt = build_ideation_conv_candidate_feasibility_prompt(state["notice_and_criteria"], [selected_idea], retrieved)
    raw, ok, attempts = _safe_call_structured_json(
        llm_call, prompt, _validate_candidate_feasibility_response, "selected_topic_feasibility_review"
    )
    used += attempts
    if not ok:
        return result, used

    merged = _merge_candidate_reviews([selected_idea], raw.get("candidate_reviews") or [])
    if not merged:
        return result, used
    return {**result, "selected_idea": merged[0]}, used


def make_candidate_selection_node(
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
    index_target_evidence: IndexTargetEvidenceFn | None = None,
) -> Callable[[IdeationConvState], dict]:
    """사용자의 후보 선택/결합/재추천/전문가추천 요청을 처리한다. 단순 번호·제목 선택과
    재추천 키워드는 LLM 없이 코드가 결정적으로 처리하고, 결합·전문가추천·모호한 답변만
    LLM을 호출한다(요청: 단순 선택은 코드로, 자연어 결합/수정 요청에만 LLM)."""

    def node(state: IdeationConvState) -> dict:
        candidates = state.get("idea_candidates") or []
        last_answer = _last_user_answer(state["messages"])
        text = _normalize((last_answer or {}).get("content", ""))

        matched = _match_single_candidate(text, candidates)
        if matched is not None:
            result = _resolve_selection(
                state,
                idea=matched,
                reason=f"사용자가 '{matched.get('title') or matched.get('candidate_id')}'를 선택했습니다.",
                source="select",
                source_ids=[matched.get("candidate_id")],
                user_selection_message=text,
                source_candidates=[matched],
                index_target_evidence=index_target_evidence,
            )
            result, feasibility_used = _apply_feasibility_review(
                llm_call, state, evidence_lookup, result, state.get("llm_calls_used", 0)
            )
            result["llm_calls_used"] = feasibility_used
            return result

        if is_keyword_reselect_request(text):
            regen_count = state.get("candidate_regeneration_count", 0)
            if regen_count >= MAX_CANDIDATE_REGENERATIONS:
                notice = _build_message(
                    persona_id="ideation_facilitator",
                    round_number=state["round"],
                    message_type="summary",
                    content=(
                        f"재추천은 최대 {MAX_CANDIDATE_REGENERATIONS}회까지 가능합니다. "
                        "현재 제시된 후보 중에서 선택하거나 '전문가 추천'을 요청해 주세요."
                    ),
                    referenced_message_ids=[],
                    evidence=[],
                )
                return {"messages": [notice], "phase": "awaiting_candidate_selection"}
            return {
                # pge/Claude(2026-07-28, 실측 요청: "이미 생성된 후보/키워드도 그냥 두는 게
                # 좋을 것 같다 — 기억X 유지O") — 이전에는 키워드/주제 목록을 통째로 비웠다.
                # 이제는 비우지 않는다: keyword_recommendation이 기존 keyword_options에
                # 새 키워드를 이어붙이고(누적, 출처별 상한 적용), idea_candidates는 이 노드가
                # 손대지 않으므로 topic_generation이 다음에 실행될 때까지 그대로 화면에
                # 남는다(새 주제가 생기면 그 위에 이어붙는다 — make_topic_generation_node
                # 참고).
                "phase": "keyword_generation",
                "candidate_regeneration_count": regen_count + 1,
            }

        if is_regenerate_request(text):
            regen_count = state.get("candidate_regeneration_count", 0)
            if regen_count >= MAX_CANDIDATE_REGENERATIONS:
                notice = _build_message(
                    persona_id="ideation_facilitator",
                    round_number=state["round"],
                    message_type="summary",
                    content=(
                        f"후보 재추천은 최대 {MAX_CANDIDATE_REGENERATIONS}회까지 가능합니다. "
                        "현재 제시된 후보 중에서 선택하거나 '전문가 추천'을 요청해 주세요."
                    ),
                    referenced_message_ids=[],
                    evidence=[],
                )
                return {"messages": [notice], "phase": "awaiting_candidate_selection"}
            return {
                # pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): 주제만 다시 만든다 —
                # 선택된 키워드는 그대로 유지한다(키워드부터 다시 고르고 싶으면 위
                # is_keyword_reselect_request 분기를 탄다).
                "phase": "topic_generation",
                "candidate_regeneration_count": regen_count + 1,
            }

        prompt = build_ideation_conv_candidate_selection_prompt(state["notice_and_criteria"], candidates, text)
        raw, ok, attempts = _safe_call_structured_json(
            llm_call, prompt, _validate_candidate_selection_response, "candidate_selection"
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": "candidate_selection", "llm_calls_used": used}

        resolution = raw["resolution"]
        if resolution == "unclear":
            question_message = _build_message(
                persona_id="ideation_facilitator",
                round_number=state["round"],
                message_type="question",
                content=raw["clarifying_question"],
                referenced_message_ids=[],
                evidence=[],
            )
            return {"messages": [question_message], "phase": "awaiting_candidate_selection", "llm_calls_used": used}

        source_ids = [i for i in (raw.get("selected_candidate_ids") or []) if not _blank(i)]
        source_candidates_full = [c for c in (_find_candidate(candidates, sid) for sid in source_ids) if c is not None]

        if resolution == "select":
            idea = _find_candidate(candidates, source_ids[0] if source_ids else None)
            if idea is None:
                # 방어적 처리 — LLM이 select를 골랐지만 candidate_id가 실제 후보와 매칭되지
                # 않는 경우(모델 오류). 조용히 잘못된 후보로 진행하지 않고 다시 묻는다.
                question_message = _build_message(
                    persona_id="ideation_facilitator",
                    round_number=state["round"],
                    message_type="question",
                    content="선택하신 후보를 특정할 수 없습니다. 후보 번호나 제목을 다시 알려 주세요.",
                    referenced_message_ids=[],
                    evidence=[],
                )
                return {
                    "messages": [question_message],
                    "phase": "awaiting_candidate_selection",
                    "llm_calls_used": used,
                }
            result = _resolve_selection(
                state,
                idea=idea,
                reason=raw.get("selection_reason", ""),
                source="select",
                source_ids=source_ids,
                user_selection_message=text,
                source_candidates=[idea],
                index_target_evidence=index_target_evidence,
            )
        elif resolution == "combine":
            merge_analysis = raw.get("merge_analysis") or {}
            if merge_analysis.get("fit") == "low":
                # 요청 5번 — 결합 적합도가 낮으면 바로 selected_idea를 확정하지 않는다.
                # 사용자가 선택한 두 후보가 무엇인지, 목적이 어떻게 다른지, 결합 시 발생하는
                # 범위/정체성 문제를 설명하고 주 방향을 물은 뒤 여전히 후보 선택 대기 상태로
                # 남는다 — 다만 이번 요청에서 파악한 컨텍스트(source_candidates/merge_analysis/
                # selection_intent/user_selection_message)는 잃지 않도록 state에 보존한다.
                candidate_lines = "\n".join(
                    f"{i + 1}. {c.get('title', '')} — {c.get('problem', '')}"
                    for i, c in enumerate(source_candidates_full)
                )
                low_fit_content = (
                    f"[선택한 후보]\n{candidate_lines or '(후보를 특정할 수 없습니다)'}\n\n"
                    f"[목적 차이]\n{merge_analysis.get('common_problem') or '두 후보가 공유하는 문제를 찾기 어렵습니다.'} "
                    "두 후보는 서로 다른 목표를 지향하고 있어, 단순히 합치면 범위가 넓어지거나 "
                    "제품의 정체성이 흐려질 수 있습니다.\n\n"
                    f"[결합 시 발생하는 문제]\n{_bullets(merge_analysis.get('conflicts'))}\n\n"
                    "[질문]\n두 후보 중 어느 쪽을 주 방향으로 삼고, 다른 쪽을 보조 요소로만 "
                    "반영할까요?"
                )
                message = _build_message(
                    persona_id="ideation_facilitator",
                    round_number=state["round"],
                    message_type="question",
                    content=low_fit_content,
                    referenced_message_ids=[],
                    evidence=[],
                )
                return {
                    "messages": [message],
                    "phase": "awaiting_candidate_selection",
                    "selection_intent": "combine",
                    "user_selection_message": text,
                    "source_candidates": source_candidates_full,
                    "merge_analysis": merge_analysis,
                    "llm_calls_used": used,
                }
            result = _resolve_selection(
                state,
                idea=raw["combined_idea"],
                reason=raw.get("selection_reason", ""),
                source="combine",
                source_ids=source_ids,
                user_selection_message=text,
                source_candidates=source_candidates_full,
                merge_analysis=merge_analysis,
                index_target_evidence=index_target_evidence,
            )
        else:  # resolution == "recommend"
            result = _resolve_selection(
                state,
                idea=raw["combined_idea"],
                reason=raw.get("selection_reason", ""),
                source="recommend",
                source_ids=source_ids,
                user_selection_message=text,
                source_candidates=source_candidates_full,
                index_target_evidence=index_target_evidence,
            )

        result, used = _apply_feasibility_review(llm_call, state, evidence_lookup, result, used)
        result["llm_calls_used"] = used
        assumptions = [a for a in (raw.get("unverified_assumptions") or []) if a and a not in state["unresolved_issues"]]
        if assumptions:
            result["unresolved_issues"] = list(state["unresolved_issues"]) + assumptions
        return result

    return node
