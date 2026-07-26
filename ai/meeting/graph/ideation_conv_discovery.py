# 작성자: 용준/Claude(2026-07-21)
# 목적: 대화형 아이디어 발전 회의(ideation-conversation)의 discovery(아이디어 발굴) 모드
#       전용 LangGraph 노드. 초기 아이디어가 없는 사용자를 위해 기획 전문가가 후보 2~3개를
#       만들고(candidate_planning), 개발 전문가가 후보별 실현 가능성을 검토해 병합하고
#       (candidate_feasibility), 사용자의 선택/결합/재추천/전문가추천 요청을 처리해
#       (candidate_selection) 최종적으로 refinement의 첫 phase("planning_question")로
#       합류시킨다. refinement 전용 노드(ideation_conv_nodes.py)는 이 파일이 건드리지 않고
#       그대로 재사용한다(_build_message/_safe_call_structured_json/_blank/_last_user_answer).
# import: prompts.build_ideation_conv_candidate_*(형제 패키지), 같은 패키지의
#         ideation_conv_state/ideation_conv_nodes/llm.

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from prompts import (
    build_ideation_conv_candidate_feasibility_prompt,
    build_ideation_conv_candidate_planning_prompt,
    build_ideation_conv_candidate_selection_prompt,
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

_VALID_FEASIBILITY = {"high", "medium", "low"}
_VALID_RESOLUTIONS = {"select", "combine", "recommend", "unclear"}
# 결합(combine) 해석 시 merge_analysis.fit이 가질 수 있는 값 — feasibility와 값 집합은
# 같지만(high/medium/low) 의미가 다르므로(실현 가능성이 아니라 "결합 적합도") 별도 상수로
# 분리한다.
_VALID_MERGE_FIT = {"high", "medium", "low"}
_REQUIRED_CANDIDATE_FIELDS = (
    "title",
    "problem",
    "target_user",
    "usage_scenario",
    "core_value",
    "solution",
    "differentiation",
    "contest_fit",
)
_REQUIRED_IDEA_FIELDS = ("title", "problem", "target_user", "solution")

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
# LLM으로).
_NUMERIC_SELECT_RE = re.compile(r"^(candidate[_\s]?)?([1-3])\s*(번|번째)?$", re.IGNORECASE)


def _validate_candidate_planning_response(raw: dict) -> str | None:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not (2 <= len(candidates) <= 3):
        return "candidates_count_invalid"
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return "candidate_not_object"
        candidate_id = candidate.get("candidate_id")
        if _blank(candidate_id) or candidate_id in seen_ids:
            return "candidate_id_missing_or_duplicate"
        seen_ids.add(candidate_id)
        for field in _REQUIRED_CANDIDATE_FIELDS:
            if _blank(candidate.get(field)):
                return f"missing_or_empty_field:{field}"
        if not isinstance(candidate.get("main_features"), list) or not candidate.get("main_features"):
            return "missing_or_empty_field:main_features"
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

    # 신청 양식이 있으면 작성 코치 전환 안내를, 없으면 기존 라운드테이블 안건을 붙인다.
    # 두 메시지 모두 LLM 호출 없이 현재 후보 데이터만 사용한다.
    if state.get("application_form_items"):
        opening_message = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="summary",
            content=f"선택한 '{title}' 아이디어를 바탕으로 신청 양식 초안을 함께 작성하겠습니다.",
            referenced_message_ids=[],
            evidence=[],
        )
    else:
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
        # 세션에 저장돼 이후 reply_to_interjection이 재개 불가능한 phase로 오인해 거부했다.
        "phase": "expert_discussion",
        "next_route": "to_refinement",
        "selection_intent": source,
        "user_selection_message": user_selection_message,
        "source_candidates": source_candidates if source_candidates is not None else [],
        "merge_analysis": merge_analysis,
    }


def make_candidate_planning_node(
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
) -> Callable[[IdeationConvState], dict]:
    """기획 전문가가 공모전 분석 + 서로 다른 아이디어 후보 2~3개를 만드는 노드. 개발
    전문가의 실현 가능성 검토(candidate_feasibility)로 정지 없이 바로 이어진다(요청
    3-2/3-3 — 후보 제시 전까지는 사용자에게 정지 지점을 보이지 않는다)."""

    def node(state: IdeationConvState) -> dict:
        retrieved = call_evidence_lookup(
            evidence_lookup, "planning_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        previous_candidates = state.get("idea_candidates") or []
        regeneration_reason = None
        if previous_candidates:
            last_answer = _last_user_answer(state["messages"])
            regeneration_reason = (last_answer or {}).get("content")

        prompt = build_ideation_conv_candidate_planning_prompt(
            state["notice_and_criteria"], retrieved, previous_candidates, regeneration_reason
        )
        raw, ok, attempts = _safe_call_structured_json(
            llm_call, prompt, _validate_candidate_planning_response, "candidate_planning"
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": "candidate_planning", "llm_calls_used": used}

        return {
            "idea_candidates": raw["candidates"],
            "contest_analysis": raw.get("contest_analysis"),
            "llm_calls_used": used,
        }

    return node


def make_candidate_feasibility_node(
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
) -> Callable[[IdeationConvState], dict]:
    """개발 전문가가 기획 전문가의 후보들을 실현 가능성 관점에서 검토하고 병합해, 사용자에게
    선택 질문 하나를 던지고 멈춘다(awaiting_candidate_selection)."""

    def node(state: IdeationConvState) -> dict:
        candidates = state.get("idea_candidates") or []
        retrieved = call_evidence_lookup(
            evidence_lookup, "dev_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        prompt = build_ideation_conv_candidate_feasibility_prompt(state["notice_and_criteria"], candidates, retrieved)
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
        update: dict[str, Any] = {
            "idea_candidates": merged,
            "messages": [question_message],
            "phase": "awaiting_candidate_selection",
            "llm_calls_used": used,
        }
        if not state.get("original_idea_candidates"):
            # 최초 생성일 때만 캡처한다 — 재추천으로 idea_candidates가 갱신돼도 이 값은
            # 그대로 남아 최종 결과의 "최초 생성 후보" 이력이 된다(요청 8번).
            update["original_idea_candidates"] = merged
        return update

    return node


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
            return _resolve_selection(
                state,
                idea=matched,
                reason=f"사용자가 '{matched.get('title') or matched.get('candidate_id')}'를 선택했습니다.",
                source="select",
                source_ids=[matched.get("candidate_id")],
                user_selection_message=text,
                source_candidates=[matched],
                index_target_evidence=index_target_evidence,
            )

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
                "phase": "candidate_generation",
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

        result["llm_calls_used"] = used
        assumptions = [a for a in (raw.get("unverified_assumptions") or []) if a and a not in state["unresolved_issues"]]
        if assumptions:
            result["unresolved_issues"] = list(state["unresolved_issues"]) + assumptions
        return result

    return node
