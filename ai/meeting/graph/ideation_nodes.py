# 작성자: 용준/Claude(2026-07-20)
# 목적: "아이디어 발전 회의(ideation)" LangGraph 노드. 기획 전문가(planning_expert)/개발
#       전문가(dev_expert)가 순차로 실행되며 서로의 직전 발언을 참조하고, 회의 진행자
#       (ideation_facilitator)가 라운드 종료 시 합의/이견/다음 행동을 판단하고, 최종
#       종합 노드가 아이디어 제안서를 만든다.
# import: prompts.build_ideation_turn_prompt 등(형제 패키지), 같은 패키지의 llm/state.

from __future__ import annotations

import inspect
from typing import Any, Callable

from prompts import (
    build_ideation_facilitator_prompt,
    build_ideation_synthesis_prompt,
    build_ideation_turn_prompt,
)

from .ideation_llm_log import log_llm_response
from .ideation_state import IdeationState
from .llm import LLMCall, parse_json_response

# backend가 주입하는 근거 조회 콜백: (persona_id, topic_query) -> retrieved_evidence(list[dict]).
# criterion 개념이 없는 ideation 모드에서는 RAG-004(근거 연결) 사후 링크 대신, 사전 검색
# 결과를 그대로 프롬프트에 넣는다(ai/rag/orchestration/ideation_evidence_service.py 참고).
# None이면 근거 없이 진행한다(검색 결과가 비면 프롬프트 스스로 "근거 부족"으로 처리하도록
# ideation_common.txt 근거 사용 규칙에 이미 명시돼 있다).
#
# 용준/Claude(2026-07-23, 요청: stale closure 수정) — 대화형 회의(ideation-conversation)
# evidence_lookup은 선택적으로 키워드 인자 runtime_scope(dict)도 받을 수 있다(호출 시점의
# 최신 session_id/selected_candidate_document_id). 이 alias 자체는 여전히 필수 시그니처만
# 표시한다 — 실제 호출은 아래 call_evidence_lookup()을 거쳐야 신구 evidence_lookup(2-인자
# 콜러블/테스트 fake 포함)이 모두 안전하게 동작한다.
EvidenceLookup = Callable[[str, str], list[dict]]


def _lookup_accepts_runtime_scope(fn: Any) -> bool:
    """evidence_lookup 콜러블이 runtime_scope 키워드 인자를 받는지 검사한다."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if "runtime_scope" in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


# 용준/Claude(2026-07-23, 요청: RAG 근거 실제 활용 강화 — evidence 참조 안정화) — chunk_id는
# 색인 시 결정적 해시로 생성되는 20자 안팎의 불투명 문자열이다(ai/rag/chunking/chunker.py::
# _generate_chunk_id, 예: "chk_3f9a1b2c4d5e6f70"). LLM이 claims[].evidence_refs에 이 값을
# 한 글자도 틀리지 않고 그대로 베껴 써야만 claim_grounding이 연결에 성공하는데, 실제 운영
# 로그에서 retrieved_evidence_count>0인데도 grounded_claim_count=0·evidence_status=
# "expert_judgment_only"인 턴이 반복되는 것과 일치하는 실패 양상이다 — LLM이 해시를 정확히
# 재현하는 데 실패하거나(존재하지 않는 chunk_id로 참조), 애초에 확신이 없어 인용 자체를
# 시도하지 않는 쪽을 택한다. 이번 턴에 실제로 검색된 항목에 한해 "E1", "E2"처럼 짧고 순서만
# 있으면 되는 참조 ID를 부여해 프롬프트/claim_grounding이 이 값으로 매칭하게 하면, LLM은
# 화면에 보이는 그대로의 짧은 토큰만 복사하면 된다 — 원래 chunk_id는 각 항목에 그대로 남아
# 있으므로(단순 추가 필드), frontend가 message.linked_evidence_refs를 evidence[].chunk_id와
# 대조하는 기존 계약(IdeationConversationScreen.jsx)은 ai.rag.evidence_linking.claim_grounding.
# ground_claims()가 매칭에 성공한 뒤 실제 chunk_id로 되돌려 채우므로 그대로 유지된다.
def _assign_evidence_refs(items: list[dict]) -> list[dict]:
    """검색 결과 각 항목에 이번 턴 한정 순번 참조 ID("ref")를 부여한 새 리스트를 반환한다.
    항목이 dict가 아니거나 이미 ref가 있으면 그대로 둔다(방어적 — 실제로는 항상 plain dict가
    들어온다)."""
    result: list[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            result.append(item)
            continue
        if "ref" in item:
            result.append(item)
            continue
        item_with_ref = dict(item)
        item_with_ref["ref"] = f"E{idx + 1}"
        result.append(item_with_ref)
    return result


def _exclude_user_session_answers(items: list[dict]) -> list[dict]:
    """용준/Claude(2026-07-30, 요청: "사용자 답변은 문서 기반 grounded evidence로 사용하지
    마세요") — ideation_target_indexing_service.py::index_user_answer_as_target()가 회의 중
    사용자 채팅 답변을 document_role="target"으로 색인해 두므로, 검색 자체는 이 청크를 그대로
    찾아온다(target starvation 대응 등 다른 목적으로는 여전히 필요). 하지만 evidence_lookup의
    이 최종 반환 지점을 지나면 곧바로 프롬프트의 retrieved_evidence/turn_evidence가 되고 ref가
    부여돼 LLM이 그대로 인용(evidence_refs)할 수 있게 되므로, 여기서 미리 제외해야 이후
    ground_claims 검증 대상·claim_evidence_links·reviewed_target_refs 등 근거 인용 파이프라인
    전체에 이 청크가 등장할 가능성 자체가 차단된다(대화 맥락/session_state 응답은 RAG 청크가
    아니라 state["messages"]에서 오므로 이 필터와 무관하게 그대로 동작한다)."""
    return [
        item
        for item in items
        if not (isinstance(item, dict) and item.get("ideation_source_type") == "user_session_answer")
    ]


def call_evidence_lookup(
    evidence_lookup: "EvidenceLookup | None",
    persona_id: str,
    query: str,
    *,
    runtime_scope: dict[str, Any] | None = None,
) -> list[dict]:
    """evidence_lookup(persona_id, query)를 호출하되, runtime_scope가 주어지고 콜러블이
    이를 지원하면 함께 전달한다(요청: candidate_selection 직후 같은 요청 안에서 이어지는
    전문가 검색이 요청 시작 시점에 캡처된 stale selected_candidate_document_id 대신 호출
    시점의 최신 값을 쓰도록 하기 위함 — ai/meeting/graph/ideation_conv_nodes.py의 evidence_lookup
    호출부가 이 함수를 통해서만 호출한다). ai/meeting은 ai.rag를 모르므로 runtime_scope의
    내용(session_id/selected_candidate_document_id)에 대해 아무 것도 가정하지 않고 그대로
    전달만 한다.

    runtime_scope를 지원하지 않는 콜러블(기존 (persona_id, query) 2-인자 테스트 fake, 배치형
    호출자 등)은 기존과 동일하게 2-인자로만 호출한다 — 완전히 하위 호환이다.

    반환 직전 사용자 세션 답변 청크를 제외하고(_exclude_user_session_answers) _assign_evidence_refs로
    순번 참조 ID를 부여한다 — evidence_lookup이 무엇을 반환하든(실제 RAG 검색이든 테스트
    fake든) 이 함수를 거치는 모든 호출자가 동일하게 안정적인 ref를 받고, 사용자 답변은 어떤
    호출자에게도 인용 가능한 근거로 노출되지 않는다."""
    if evidence_lookup is None:
        return []
    if runtime_scope is not None and _lookup_accepts_runtime_scope(evidence_lookup):
        items = evidence_lookup(persona_id, query, runtime_scope=runtime_scope)
    else:
        items = evidence_lookup(persona_id, query)
    return _assign_evidence_refs(_exclude_user_session_answers(items))

_VALID_STANCES = {"동의", "조건부_동의", "반박", "보완", "대안_제시", "사용자에게_질문"}
_TURN_LIST_FIELDS = (
    "observations",
    "proposals",
    "risks",
    "questions_for_expert",
    "questions_for_user",
    "evidence",
    "unresolved_issues",
)


def _topic_query(state: IdeationState) -> str:
    """근거 검색 질의를 만든다. 라운드 1은 사용자 아이디어 자체를, 이후 라운드는 아직
    풀리지 않은 쟁점을 우선한다(같은 검색을 반복하지 않기 위함)."""
    if state["round"] > 1 and state.get("unresolved_issues"):
        return " ".join(state["unresolved_issues"])
    idea = state["user_idea"]
    if isinstance(idea, dict):
        return " ".join(str(v) for v in idea.values() if v)
    return str(idea)


def _normalize_turn(raw: dict[str, Any], persona_id: str, round_number: int) -> dict[str, Any]:
    """LLM이 반환한 turn(raw)을 정규화한다.

    speaker_id/round는 LLM 출력을 신뢰하지 않고 항상 호출부가 아는 값으로 덮어쓴다 —
    기존 위원 평가 파이프라인(ai/meeting/graph/run.py::assemble_document())에서 LLM이
    persona_id를 지어내 committee 매칭이 깨졌던 것과 같은 사고를 방지하기 위한 동일한
    안전장치다. stance가 허용값이 아니면 "보완"으로 보수적으로 대체한다(회의가 어떤
    stance도 없이 진행되는 것을 막기 위함).
    """
    stance = raw.get("stance")
    if stance not in _VALID_STANCES:
        stance = "보완"
    turn = {
        "speaker_id": persona_id,
        "speaker_name": raw.get("speaker_name", persona_id),
        "role": raw.get("role", ""),
        "round": round_number,
        "topic": raw.get("topic", ""),
        "stance": stance,
        "summary": raw.get("summary", ""),
    }
    for field in _TURN_LIST_FIELDS:
        turn[field] = raw.get(field) or []
    return turn


def _round_context(state: IdeationState) -> dict[str, Any]:
    turns = state["turns"]
    return {
        "round": state["round"],
        "previous_turn": turns[-1] if turns else None,
        "consensus_so_far": state["consensus"],
        "unresolved_issues": state["unresolved_issues"],
        "user_answer": state.get("user_answer"),
    }


def _safe_call_json(
    llm_call: LLMCall, prompt: str, node_name: str = "unknown"
) -> tuple[dict[str, Any] | None, bool]:
    """LLM 호출 + JSON 파싱을 시도하고, 실패하면 한 번 재시도한다.

    요청 9번 14항(모델 호출 실패·JSON 파싱 실패 시 폴백)에 따라, 두 번째 시도도 실패하면
    예외를 올리지 않고 (None, False)를 반환해 호출부가 stage="실패"로 그래프를 안전하게
    끝낼 수 있게 한다(기존 reviewer/chair 노드에는 이 폴백이 없었으나, 회의 도중 사용자
    응답을 기다리는 흐름이 새로 생긴 ideation 모드에서는 실패를 조용히 전파시키지 않는
    것이 더 안전하다는 판단).

    node_name은 선택 인자(기본 "unknown")다 — 호출부 대부분이 이미 존재해 시그니처를
    바꾸지 않으려 기본값을 두되, ideation_llm_log.log_llm_response가 실패 응답까지
    포함한 원본을 로컬 로그 파일에 남길 때 어느 노드에서 났는지 구분하는 용도다."""
    for attempt in range(1, 3):
        raw_text = llm_call(prompt)
        try:
            parsed = parse_json_response(raw_text)
        except (ValueError, KeyError, TypeError):
            log_llm_response(
                node_name=node_name, attempt=attempt, ok=False, reason="json_parse_failed", raw_response=raw_text
            )
            continue
        log_llm_response(node_name=node_name, attempt=attempt, ok=True, reason=None, raw_response=raw_text)
        return parsed, True
    return None, False


def make_ideation_expert_node(
    persona_id: str,
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
) -> Callable[[IdeationState], dict]:
    """persona_id(planning_expert 또는 dev_expert) 전문가 1턴 노드를 만든다.

    같은 그래프 안에서 순차로 실행되므로(요청 5번 핵심 요구사항), _round_context()가
    state["turns"][-1](상대 전문가의 직전 발언)을 그대로 프롬프트에 넣을 수 있다 —
    기존 심사형 그래프(reviewer 노드, 완전 병렬)에서는 불가능했던 부분이다.
    """

    def node(state: IdeationState) -> dict:
        query = _topic_query(state)
        retrieved = call_evidence_lookup(evidence_lookup, persona_id, query)
        prompt = build_ideation_turn_prompt(
            persona_id,
            state["notice_and_criteria"],
            state["user_idea"],
            retrieved,
            _round_context(state),
        )
        raw, ok = _safe_call_json(llm_call, prompt, node_name=f"expert__{persona_id}")
        if not ok:
            return {"stage": "실패", "failed_node": f"expert__{persona_id}"}
        turn = _normalize_turn(raw, persona_id, state["round"])
        return {"turns": [turn]}

    return node


def make_facilitator_node(llm_call: LLMCall) -> Callable[[IdeationState], dict]:
    """라운드 종료 노드. 이번 라운드 두 전문가 발언을 종합해 합의/이견을 정리하고,
    다음 행동(continue_round/ask_user/finalize)을 판단한다.

    round_number >= max_rounds이면 LLM이 무엇을 반환하든 next_action을 "finalize"로
    강제한다(요청 9번 10항 "무한 반복 방지" — 서버가 신뢰하지 않고 직접 재계산하는 것은
    기존 rubric.py::build_dynamic_rubric_mapping()의 "배점 합계를 항상 서버에서
    재계산" 정책과 같은 원칙이다). continue_round면 round를 1 증가시킨다.
    """

    def node(state: IdeationState) -> dict:
        round_number = state["round"]
        turns_this_round = [t for t in state["turns"] if t.get("round") == round_number]
        prompt = build_ideation_facilitator_prompt(
            state["notice_and_criteria"],
            state["user_idea"],
            round_number,
            state["max_rounds"],
            turns_this_round,
            state["consensus"],
            state["unresolved_issues"],
        )
        raw, ok = _safe_call_json(llm_call, prompt, node_name="facilitator")
        if not ok:
            return {"stage": "실패", "failed_node": "facilitator"}

        next_action = raw.get("next_action")
        if round_number >= state["max_rounds"]:
            next_action = "finalize"
        elif next_action not in ("continue_round", "ask_user", "finalize"):
            next_action = "continue_round"

        new_consensus = list(state["consensus"])
        for item in raw.get("consensus", []) or []:
            if item not in new_consensus:
                new_consensus.append(item)

        update: dict[str, Any] = {
            "consensus": new_consensus,
            "unresolved_issues": raw.get("unresolved_issues", []) or [],
            "next_action": next_action,
        }
        if next_action == "ask_user":
            # round도 증가시킨다 — resume_ideation_state()로 재개되면 그래프가 다시
            # START부터 planning_expert를 돈다. round를 그대로 두면 이미 발언이 쌓인
            # 같은 라운드 번호로 새 turn이 또 쌓여 라운드 경계가 헷갈린다. 사용자 답변을
            # 반영한 다음 발언들은 새 라운드로 취급하는 게 자연스럽다.
            update["stage"] = "사용자_대기"
            update["pending_question"] = raw.get("question_for_user")
            update["round"] = round_number + 1
        elif next_action == "finalize":
            update["stage"] = "종합"
            update["pending_question"] = None
        else:
            update["stage"] = "진행중"
            update["pending_question"] = None
            update["round"] = round_number + 1
        return update

    return node


def make_synthesis_node(llm_call: LLMCall) -> Callable[[IdeationState], dict]:
    """회의 종료 시(next_action="finalize") 전체 발언과 누적 합의/미해결 쟁점으로
    최종 아이디어 제안서를 조립한다."""

    def node(state: IdeationState) -> dict:
        prompt = build_ideation_synthesis_prompt(
            state["notice_and_criteria"],
            state["user_idea"],
            state["turns"],
            state["consensus"],
            state["unresolved_issues"],
        )
        raw, ok = _safe_call_json(llm_call, prompt, node_name="synthesis")
        if not ok:
            return {"stage": "실패", "failed_node": "synthesis"}
        return {"idea_proposal": raw, "stage": "완료"}

    return node
