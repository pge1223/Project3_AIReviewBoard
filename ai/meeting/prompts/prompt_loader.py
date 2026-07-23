# 작성자: 경이
# 목적: LangGraph 위원/위원장 노드에서 쓸 실행 프롬프트를 조립한다.
#       공통 프롬프트(reviewer_prompt.txt / chair_prompt.txt)에 persona_cards.json 기반
#       페르소나 블록과 실행 컨텍스트(rubric/submission/evidence 등)를 <<...>> 토큰 치환으로 주입한다.
# import: 표준 라이브러리 json, pathlib. (외부 의존성 없음)
# 참고: 페르소나 프롬프트 prose(docs/prompts/persona_prompts.md)는 중복 저장하지 않고
#       구조화 카드(ai/meeting/personas/persona_cards.json)에서 render_persona_block()으로 생성한다.

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent
PERSONAS_DIR = PROMPTS_DIR.parent / "personas"
PERSONA_CARDS_PATH = PERSONAS_DIR / "persona_cards.json"

REVIEWER_TEMPLATE = "reviewer_prompt.txt"
CHAIR_TEMPLATE = "chair_prompt.txt"


def _read_text(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _as_text(obj: Any) -> str:
    """dict/list 는 JSON 문자열로, 이미 str 이면 그대로 사용."""
    return obj if isinstance(obj, str) else _json(obj)


@lru_cache(maxsize=1)
def load_persona_cards() -> dict[str, dict]:
    """persona_id -> persona card 매핑."""
    data = json.loads(PERSONA_CARDS_PATH.read_text(encoding="utf-8"))
    return {card["persona_id"]: card for card in data.get("personas", [])}


def get_persona_card(persona_id: str) -> dict:
    cards = load_persona_cards()
    if persona_id not in cards:
        raise KeyError(f"알 수 없는 persona_id: {persona_id!r}. 가능한 값: {sorted(cards)}")
    return cards[persona_id]


def reviewer_personas_for_domain(domain: str | None = None) -> list[str]:
    """도메인 태그로 위원(위원장 제외) persona_id 목록을 반환한다.
    실제 회의 참석자 선발·발언 순서는 LangGraph 빌더가 도메인 구성 파일에서 확정한다."""
    cards = load_persona_cards()
    result = []
    for pid, card in cards.items():
        if card.get("is_chair"):
            continue
        if domain is None or domain in card.get("domain_tags", []):
            result.append(pid)
    return result


def render_persona_block(card: dict) -> str:
    """persona card(구조화)를 프롬프트용 텍스트 블록으로 렌더링한다."""
    lines: list[str] = []
    lines.append(f"당신은 AI Review Board의 {card['display_name']}입니다. ({card.get('role', '')})")
    if card.get("mission"):
        lines.append(f"[미션] {card['mission']}")

    perspectives = card.get("evaluation_perspectives", [])
    if perspectives:
        lines.append("")
        lines.append("[평가 관점]")
        for p in perspectives:
            lines.append(f"- {p['name']}: {p['description']}")

    tone = card.get("tone", {})
    if tone:
        lines.append("")
        lines.append("[말투]")
        if tone.get("keywords"):
            lines.append(f"- 키워드: {', '.join(tone['keywords'])}")
        if tone.get("speaking_style"):
            lines.append(f"- {tone['speaking_style']}")
        if tone.get("preferred_structure"):
            lines.append(f"- 작성 순서: {' → '.join(tone['preferred_structure'])}")
        if tone.get("avoid_expressions"):
            avoid = ", ".join(f'"{e}"' for e in tone["avoid_expressions"])
            lines.append(f"- 피해야 할 표현: {avoid}")

    scope = card.get("scope", {})
    if scope.get("include"):
        lines.append("")
        lines.append("[포함 범위]")
        lines.extend(f"- {item}" for item in scope["include"])
    if scope.get("exclude"):
        lines.append("")
        lines.append("[제외 범위]")
        lines.extend(f"- {item}" for item in scope["exclude"])
    if scope.get("handoff_to"):
        lines.append("")
        lines.append("[전문 범위를 벗어난 판단 핸드오프]")
        for h in scope["handoff_to"]:
            lines.append(f"- {h['persona_id']}: {h['when']}")

    # 용준/Claude(2026-07-20): 아이디어 발전 회의(ideation) 전문가 카드에만 있는 선택 필드.
    # 기존 심사형 카드엔 이 키가 없으므로 없으면 그냥 건너뛴다(하위호환).
    if card.get("collaboration_stances"):
        lines.append("")
        lines.append("[발언 시 명시해야 하는 태도(stance) — 다음 중 하나를 선택]")
        lines.extend(f"- {s}" for s in card["collaboration_stances"])

    return "\n".join(lines)


def _render_evidence_guards(evidence_guards: Any | None) -> str:
    """(criterion_id, prompt_guard) 목록을 프롬프트용 텍스트로 렌더링한다(RAG-005 사전 판정).
    판정이 없으면(레거시 경로) 통상 평가 안내만 반환한다."""
    _default = "항목별 근거 충족도 판정이 제공되지 않았습니다. rubric에 따라 통상적으로 평가하세요."
    if not evidence_guards:
        return _default
    lines: list[str] = []
    for criterion_id, guard_text in evidence_guards:
        guard_text = (guard_text or "").strip()
        if guard_text:
            lines.append(f"- [{criterion_id}] {guard_text}")
    return "\n".join(lines) if lines else _default


def build_reviewer_prompt(
    persona_id: str,
    rubric: Any,
    submission: Any,
    retrieved_evidence: Any,
    previous_reviews: Any | None = None,
    evidence_guards: Any | None = None,
) -> str:
    """1~8번 위원용 실행 프롬프트를 조립한다. previous_reviews 기본값은 빈 배열(1회차).
    evidence_guards는 (criterion_id, prompt_guard) 목록(RAG-005 사전 판정)으로, 없으면
    통상 평가 안내가 들어간다."""
    card = get_persona_card(persona_id)
    if card.get("is_chair"):
        raise ValueError(f"{persona_id!r} 는 위원장입니다. build_chair_prompt()를 사용하세요.")
    template = _read_text(REVIEWER_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<RUBRIC_JSON>>": _as_text(rubric),
        "<<SUBMISSION_JSON>>": _as_text(submission),
        "<<RETRIEVED_EVIDENCE_JSON>>": _as_text(retrieved_evidence),
        "<<EVIDENCE_GUARD>>": _render_evidence_guards(evidence_guards),
        "<<PREVIOUS_REVIEWS_JSON>>": _as_text(previous_reviews if previous_reviews is not None else []),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_chair_prompt(
    reviewer_results: Any,
    rubric: Any,
    evidence: Any | None = None,
    chair_persona_id: str = "review_chair",
) -> str:
    """위원장용 실행 프롬프트를 조립한다."""
    card = get_persona_card(chair_persona_id)
    template = _read_text(CHAIR_TEMPLATE)
    replacements = {
        "<<CHAIR_BLOCK>>": render_persona_block(card),
        "<<RUBRIC_JSON>>": _as_text(rubric),
        "<<REVIEWER_RESULTS_JSON>>": _as_text(reviewer_results),
        "<<EVIDENCE_JSON>>": _as_text(evidence if evidence is not None else []),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


# ============================================================================
# 용준/Claude(2026-07-20): 아이디어 발전 회의(ideation) 모드 프롬프트 빌더.
# 기존 build_reviewer_prompt/build_chair_prompt는 건드리지 않고 새 함수만 추가한다.
# 템플릿: ideation_common.txt(전문가 공통, reviewer_prompt.txt와 같은 패턴),
# ideation_facilitator_prompt.txt(라운드 열기/닫기), ideation_synthesis_prompt.txt(최종 종합).
# ============================================================================

IDEATION_TURN_TEMPLATE = "ideation_common.txt"
IDEATION_FACILITATOR_TEMPLATE = "ideation_facilitator_prompt.txt"
IDEATION_SYNTHESIS_TEMPLATE = "ideation_synthesis_prompt.txt"


def build_ideation_turn_prompt(
    persona_id: str,
    notice_and_criteria: Any,
    user_idea: Any,
    retrieved_evidence: Any,
    round_context: Any,
) -> str:
    """기획/개발 전문가 1턴 실행 프롬프트를 조립한다.

    round_context는 전체 회의록이 아니라 상태(ideation_state.py)가 미리 압축한
    {"round": int, "previous_turn": dict|None, "consensus_so_far": [str],
    "unresolved_issues": [str]} 형태다 — 토큰 사용량을 줄이기 위해 직전 발언 1개와
    누적 요약만 전달한다(요청 7번 "토큰 사용량 고려").
    """
    card = get_persona_card(persona_id)
    template = _read_text(IDEATION_TURN_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<USER_IDEA_JSON>>": _as_text(user_idea),
        "<<RETRIEVED_EVIDENCE_JSON>>": _as_text(retrieved_evidence),
        "<<ROUND_CONTEXT_JSON>>": _as_text(round_context),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_facilitator_prompt(
    notice_and_criteria: Any,
    user_idea: Any,
    round_number: int,
    max_rounds: int,
    turns_this_round: Any,
    consensus_so_far: Any,
    unresolved_issues: Any,
) -> str:
    """라운드 시작(쟁점 정리) + 라운드 종료(합의/이견/다음 행동 판단) 겸용 프롬프트."""
    card = get_persona_card("ideation_facilitator")
    template = _read_text(IDEATION_FACILITATOR_TEMPLATE)
    replacements = {
        "<<FACILITATOR_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<USER_IDEA_JSON>>": _as_text(user_idea),
        "<<ROUND_NUMBER>>": str(round_number),
        "<<MAX_ROUNDS>>": str(max_rounds),
        "<<TURNS_THIS_ROUND_JSON>>": _as_text(turns_this_round),
        "<<CONSENSUS_SO_FAR_JSON>>": _as_text(consensus_so_far if consensus_so_far is not None else []),
        "<<UNRESOLVED_ISSUES_JSON>>": _as_text(unresolved_issues if unresolved_issues is not None else []),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_synthesis_prompt(
    notice_and_criteria: Any,
    user_idea: Any,
    all_turns: Any,
    consensus_so_far: Any,
    unresolved_issues: Any,
) -> str:
    """회의 종료 시 최종 아이디어 제안서를 조립하는 프롬프트."""
    card = get_persona_card("ideation_facilitator")
    template = _read_text(IDEATION_SYNTHESIS_TEMPLATE)
    replacements = {
        "<<FACILITATOR_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<USER_IDEA_JSON>>": _as_text(user_idea),
        "<<ALL_TURNS_JSON>>": _as_text(all_turns),
        "<<CONSENSUS_SO_FAR_JSON>>": _as_text(consensus_so_far if consensus_so_far is not None else []),
        "<<UNRESOLVED_ISSUES_JSON>>": _as_text(unresolved_issues if unresolved_issues is not None else []),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


# ============================================================================
# 용준/Claude(2026-07-20): 대화형(ideation-conversation) 개발용 프리뷰 프롬프트 빌더.
# 배치형(위 build_ideation_turn_prompt 등)과 별개 함수만 추가한다 — 기존 함수는 무수정.
# 최종 종합(finalize)은 새 템플릿을 만들지 않고 build_ideation_synthesis_prompt()를
# 그대로 재사용한다(all_turns 자리에 conversation messages를 그대로 전달해도 문제없다 —
# 그 프롬프트는 all_turns를 JSON 데이터로만 다루고 필드 이름을 강제하지 않는다).
# ============================================================================

IDEATION_CONV_QUESTION_TEMPLATE = "ideation_conv_question.txt"
IDEATION_CONV_DISCUSSION_TEMPLATE = "ideation_conv_discussion.txt"
IDEATION_CONV_SYNTHESIS_TEMPLATE = "ideation_conv_synthesis.txt"
IDEATION_CONV_SUFFICIENCY_TEMPLATE = "ideation_conv_sufficiency.txt"
# 용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선): 사용자가 질문에 답하는 대신 전문가
# 판단에 위임했을 때(answer_type="expert_delegation") 담당 전문가가 제안을 만드는 템플릿.
IDEATION_CONV_EXPERT_DELEGATION_TEMPLATE = "ideation_conv_expert_delegation.txt"

# 용준/Claude(2026-07-21): discovery(아이디어 발굴) 모드 전용 템플릿 3종. refinement 전용
# 템플릿(위 IDEATION_CONV_QUESTION_TEMPLATE 등)은 하나도 건드리지 않는다.
IDEATION_CONV_CANDIDATE_PLANNING_TEMPLATE = "ideation_conv_candidate_planning.txt"
IDEATION_CONV_CANDIDATE_FEASIBILITY_TEMPLATE = "ideation_conv_candidate_feasibility.txt"
IDEATION_CONV_CANDIDATE_SELECTION_TEMPLATE = "ideation_conv_candidate_selection.txt"


def build_ideation_conv_question_prompt(
    persona_id: str,
    notice_and_criteria: Any,
    user_idea: Any,
    retrieved_evidence: Any,
    conversation_context: Any,
    resolved_topics: Any = None,
    remaining_topics: Any = None,
    roadmap_allowed: bool = False,
    selection_context: Any = None,
    require_combine_structure: bool = False,
) -> str:
    """기획/개발 전문가의 "질문 턴" 실행 프롬프트를 조립한다(핵심 질문 1개 + 보조 질문 최대 2개).

    resolved_topics/remaining_topics/roadmap_allowed(요청: 질문 주제 구조화)는
    ideation_conv_state.py::remaining_topics_for()가 계산한 값을 그대로 받는다 —
    remaining_topics는 이미 우선순위 순으로 정렬돼 있고, roadmap 선행 주제가 충족되지
    않았으면 roadmap 자체가 빠져 있다(질문 노드가 애초에 roadmap을 후보로 볼 수 없게
    막는 1차 방어선). 값을 넘기지 않으면(호출부 하위 호환) 빈 목록으로 표시된다.

    selection_context/require_combine_structure(용준/Claude(2026-07-21, 후보 결합
    컨텍스트 보존, 요청 6·9번)는 discovery 모드에서 후보를 선택/결합/추천한 세션에서만
    값이 채워진다 — ideation_conv_nodes.py::_selection_context_for/
    _is_first_question_after_combine가 계산한다. selection_context는 값이 없으면 빈
    dict({})로 넘겨 "선택 컨텍스트 없음"을 표현한다."""
    card = get_persona_card(persona_id)
    template = _read_text(IDEATION_CONV_QUESTION_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<USER_IDEA_JSON>>": _as_text(user_idea),
        "<<RETRIEVED_EVIDENCE_JSON>>": _as_text(retrieved_evidence),
        "<<CONVERSATION_CONTEXT_JSON>>": _as_text(conversation_context),
        "<<RESOLVED_TOPICS_JSON>>": _as_text(resolved_topics if resolved_topics is not None else []),
        "<<REMAINING_TOPICS_JSON>>": _as_text(remaining_topics if remaining_topics is not None else []),
        "<<ROADMAP_ALLOWED>>": "true" if roadmap_allowed else "false",
        "<<SELECTION_CONTEXT_JSON>>": _as_text(selection_context if selection_context is not None else {}),
        "<<REQUIRE_COMBINE_STRUCTURE>>": "true" if require_combine_structure else "false",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_conv_discussion_prompt(
    persona_id: str,
    notice_and_criteria: Any,
    user_idea: Any,
    retrieved_evidence: Any,
    conversation_context: Any,
    speaks_second: bool,
    discussion_stage: str = "initial_position",
    active_issue_id: str | None = None,
    open_issues: Any = None,
    resolved_issues: Any = None,
    current_speaker: Any = None,
    responding_to_message: Any = None,
    evidence_plan_notice: str = "",
    application_form_items: Any = None,
) -> str:
    """기획/개발 전문가의 "보완 의견 턴" 실행 프롬프트를 조립한다.

    evidence_plan_notice(용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection"), 기본값
    빈 문자열): 비어 있으면 <<RETRIEVED_EVIDENCE_JSON>>은 기존과 완전히 동일한 문자열이다
    (템플릿 자체는 수정하지 않는다 — 플래그가 꺼져 있을 때 프롬프트가 한 글자도 달라지지
    않아야 한다는 요구사항 때문). 값이 있으면(discussion evidence planner가 active 모드로
    선정한 근거를 쓸 때만) retrieved_evidence JSON 앞에 안내문으로 붙인다 — 이번 발언이 다룰
    쟁점과, 선택되지 않은 근거는 이 목록에 없다는 사실을 명시한다.

    용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): discussion_stage는 이제
    "initial_position"(이번 쟁점에서 처음 말함, responding_to 없어도 됨) / "response"(상대
    또는 자신의 직전 발언에 실제로 반응함, responding_to 필수) 두 값뿐이다("review"/
    "revision" 구분은 없앴다 — 라운드 안에서 몇 번째 발언인지가 아니라 "이 쟁점에 대해
    처음 말하는가/반응하는가"만 따진다). speaks_second는 더 이상 next_action 판단 여부를
    결정하지 않는다(다음 발언자/라운드 종료는 그래프 라우터가 계산 — ideation_conv_build.py::
    _route_next_expert_turn) — 하위 호환을 위해 필드와 프롬프트 문구는 유지하되
    "discussion_stage=='response'와 동일한 신호"로만 취급한다.

    active_issue_id/open_issues/resolved_issues(새 파라미터, 기본값 None — 호출부가 안 넘기면
    빈 값으로 렌더돼 기존 동작과 동일)는 지금 회의가 쟁점 단위로 어디까지 왔는지 전문가에게
    알려준다 — 전문가는 이 값을 보고 새 쟁점을 열지, 기존 쟁점을 이어갈지 판단한다.

    가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입): application_form_items는 순수
    추가 파라미터(기본값 None)다 — [{"field_name","description","char_limit"}] 형태의
    항목 목록을 "참고 자료"(지시 아님)로 주입한다. None/빈 리스트면 템플릿의
    [신청양식 참고 규칙] 섹션이 실질적으로 아무 효과가 없다(참고할 항목 자체가 없으므로)."""
    card = get_persona_card(persona_id)
    template = _read_text(IDEATION_CONV_DISCUSSION_TEMPLATE)
    evidence_text = _as_text(retrieved_evidence)
    if evidence_plan_notice:
        evidence_text = f"{evidence_plan_notice}\n\n{evidence_text}"
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<USER_IDEA_JSON>>": _as_text(user_idea),
        "<<RETRIEVED_EVIDENCE_JSON>>": evidence_text,
        "<<CONVERSATION_CONTEXT_JSON>>": _as_text(conversation_context),
        "<<SPEAKS_SECOND>>": "true" if speaks_second else "false",
        "<<DISCUSSION_STAGE>>": discussion_stage,
        "<<ACTIVE_ISSUE_ID>>": active_issue_id or "(아직 없음 — 이번 발언에서 새로 엽니다)",
        "<<OPEN_ISSUES_JSON>>": _as_text(open_issues if open_issues is not None else []),
        "<<RESOLVED_ISSUES_JSON>>": _as_text(resolved_issues if resolved_issues is not None else []),
        "<<CURRENT_SPEAKER_JSON>>": _as_text(current_speaker if current_speaker is not None else {}),
        "<<RESPONDING_TO_MESSAGE_JSON>>": _as_text(
            responding_to_message if responding_to_message is not None else {}
        ),
        "<<APPLICATION_FORM_ITEMS_JSON>>": _as_text(application_form_items if application_form_items else None),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


IDEATION_CONV_DISCUSSION_FACILITATOR_TEMPLATE = "ideation_conv_discussion_facilitator.txt"
IDEATION_CONV_CANVAS_UPDATE_TEMPLATE = "ideation_conv_canvas_update.txt"


def build_ideation_conv_canvas_update_prompt(
    current_canvas: Any,
    selected_idea: Any,
    initial_idea: str | None,
    planning_position: Any,
    development_review: Any,
    revised_proposal: Any,
    consensus_so_far: Any,
    unresolved_issues: Any,
    notice_and_criteria: Any,
) -> str:
    """한 라운드의 발언을 바탕으로 아이디어 캔버스 갱신 프롬프트를 조립한다."""
    template = _read_text(IDEATION_CONV_CANVAS_UPDATE_TEMPLATE)
    replacements = {
        "<<CURRENT_CANVAS_JSON>>": _as_text(current_canvas if current_canvas is not None else None),
        "<<SELECTED_IDEA_JSON>>": _as_text(selected_idea if selected_idea is not None else None),
        "<<INITIAL_IDEA>>": (initial_idea or "").strip() or "null",
        "<<PLANNING_POSITION_JSON>>": _as_text(planning_position),
        "<<DEVELOPMENT_REVIEW_JSON>>": _as_text(development_review),
        "<<REVISED_PROPOSAL_JSON>>": _as_text(revised_proposal if revised_proposal is not None else None),
        "<<CONSENSUS_SO_FAR_JSON>>": _as_text(consensus_so_far if consensus_so_far is not None else []),
        "<<UNRESOLVED_ISSUES_JSON>>": _as_text(unresolved_issues if unresolved_issues is not None else []),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_conv_discussion_facilitator_prompt(
    notice_and_criteria: Any,
    planning_position: Any,
    development_review: Any,
    revised_proposal: Any,
    consensus_so_far: Any,
    unresolved_issues: Any,
    decided_next_action: str,
    round_number: int,
    max_rounds: int,
    open_issues: Any = None,
    resolved_issues: Any = None,
    stop_reason: str | None = None,
    next_issue_hint: str | None = None,
) -> str:
    """진행자가 전문가 토론을 정리하고 사용자 질문이 꼭 필요한지 판단하는 프롬프트를
    조립한다. decided_next_action은 이미 라우터(ideation_conv_build.py::
    _route_next_expert_turn)가 정한 값을 그대로 받는다 — 이 프롬프트가 그 결정 자체를
    바꾸지 않는다(요청: 기존 라운드 진행/max_rounds 강제 로직 재사용).

    용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): planning_position/
    development_review는 더 이상 "이번 라운드 발언"이 아니라 각 전문가의 "가장 최근 발언"
    스냅샷이다(발언 횟수가 라운드마다 고정이 아니게 됐으므로). open_issues/resolved_issues/
    stop_reason(모두 새 파라미터, 기본값 None — 호출부가 안 넘기면 기존과 동일하게 렌더)은
    이 라운드가 왜 끝났는지, 아직 안 끝난 쟁점이 남아있는지를 진행자에게 알려준다.

    용준/Claude(2026-07-23, 요청: "진행자에게 갱신된 쟁점 상태 전달") — next_issue_hint는
    발언 상한으로 방금 쟁점이 강제 종료되고 다음 공식 쟁점으로 로테이션됐을 때만 채워지는
    사람이 읽는 제목이다. 채워져 있으면 spoken_text가 방금 닫힌 쟁점을 더 논의하자고
    요구하지 않고 이 다음 쟁점으로 자연스럽게 넘어가야 한다는 신호다."""
    card = get_persona_card("ideation_facilitator")
    template = _read_text(IDEATION_CONV_DISCUSSION_FACILITATOR_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<PLANNING_POSITION_JSON>>": _as_text(planning_position),
        "<<DEVELOPMENT_REVIEW_JSON>>": _as_text(development_review),
        "<<REVISED_PROPOSAL_JSON>>": _as_text(revised_proposal if revised_proposal is not None else None),
        "<<CONSENSUS_SO_FAR_JSON>>": _as_text(consensus_so_far if consensus_so_far is not None else []),
        "<<UNRESOLVED_ISSUES_JSON>>": _as_text(unresolved_issues if unresolved_issues is not None else []),
        "<<DECIDED_NEXT_ACTION>>": decided_next_action,
        "<<ROUND_NUMBER>>": str(round_number),
        "<<MAX_ROUNDS>>": str(max_rounds),
        "<<OPEN_ISSUES_JSON>>": _as_text(open_issues if open_issues is not None else []),
        "<<RESOLVED_ISSUES_JSON>>": _as_text(resolved_issues if resolved_issues is not None else []),
        "<<STOP_REASON>>": stop_reason or "",
        "<<NEXT_ISSUE_HINT>>": next_issue_hint or "",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_conv_sufficiency_prompt(
    persona_id: str,
    pending_question: str,
    user_answer: str,
    retry_count: int,
    conversation_context: Any,
    expected_answer_type: str | None = None,
    user_idea: Any = None,
    idea_candidates: Any = None,
) -> str:
    """사용자 메시지가 답변(answer)/설명 요청(clarification_request)/불충분한 답변
    (insufficient_answer) 중 무엇인지 판정하는 프롬프트를 조립한다(요청 3번 재질문 조건 +
    5번 무한 반복 방지 + 용어 설명 요청을 불충분한 답변으로 오판하지 않기).
    질문/의견 턴과 별개의 짧은 LLM 호출 1회로, 회의 내용은 만들지 않고 판정만 한다.

    expected_answer_type(질문 노드가 만든 이번 질문의 기대 답변 유형 — 예: "preference",
    "selection")을 넘기면, 이 판정이 "답변 충분성"(방금 질문에 답했는가)과 "아이디어
    완성도"(전체적으로 충분히 구체적인가)를 혼동하지 않도록 프롬프트가 요구 수준을
    맞춘다. None이면(질문 노드가 값을 만들지 못했거나 구버전 응답) 템플릿이 "미상"으로
    표시하고 기존의 일반 기준으로만 판정한다.

    user_idea/idea_candidates는 clarification_request일 때 "현재 후보와 대화 맥락에 맞는"
    선택지를 만드는 근거 자료로만 쓰인다(답을 대신 결정하는 근거가 아니다)."""
    card = get_persona_card(persona_id)
    template = _read_text(IDEATION_CONV_SUFFICIENCY_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<PENDING_QUESTION>>": _as_text(pending_question),
        "<<USER_ANSWER>>": _as_text(user_answer),
        "<<RETRY_COUNT>>": str(retry_count),
        "<<CONVERSATION_CONTEXT_JSON>>": _as_text(conversation_context),
        "<<EXPECTED_ANSWER_TYPE>>": expected_answer_type or "미상(알 수 없음)",
        "<<USER_IDEA_JSON>>": _as_text(user_idea if user_idea is not None else {}),
        "<<IDEA_CANDIDATES_JSON>>": _as_text(idea_candidates if idea_candidates is not None else []),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_conv_expert_delegation_prompt(
    persona_id: str,
    notice_and_criteria: Any,
    user_idea: Any,
    retrieved_evidence: Any,
    conversation_context: Any,
    pending_question: str,
    stage: str = "initial",
    counterpart_review: Any = None,
) -> str:
    """용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선): 사용자가 pending_question에 답하는
    대신 전문가 판단에 위임했을 때(answer_type="expert_delegation") 담당 전문가가 자신의
    평가 범위 안에서 임시 가정을 제안하는 프롬프트를 조립한다.

    stage/counterpart_review(용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간
    상호 검토로 확장)는 이 제안이 상대 전문가의 검토를 반영해 수정하는 두 번째 발언인지
    나타낸다 — stage="initial"(기본값, 기존 호출부 하위 호환)이면 counterpart_review는
    무시되고 프롬프트에 "검토 없음"으로 표시된다."""
    card = get_persona_card(persona_id)
    template = _read_text(IDEATION_CONV_EXPERT_DELEGATION_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<USER_IDEA_JSON>>": _as_text(user_idea),
        "<<RETRIEVED_EVIDENCE_JSON>>": _as_text(retrieved_evidence),
        "<<CONVERSATION_CONTEXT_JSON>>": _as_text(conversation_context),
        "<<PENDING_QUESTION>>": _as_text(pending_question),
        "<<STAGE>>": stage,
        "<<COUNTERPART_REVIEW_JSON>>": _as_text(counterpart_review if counterpart_review is not None else None),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


IDEATION_CONV_EXPERT_DELEGATION_REVIEW_TEMPLATE = "ideation_conv_expert_delegation_review.txt"
IDEATION_CONV_EXPERT_DELEGATION_FACILITATOR_TEMPLATE = "ideation_conv_expert_delegation_facilitator.txt"


def build_ideation_conv_expert_delegation_review_prompt(
    persona_id: str,
    notice_and_criteria: Any,
    user_idea: Any,
    retrieved_evidence: Any,
    conversation_context: Any,
    pending_question: str,
    proposal_under_review: Any,
) -> str:
    """용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장): 담당
    전문가의 임시 제안을 반대 역할 전문가(persona_id)가 검토하는 프롬프트를 조립한다."""
    card = get_persona_card(persona_id)
    template = _read_text(IDEATION_CONV_EXPERT_DELEGATION_REVIEW_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<USER_IDEA_JSON>>": _as_text(user_idea),
        "<<RETRIEVED_EVIDENCE_JSON>>": _as_text(retrieved_evidence),
        "<<CONVERSATION_CONTEXT_JSON>>": _as_text(conversation_context),
        "<<PENDING_QUESTION>>": _as_text(pending_question),
        "<<PROPOSAL_UNDER_REVIEW_JSON>>": _as_text(proposal_under_review),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_conv_expert_delegation_facilitator_prompt(
    notice_and_criteria: Any,
    pending_question: str,
    proposal: Any,
    review: Any,
    revision: Any,
) -> str:
    """용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장): 제안 ->
    검토 -> (있다면) 수정까지 끝난 뒤, 진행자가 최종 권고안 하나로 정리하는 프롬프트를
    조립한다. 출력 스키마에 사용자 재질문 필드가 없어 구조적으로 같은 질문을 반복할 수
    없다(요청: "다시 사용자에게 같은 질문을 넘기면 안 됩니다")."""
    card = get_persona_card("ideation_facilitator")
    template = _read_text(IDEATION_CONV_EXPERT_DELEGATION_FACILITATOR_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<PENDING_QUESTION>>": _as_text(pending_question),
        "<<PROPOSAL_JSON>>": _as_text(proposal),
        "<<REVIEW_JSON>>": _as_text(review),
        "<<REVISION_JSON>>": _as_text(revision if revision is not None else None),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_conv_synthesis_prompt(
    notice_and_criteria: Any,
    user_idea: Any,
    all_messages: Any,
    consensus_so_far: Any,
    unresolved_issues: Any,
    discovery_history: Any | None = None,
) -> str:
    """사용자가 "주제 확정하고 초안 받기"를 눌렀을 때만 호출되는 최종 종합 프롬프트.
    배치형 build_ideation_synthesis_prompt와 출력 스키마는 동일(idea_proposal)하지만,
    입력 섹션 이름을 all_turns가 아니라 all_messages(대화 메시지 스키마)로 명확히 한다.

    discovery_history는 discovery(아이디어 발굴) 모드에서 finalize할 때만 채워 넘긴다
    (요청 8번 — 최초 생성 후보/선택된 후보/선택 이유를 최종 결과에 포함). refinement
    세션은 이 인자를 넘기지 않으므로(기본값 None) 기존 호출부는 전혀 바뀌지 않는다."""
    card = get_persona_card("ideation_facilitator")
    template = _read_text(IDEATION_CONV_SYNTHESIS_TEMPLATE)
    replacements = {
        "<<FACILITATOR_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<USER_IDEA_JSON>>": _as_text(user_idea),
        "<<ALL_MESSAGES_JSON>>": _as_text(all_messages),
        "<<CONSENSUS_SO_FAR_JSON>>": _as_text(consensus_so_far if consensus_so_far is not None else []),
        "<<UNRESOLVED_ISSUES_JSON>>": _as_text(unresolved_issues if unresolved_issues is not None else []),
        "<<DISCOVERY_HISTORY_JSON>>": _as_text(discovery_history),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


# ============================================================================
# 용준/Claude(2026-07-21): discovery(아이디어 발굴) 모드 프롬프트 빌더 3종.
# refinement 전용 빌더(build_ideation_conv_question_prompt 등)는 무수정.
# ============================================================================


def build_ideation_conv_candidate_planning_prompt(
    notice_and_criteria: Any,
    retrieved_evidence: Any,
    previous_candidates: Any | None = None,
    regeneration_reason: str | None = None,
) -> str:
    """기획 전문가의 "후보 생성" 프롬프트를 조립한다(공모전 분석 + 서로 다른 후보 2~3개)."""
    card = get_persona_card("planning_expert")
    template = _read_text(IDEATION_CONV_CANDIDATE_PLANNING_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<RETRIEVED_EVIDENCE_JSON>>": _as_text(retrieved_evidence),
        "<<PREVIOUS_CANDIDATES_JSON>>": _as_text(previous_candidates if previous_candidates is not None else []),
        "<<REGENERATION_REASON>>": _as_text(regeneration_reason),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_conv_candidate_feasibility_prompt(
    notice_and_criteria: Any,
    candidates: Any,
    retrieved_evidence: Any,
) -> str:
    """개발 전문가의 "후보별 실현 가능성 검토" 프롬프트를 조립한다."""
    card = get_persona_card("dev_expert")
    template = _read_text(IDEATION_CONV_CANDIDATE_FEASIBILITY_TEMPLATE)
    replacements = {
        "<<PERSONA_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<CANDIDATES_JSON>>": _as_text(candidates),
        "<<RETRIEVED_EVIDENCE_JSON>>": _as_text(retrieved_evidence),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def build_ideation_conv_candidate_selection_prompt(
    notice_and_criteria: Any,
    candidates: Any,
    user_message: str,
) -> str:
    """번호/제목 단순 선택이 아닌 결합·전문가 추천·모호한 답변을 해석하는 프롬프트를 조립한다.
    단순 선택은 코드가 결정적으로 처리하므로 이 프롬프트를 부르지 않는다(ideation_conv_discovery.py 참고)."""
    card = get_persona_card("ideation_facilitator")
    template = _read_text(IDEATION_CONV_CANDIDATE_SELECTION_TEMPLATE)
    replacements = {
        "<<FACILITATOR_BLOCK>>": render_persona_block(card),
        "<<NOTICE_AND_CRITERIA_JSON>>": _as_text(notice_and_criteria),
        "<<CANDIDATES_JSON>>": _as_text(candidates),
        "<<USER_MESSAGE>>": _as_text(user_message),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


if __name__ == "__main__":
    # 스모크 테스트: 실제 LLM 호출 없이 프롬프트 조립만 확인한다.
    demo_rubric = {
        "rubric_id": "RUBRIC-DEMO",
        "total_max_score": 30,
        "criteria": [
            {"criterion_id": "marketability", "criterion_name": "시장성", "max_score": 30, "required": True},
        ],
    }
    demo_submission = {"document_name": "데모.pdf", "text": "타깃 고객은 소규모 오프라인 매장이다."}
    demo_evidence = [{"source_id": "DOC-DEMO", "chunk_id": "CHUNK-1", "page": 1, "text": "..."}]

    reviewers = reviewer_personas_for_domain("startup")
    print("startup 위원 후보:", reviewers)

    rp = build_reviewer_prompt("business_strategy", demo_rubric, demo_submission, demo_evidence)
    cp = build_chair_prompt([], demo_rubric, demo_evidence)

    assert "<<" not in rp, "reviewer 프롬프트에 치환되지 않은 토큰이 있습니다"
    assert "<<" not in cp, "chair 프롬프트에 치환되지 않은 토큰이 있습니다"
    assert "사업전략 전문가" in rp
    assert "위원장" in cp
    print(f"reviewer_prompt 길이: {len(rp)}  / chair_prompt 길이: {len(cp)}")
    print("OK: 프롬프트 조립 및 토큰 치환 정상")

    # 용준/Claude(2026-07-20): ideation 프롬프트 빌더 스모크 테스트.
    demo_notice = {"notice_summary": "지역 소상공인 디지털전환 공모전", "criteria": ["실현가능성", "차별성"]}
    demo_idea = {"title": "동네 가게 챗봇", "description": "소상공인이 손님 문의에 자동으로 답하는 챗봇"}
    demo_round_context = {"round": 1, "previous_turn": None, "consensus_so_far": [], "unresolved_issues": []}

    tp_planning = build_ideation_turn_prompt("planning_expert", demo_notice, demo_idea, demo_evidence, demo_round_context)
    tp_dev = build_ideation_turn_prompt("dev_expert", demo_notice, demo_idea, demo_evidence, demo_round_context)
    fp = build_ideation_facilitator_prompt(demo_notice, demo_idea, 1, 3, [], [], [])
    sp = build_ideation_synthesis_prompt(demo_notice, demo_idea, [], [], [])

    assert "<<" not in tp_planning, "ideation 기획 전문가 프롬프트에 치환되지 않은 토큰이 있습니다"
    assert "<<" not in tp_dev, "ideation 개발 전문가 프롬프트에 치환되지 않은 토큰이 있습니다"
    assert "<<" not in fp, "ideation 진행자 프롬프트에 치환되지 않은 토큰이 있습니다"
    assert "<<" not in sp, "ideation 종합 프롬프트에 치환되지 않은 토큰이 있습니다"
    assert "당신은 AI Review Board의 기획 전문가입니다" in tp_planning
    assert "당신은 AI Review Board의 개발 전문가입니다" in tp_dev
    assert "당신은 AI Review Board의 개발 전문가입니다" not in tp_planning
    assert "당신은 AI Review Board의 기획 전문가입니다" not in tp_dev
    assert "회의 진행자" in fp
    print(f"ideation_turn(planning) 길이: {len(tp_planning)} / ideation_turn(dev) 길이: {len(tp_dev)}")
    print(f"ideation_facilitator 길이: {len(fp)} / ideation_synthesis 길이: {len(sp)}")
    print("OK: ideation 프롬프트 조립 및 토큰 치환 정상")
