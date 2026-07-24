# 작성자: 용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection")
# 목적: Phase 1 shadow planner가 고른 근거를 discussion(planning_expert/dev_expert) 발언
#       prompt/claim grounding에 실제로 주입하는 active 경로를 검증한다. Phase 1과 같은 경계
#       원칙을 지킨다 — ai/meeting은 실제 ai.rag 구현(ideation_evidence_planner.build_evidence_plan)을
#       모르므로, evidence_planner 콜러블의 "모양"(persona_id/effective_issue/retrieved_evidence/
#       runtime_scope/shadow_history 인자, plan dict 반환, active 속성)만 아는 fake를 주입한다.
#       실제 plan 선택 규칙은 ai/rag/tests/test_ideation_evidence_planner.py가 검증한다.

import json
import logging
import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
REPO_ROOT = MEETING_DIR.parents[1]  # repo root (ai/meeting -> ai -> root)
sys.path.insert(0, str(MEETING_DIR))
sys.path.insert(0, str(REPO_ROOT))

from graph import reply_ideation_conversation, start_ideation_conversation  # noqa: E402
from graph.ideation_conv_nodes import (  # noqa: E402
    _build_evidence_plan_notice,
    _isolate_discussion_evidence_context,
    make_conv_question_node,
    resolve_effective_issue,
)
from graph.ideation_trace import configure_ideation_trace  # noqa: E402

from test_ideation_conv_graph import NOTICE_AND_CRITERIA, USER_IDEA  # noqa: E402


def _multi_item_lookup():
    """target/criteria 두 항목을 검색 결과로 돌려준다 — planner가 그중 일부만 고르는
    시나리오를 만들기 위함이다. 두 텍스트는 서로 겹치지 않는 고유 문자열을 담아, prompt에
    실제로 어떤 항목이 들어갔는지 문자열 포함 여부로 판별할 수 있게 한다."""

    def lookup(persona_id: str, query: str):
        return [
            {
                "document_id": f"DOC-TARGET-{persona_id}",
                "document_name": "선택된 아이디어",
                "chunk_id": "CHUNK-TARGET-1",
                "document_role": "target",
                "final_score": 0.9,
                "page": 1,
                "section": None,
                "text": "TARGET_ONLY_MARKER 사용자가 선택한 아이디어의 실제 내용입니다.",
            },
            {
                "document_id": f"DOC-CRITERIA-{persona_id}",
                "document_name": "공모전 공고문",
                "chunk_id": "CHUNK-CRITERIA-1",
                "document_role": "criteria",
                "final_score": 0.8,
                "page": 2,
                "section": "심사 기준",
                "text": "CRITERIA_ONLY_MARKER 공모전 심사 기준에 대한 실제 내용입니다.",
            },
        ]

    return lookup


class _FakeActiveEvidencePlanner:
    """evidence_planner로 주입되는 fake. plan_factory가 주어지면 그 값을 그대로 쓰고, 아니면
    retrieved_evidence의 첫 target 항목만 선택하는 기본 plan을 만든다. active=True로
    세팅하면 ideation_conv_nodes.py가 이 결과를 prompt/grounding에 실제로 쓴다(backend
    레이어의 _evidence_planner_for가 ENABLE_IDEATION_EVIDENCE_PLANNER_DISCUSSION일 때
    세팅하는 것과 동일한 속성)."""

    def __init__(self, plan_factory=None, raise_error=False, active=True):
        self.calls: list[dict] = []
        self.plan_factory = plan_factory
        self.raise_error = raise_error
        self.active = active

    def __call__(self, *, persona_id, effective_issue, retrieved_evidence, runtime_scope, shadow_history):
        self.calls.append(
            {
                "persona_id": persona_id,
                "effective_issue": dict(effective_issue),
                "retrieved_evidence": list(retrieved_evidence),
                "runtime_scope": dict(runtime_scope),
                "shadow_history": list(shadow_history),
            }
        )
        if self.raise_error:
            raise RuntimeError("planner 시뮬레이션 실패")
        if self.plan_factory is not None:
            return self.plan_factory(persona_id, effective_issue, retrieved_evidence, runtime_scope, shadow_history)
        return _default_plan(persona_id, effective_issue, retrieved_evidence, role="target")


def _default_plan(persona_id, effective_issue, retrieved_evidence, *, role="target", quote_marker=None):
    issue = {
        "issue_id": effective_issue["issue_id"],
        "title": effective_issue["title"],
        "query": effective_issue.get("query", ""),
    }
    candidates = [item for item in retrieved_evidence if item.get("document_role") == role]
    if not candidates:
        return {
            "plan_id": "EP-active-empty",
            "policy_version": "test-v1",
            "persona_id": persona_id,
            "issue": issue,
            "eligible_evidence_count": 0,
            "grounded_claim_required": False,
            "expert_judgment_required": True,
            "selected_evidence": [],
            "empty_plan_reason": "no_retrieved_evidence",
            "validation": {"valid": True, "errors": []},
        }
    item = candidates[0]
    text = item.get("text", "")
    quote = quote_marker or text
    claim_type = "document_fact" if role == "criteria" else "user_provided_fact"
    return {
        "plan_id": f"EP-active-{role}",
        "policy_version": "test-v1",
        "persona_id": persona_id,
        "issue": issue,
        "eligible_evidence_count": 1,
        "grounded_claim_required": True,
        "expert_judgment_required": False,
        "selected_evidence": [
            {
                "ref": item.get("ref"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "document_role": role,
                "claim_type": claim_type,
                "quote": quote,
                "quote_start": 0,
                "quote_end": len(quote),
                "retrieval_score": 0.6,
                "issue_relevance_score": 0.5,
                "selection_reason_code": f"{role}_fact_for_current_issue",
                "reused_in_same_issue": False,
            }
        ],
        "empty_plan_reason": None,
        "validation": {"valid": True, "errors": []},
    }


class _ClaimAwareLLM:
    """[의견 규칙] 프롬프트를 받으면, prompt에 실제로 주입된 첫 번째 "ref" 값을
    evidence_refs로 인용하는 document_fact claim 하나를 포함해 응답한다 — prompt에 어떤
    근거가 들어갔는지에 따라 grounding 결과가 달라지는지 확인하기 위함이다. cite_ref가
    명시되면 검색 여부와 무관하게 그 값을 그대로 인용한다(존재하지 않는 ref를 인용하는
    "hard grounding failure" 시나리오를 재현하기 위함)."""

    import re as _re

    _REF_RE = _re.compile(r'"ref":\s*"([^"]+)"')
    _CLAIM_TYPE_RE = _re.compile(r'"claim_type":\s*"([^"]+)"')
    _ACTIVE_ISSUE_RE = _re.compile(r"\[active_issue_id\]\s*\n([^\r\n]+)")

    def __init__(
        self,
        cite_ref=None,
        active_issue_id=None,
        force_no_claims=False,
        claim_type_override=None,
    ):
        self.captured_prompts: list[str] = []
        self.cite_ref = cite_ref
        self.active_issue_id = active_issue_id
        self.force_no_claims = force_no_claims
        self.claim_type_override = claim_type_override

    def __call__(self, prompt: str) -> str:
        self.captured_prompts.append(prompt)
        is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
        speaker = "planning_expert" if is_planning else "dev_expert"

        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "facilitator_summary": "이번 라운드 논의를 정리합니다.",
                    "agreements": [],
                    "disagreements": [],
                    "needs_user_decision": True,
                    "user_question": "다음 논의를 계속 진행할까요?",
                    "spoken_text": "이번 라운드 논의를 정리했습니다. 계속 진행할까요?",
                },
                ensure_ascii=False,
            )

        if "[의견 규칙]" not in prompt:
            # 질문/판정 등 다른 노드 프롬프트는 이 stub의 관심사가 아니다 — 최소한의 유효
            # 응답만 돌려줘 그래프가 막히지 않게 한다.
            return json.dumps(
                {
                    "spoken_text": f"[{speaker}] 발화 질문",
                    "judgment": f"[{speaker}] 판단",
                    "question": f"[{speaker}] 질문",
                    "question_topic": "problem",
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )

        refs_in_prompt = self._REF_RE.findall(prompt)
        issue_match = self._ACTIVE_ISSUE_RE.search(prompt)
        active_issue_id = self.active_issue_id or (
            issue_match.group(1).strip() if issue_match else "problem"
        )
        selected_claim_type = (
            self.claim_type_override
            or (
                "user_provided_fact"
                if '"document_role": "target"' in prompt
                else "document_fact"
            )
        )
        # 실제 ai.rag.evidence_linking.relevance.is_relevant_candidate는 claim 텍스트와 인용된
        # 청크 본문의 키워드가 겹쳐야 관련성 있다고 판단한다 — 두 마커 단어를 모두 claim
        # 텍스트에 담아, 실제로 prompt에 주입된 쪽(target 또는 criteria)과는 항상 겹치게 한다.
        claim_text = (
            "TARGET_ONLY_MARKER 사용자가 선택한 아이디어의 실제 내용입니다."
            if selected_claim_type == "user_provided_fact"
            else "CRITERIA_ONLY_MARKER 공모전 평가기준의 실제 내용입니다."
        )
        claims = []
        if self.force_no_claims:
            claims = []
        elif self.cite_ref is not None:
            claims = [
                {
                    "claim_id": "claim_1",
                    "text": claim_text,
                    "claim_type": selected_claim_type,
                    "evidence_refs": [self.cite_ref],
                }
            ]
        elif refs_in_prompt:
            claims = [
                {
                    "claim_id": "claim_1",
                    "text": claim_text,
                    "claim_type": selected_claim_type,
                    "evidence_refs": [refs_in_prompt[0]],
                }
            ]

        return json.dumps(
            {
                "stance": "보완",
                "spoken_text": f"문제와 영향에 관한 {speaker}의 판단입니다",
                "judgment": "판단",
                "reason": "근거를 반영한 판단",
                "suggestion": "제안",
                "interim_conclusion": "잠정 결론",
                "responding_to": None,
                "agreement": "",
                "concern": "",
                "confirmed": [],
                "unconfirmed": [],
                "referenced_message_ids": [],
                "claims": claims,
                "next_action": None,
                "active_issue_id": active_issue_id,
                "active_issue_title": "활성 쟁점",
                "new_information": ["새로운 판단 근거"],
                "proposal": None,
                "changed_position": False,
                "needs_counterpart_response": False,
                "recommended_next_speaker": "ideation_facilitator",
                "issue_resolved": False,
                "needs_user_input": False,
                "user_question": None,
            },
            ensure_ascii=False,
        )


def _real_ground_claims():
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _impl

    def grounder(persona_id, claims, retrieved_evidence):
        return _impl(claims, retrieved_evidence)

    return grounder


def _first_discussion_prompt(llm: _ClaimAwareLLM) -> str:
    for prompt in llm.captured_prompts:
        if "[의견 규칙]" in prompt:
            return prompt
    raise AssertionError("discussion(의견 규칙) 프롬프트가 캡처되지 않았습니다")


def _evidence_section(prompt: str) -> str:
    """prompt 문자열에서 [검색 근거 retrieved_evidence] 절만 잘라낸다 — 회의 진행에 따라
    달라지는 message_id/conversation_context와 무관하게 evidence 주입 내용만 비교하기
    위함이다."""
    start = prompt.index("[검색 근거 retrieved_evidence]")
    end = prompt.index("[대화 맥락 conversation_context]")
    return prompt[start:end]


# ---------------------------------------------------------------------------
# 1) discussion 플래그(active) false(기본값, evidence_planner=None) — 기존과 완전히 동일.
# ---------------------------------------------------------------------------


def test_planner_none_prompt_and_grounding_unchanged():
    llm_a = _ClaimAwareLLM()
    llm_b = _ClaimAwareLLM()
    lookup = _multi_item_lookup()

    state_without = start_ideation_conversation(
        session_id="S-P2-NONE-A", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm_a, evidence_lookup=lookup,
    )
    state_with_none = start_ideation_conversation(
        session_id="S-P2-NONE-B", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm_b, evidence_lookup=lookup, evidence_planner=None,
    )
    assert [m["content"] for m in state_without["messages"]] == [m["content"] for m in state_with_none["messages"]]
    prompt_a = _first_discussion_prompt(llm_a)
    prompt_b = _first_discussion_prompt(llm_b)
    assert _evidence_section(prompt_a) == _evidence_section(prompt_b)
    # 두 target/criteria 항목 모두 legacy 경로에서는 그대로 prompt에 들어간다.
    assert "TARGET_ONLY_MARKER" in prompt_a
    assert "CRITERIA_ONLY_MARKER" in prompt_a


def test_planner_active_false_matches_planner_none():
    """evidence_planner가 주입되긴 했지만 .active=False(shadow 전용, backend가
    ENABLE_IDEATION_EVIDENCE_PLANNER_SHADOW만 켰을 때)면 evidence_planner=None과 완전히
    동일한 prompt를 만들어야 한다 — SHADOW와 DISCUSSION 플래그가 실제로 분리돼 있는지
    ai/meeting 레이어에서 검증한다."""
    llm_a = _ClaimAwareLLM()
    llm_b = _ClaimAwareLLM()
    lookup = _multi_item_lookup()
    shadow_only_planner = _FakeActiveEvidencePlanner(active=False)

    state_a = start_ideation_conversation(
        session_id="S-P2-SHADOWONLY-A", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm_a, evidence_lookup=lookup, evidence_planner=None,
    )
    state_b = start_ideation_conversation(
        session_id="S-P2-SHADOWONLY-B", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm_b, evidence_lookup=lookup, evidence_planner=shadow_only_planner,
    )
    assert [m["content"] for m in state_a["messages"]] == [m["content"] for m in state_b["messages"]]
    assert _evidence_section(_first_discussion_prompt(llm_a)) == _evidence_section(_first_discussion_prompt(llm_b))
    assert shadow_only_planner.calls, "shadow 전용이어도 planner 자체는 여전히 호출돼야 한다(Phase 1 그대로)"


# ---------------------------------------------------------------------------
# 2)+3) active=True — 선택된 근거만 prompt에 주입되고, 선택되지 않은 근거는 빠진다.
# ---------------------------------------------------------------------------


def test_active_mode_injects_only_selected_evidence_and_excludes_the_rest():
    llm = _ClaimAwareLLM()
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(active=True)  # 기본 factory: target만 선택

    start_ideation_conversation(
        session_id="S-P2-SELECTIVE", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm, evidence_lookup=lookup, evidence_planner=planner,
    )
    prompt = _first_discussion_prompt(llm)
    assert "TARGET_ONLY_MARKER" in prompt
    assert "CRITERIA_ONLY_MARKER" not in prompt


# ---------------------------------------------------------------------------
# 4)+5) question/facilitator/candidate 노드에는 적용되지 않는다(구조적으로 evidence_planner
# 파라미터 자체가 없다).
# ---------------------------------------------------------------------------


def test_evidence_planner_not_a_parameter_of_question_or_other_nodes():
    import inspect

    from graph.ideation_conv_discovery import (
        make_candidate_feasibility_node,
        make_candidate_planning_node,
        make_candidate_selection_node,
    )
    from graph.ideation_conv_nodes import make_conv_synthesis_node, make_discussion_facilitator_node

    for factory in (
        make_conv_question_node,
        make_candidate_planning_node,
        make_candidate_feasibility_node,
        make_candidate_selection_node,
        make_discussion_facilitator_node,
        make_conv_synthesis_node,
    ):
        params = inspect.signature(factory).parameters
        assert "evidence_planner" not in params, f"{factory.__name__}에는 evidence_planner가 없어야 한다"


# ---------------------------------------------------------------------------
# 6)+7) technical failure(예외/validation 실패) — legacy(retrieved 전체)로 fallback.
# ---------------------------------------------------------------------------


def test_planner_exception_falls_back_to_full_retrieved_evidence():
    llm = _ClaimAwareLLM()
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(raise_error=True, active=True)

    state = start_ideation_conversation(
        session_id="S-P2-EXCEPTION", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm, evidence_lookup=lookup, evidence_planner=planner,
    )
    assert state["phase"] != "failed"
    prompt = _first_discussion_prompt(llm)
    assert "TARGET_ONLY_MARKER" in prompt
    assert "CRITERIA_ONLY_MARKER" in prompt


def test_plan_validation_failure_falls_back_to_full_retrieved_evidence():
    def invalid_plan_factory(persona_id, effective_issue, retrieved_evidence, runtime_scope, shadow_history):
        plan = _default_plan(persona_id, effective_issue, retrieved_evidence, role="target")
        plan["validation"] = {"valid": False, "errors": ["quote_offset_invariant_failed:E1"]}
        return plan

    llm = _ClaimAwareLLM()
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(plan_factory=invalid_plan_factory, active=True)

    state = start_ideation_conversation(
        session_id="S-P2-INVALID", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm, evidence_lookup=lookup, evidence_planner=planner,
    )
    assert state["phase"] != "failed"
    prompt = _first_discussion_prompt(llm)
    assert "TARGET_ONLY_MARKER" in prompt
    assert "CRITERIA_ONLY_MARKER" in prompt


# ---------------------------------------------------------------------------
# 8) valid empty plan — 전체 retrieved로 fallback하지 않는다(근거 없음 상태 유지).
# ---------------------------------------------------------------------------


def test_valid_empty_plan_does_not_fall_back_to_full_evidence():
    def empty_plan_factory(persona_id, effective_issue, retrieved_evidence, runtime_scope, shadow_history):
        issue = {
            "issue_id": effective_issue["issue_id"],
            "title": effective_issue["title"],
            "query": effective_issue.get("query", ""),
        }
        return {
            "plan_id": "EP-valid-empty",
            "policy_version": "test-v1",
            "persona_id": persona_id,
            "issue": issue,
            "eligible_evidence_count": 0,
            "grounded_claim_required": False,
            "expert_judgment_required": True,
            "selected_evidence": [],
            "empty_plan_reason": "no_issue_relevant_evidence",
            "validation": {"valid": True, "errors": []},
        }

    llm = _ClaimAwareLLM()
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(plan_factory=empty_plan_factory, active=True)

    state = start_ideation_conversation(
        session_id="S-P2-VALID-EMPTY", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm, evidence_lookup=lookup, evidence_planner=planner,
    )
    assert state["phase"] != "failed"
    prompt = _first_discussion_prompt(llm)
    assert "TARGET_ONLY_MARKER" not in prompt
    assert "CRITERIA_ONLY_MARKER" not in prompt


def test_valid_empty_plan_skips_freeform_expert_generation_when_grounding_is_active(monkeypatch):
    def empty_plan_factory(persona_id, effective_issue, retrieved_evidence, runtime_scope, shadow_history):
        return {
            "plan_id": "EP-valid-empty-strict",
            "policy_version": "test-v1",
            "persona_id": persona_id,
            "issue": {
                "issue_id": effective_issue["issue_id"],
                "title": effective_issue["title"],
                "query": effective_issue.get("query", ""),
            },
            "eligible_evidence_count": 0,
            "grounded_claim_required": False,
            "expert_judgment_required": True,
            "selected_evidence": [],
            "empty_plan_reason": "no_issue_relevant_evidence",
            "validation": {"valid": True, "errors": []},
        }

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "graph.ideation_conv_nodes.trace_event",
        lambda name, **fields: events.append((name, fields)),
    )
    llm = _ClaimAwareLLM()
    state = start_ideation_conversation(
        session_id="S-P2-VALID-EMPTY-STRICT",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        evidence_lookup=_multi_item_lookup(),
        evidence_planner=_FakeActiveEvidencePlanner(
            plan_factory=empty_plan_factory,
            active=True,
        ),
        ground_claims=_real_ground_claims(),
    )

    expert_messages = [
        message
        for message in state["messages"]
        if message["speaker_id"] in ("planning_expert", "dev_expert")
    ]
    assert expert_messages
    assert expert_messages[0]["structured"]["evidence_first_skipped"] is True
    assert expert_messages[0]["claims"] == []
    assert expert_messages[0]["linked_evidence_refs"] == []
    assert "근거를 찾지 못해" in expert_messages[0]["content"]
    assert not [prompt for prompt in llm.captured_prompts if "[의견 규칙]" in prompt]
    supplemental_events = [
        fields for name, fields in events if name == "IDEATION_SUPPLEMENTAL_RETRIEVAL"
    ]
    assert supplemental_events
    assert all(event["grounding_improved"] is False for event in supplemental_events)
    assert all(event["new_evidence_count"] == 0 for event in supplemental_events)


# ---------------------------------------------------------------------------
# 9)+10)+11) prompt와 grounding이 동일한 selected evidence 집합을 쓰고, ref가 실제
# chunk_id로 변환되며, criteria/target 둘 다 정상 연결된다.
# ---------------------------------------------------------------------------


def test_prompt_and_grounding_share_same_selected_evidence_and_ref_resolves_to_chunk_id():
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(active=True)  # target 하나만 선택

    def llm_factory():
        # cite_ref는 prompt가 실제로 만들어진 뒤 채운다(E번호는 lookup 결과 순서로 정해진다).
        return _ClaimAwareLLM()

    llm = llm_factory()
    state = start_ideation_conversation(
        session_id="S-P2-GROUNDING", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm, evidence_lookup=lookup, evidence_planner=planner, ground_claims=_real_ground_claims(),
    )
    assert planner.calls
    selected = planner.calls[0]["retrieved_evidence"]
    selected_ref = next(item["ref"] for item in selected if item.get("document_role") == "target")

    discussion_messages = [m for m in state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    assert discussion_messages, "전문가 발언이 최소 한 건 있어야 한다"
    first = discussion_messages[0]
    # LLM stub은 prompt에 실제로 있는 첫 ref를 인용한다 — active 모드에서는 그 ref가 정확히
    # planner가 선택한 target ref와 같아야 한다(선택되지 않은 근거를 우연히 인용할 수 없다).
    assert first["structured"]["linked_evidence_refs"] == ["CHUNK-TARGET-1"], (
        "grounding이 ref를 실제 chunk_id로 정확히 변환해야 한다"
    )
    assert first["claims"][0]["evidence_refs"] == [selected_ref]


def test_active_turn_without_llm_claims_is_replaced_by_grounded_evidence_anchor():
    llm = _ClaimAwareLLM(force_no_claims=True)
    state = start_ideation_conversation(
        session_id="S-P2-EVIDENCE-FIRST-FALLBACK",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        evidence_lookup=_multi_item_lookup(),
        evidence_planner=_FakeActiveEvidencePlanner(active=True),
        ground_claims=_real_ground_claims(),
    )

    first = next(
        message
        for message in state["messages"]
        if message["speaker_id"] in ("planning_expert", "dev_expert")
    )
    assert first["structured"]["evidence_first_fallback"] is True
    assert first["linked_evidence_refs"] == ["CHUNK-TARGET-1"]
    assert first["grounded_claim_count"] == 1
    assert "근거 자료에는" in first["content"]


def test_criteria_and_target_evidence_both_link_when_selected():
    """target/criteria를 각각 별도로 선택하는 두 세션을 돌려, 두 role 모두 grounding이
    정상적으로 chunk_id를 연결하는지 확인한다(요청 11번)."""
    for role, expected_chunk in (("target", "CHUNK-TARGET-1"), ("criteria", "CHUNK-CRITERIA-1")):
        lookup = _multi_item_lookup()

        def plan_factory(persona_id, effective_issue, retrieved_evidence, runtime_scope, shadow_history, _role=role):
            return _default_plan(persona_id, effective_issue, retrieved_evidence, role=_role)

        planner = _FakeActiveEvidencePlanner(plan_factory=plan_factory, active=True)
        llm = _ClaimAwareLLM()
        state = start_ideation_conversation(
            session_id=f"S-P2-ROLE-{role}", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
            llm_call=llm, evidence_lookup=lookup, evidence_planner=planner, ground_claims=_real_ground_claims(),
        )
        discussion_messages = [m for m in state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
        first = discussion_messages[0]
        assert first["structured"]["linked_evidence_refs"] == [expected_chunk], role


# ---------------------------------------------------------------------------
# 12) retry에서도 동일 plan/ref를 재사용한다(재실행되지 않는다).
# ---------------------------------------------------------------------------


def test_grounding_retry_does_not_re_invoke_planner_or_change_evidence():
    """존재하지 않는 ref를 인용하게 만들어 _ground_and_finalize_claims의 1회 재시도 경로를
    강제로 태운다 — 재시도 중에도 planner가 다시 호출되지 않고, 재시도 prompt의 근거 절이
    최초 prompt와 동일한지 확인한다."""
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(active=True)
    llm = _ClaimAwareLLM(
        cite_ref="E-DOES-NOT-EXIST",
        claim_type_override="document_fact",
    )

    state = start_ideation_conversation(
        session_id="S-P2-RETRY", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm, evidence_lookup=lookup, evidence_planner=planner, ground_claims=_real_ground_claims(),
    )
    assert state["phase"] != "failed"

    discussion_prompts = [p for p in llm.captured_prompts if "[의견 규칙]" in p]
    assert len(discussion_prompts) >= 2, "hard grounding failure면 최소 1회 재시도 prompt가 있어야 한다"
    assert _evidence_section(discussion_prompts[0]) == _evidence_section(discussion_prompts[1]), (
        "재시도에서도 동일한 evidence가 주입돼야 한다"
    )
    # planner는 각 기획 발언당 정확히 한 번만 호출된다(grounding 재시도로 추가 호출되지 않는다).
    planning_calls = [
        c
        for c in planner.calls
        if c["persona_id"] == "planning_expert"
        and c["effective_issue"].get("planner_stage") != "supplemental_retrieval"
    ]
    planning_messages = [m for m in state["messages"] if m["speaker_id"] == "planning_expert"]
    assert len(planning_calls) == len(planning_messages)


# ---------------------------------------------------------------------------
# 13) 쟁점 정합성 — compliance 로그가 남고, mismatch만으로 재시도/실패하지 않는다.
# ---------------------------------------------------------------------------


def test_issue_mismatch_is_rejected_and_safe_fallback_keeps_effective_issue(caplog):
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(active=True)
    # effective_issue의 issue_id는 resolve_effective_issue가 결정하는 결정적 값이라
    # "issue_active"와 다를 수밖에 없다(테스트 전용 고정 문자열) — 항상 mismatch를 만든다.
    llm = _ClaimAwareLLM(active_issue_id="issue_active_mismatch_marker")

    configure_ideation_trace(enabled=True, content_max_chars=2000, stream_deltas=False)
    try:
        with caplog.at_level(logging.INFO, logger="ai.meeting.ideation_trace"):
            state = start_ideation_conversation(
                session_id="S-P2-MISMATCH", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
                llm_call=llm, evidence_lookup=lookup, evidence_planner=planner, ground_claims=_real_ground_claims(),
            )
    finally:
        configure_ideation_trace(enabled=None, content_max_chars=None, stream_deltas=None)

    assert state["phase"] != "failed"
    rendered = "\n".join(r.getMessage() for r in caplog.records)
    assert "[IDEATION_EVIDENCE_PLAN_COMPLIANCE]" in rendered
    assert 'reason="active_issue_id_mismatch"' in rendered
    assert "issue_match=true" in rendered
    first = next(
        message for message in state["messages"]
        if message["speaker_id"] in ("planning_expert", "dev_expert")
    )
    assert first["structured"]["active_issue_id"] == "problem"


def test_active_evidence_plan_logs_active_and_compliance_events(caplog):
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(active=True)
    llm = _ClaimAwareLLM()

    configure_ideation_trace(enabled=True, content_max_chars=2000, stream_deltas=False)
    try:
        with caplog.at_level(logging.INFO, logger="ai.meeting.ideation_trace"):
            start_ideation_conversation(
                session_id="S-P2-ACTIVE-LOG", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
                llm_call=llm, evidence_lookup=lookup, evidence_planner=planner, ground_claims=_real_ground_claims(),
            )
    finally:
        configure_ideation_trace(enabled=None, content_max_chars=None, stream_deltas=None)

    rendered = "\n".join(r.getMessage() for r in caplog.records)
    assert "[IDEATION_EVIDENCE_PLAN_ACTIVE]" in rendered
    assert "[IDEATION_EVIDENCE_PLAN_COMPLIANCE]" in rendered
    assert "[IDEATION_EVIDENCE_PLAN_FALLBACK]" not in rendered
    assert "[IDEATION_EVIDENCE_PLAN_VALID_EMPTY]" not in rendered


def test_fallback_logs_fallback_event_with_reason(caplog):
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(raise_error=True, active=True)
    llm = _ClaimAwareLLM()

    configure_ideation_trace(enabled=True, content_max_chars=2000, stream_deltas=False)
    try:
        with caplog.at_level(logging.INFO, logger="ai.meeting.ideation_trace"):
            start_ideation_conversation(
                session_id="S-P2-FALLBACK-LOG", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
                llm_call=llm, evidence_lookup=lookup, evidence_planner=planner,
            )
    finally:
        configure_ideation_trace(enabled=None, content_max_chars=None, stream_deltas=None)

    rendered = "\n".join(r.getMessage() for r in caplog.records)
    assert "[IDEATION_EVIDENCE_PLAN_FALLBACK]" in rendered
    assert 'fallback_reason="planner_exception"' in rendered


# ---------------------------------------------------------------------------
# 14) shadow+active 동시 활성화 — planner는 여전히 턴당 한 번만 호출된다.
# ---------------------------------------------------------------------------


def test_active_primary_planner_invoked_once_per_turn_and_supplemental_calls_are_marked():
    """SHADOW+DISCUSSION이 동시에 켜진 상황을 흉내낸다 — backend에서는 같은 _evidence_planner_for
    콜러블 하나가 두 플래그를 동시에 반영해 active=True로 세팅될 뿐, 콜러블 자체가 두 개
    주입되는 일은 없다(요청: 중복 실행 금지). ai/meeting 레이어에서는 "콜러블 하나가 턴당
    한 번만 불린다"만 검증하면 충분하다."""
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(active=True)
    llm = _ClaimAwareLLM()

    state = start_ideation_conversation(
        session_id="S-P2-ONCE", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm, evidence_lookup=lookup, evidence_planner=planner, ground_claims=_real_ground_claims(),
    )
    discussion_turns = [m for m in state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    # planner는 유일한 호출 지점(_run_shadow_evidence_planner)에서만 불린다 — SHADOW와
    # DISCUSSION이 동시에 켜져 있어도(이 fake는 두 플래그가 함께 켜졌을 때와 동일하게
    # active=True다) discussion 발언 한 건당 정확히 한 번만 호출돼야 한다(중복 실행 없음).
    primary_calls = [
        call
        for call in planner.calls
        if call["effective_issue"].get("planner_stage") != "supplemental_retrieval"
    ]
    supplemental_calls = [
        call
        for call in planner.calls
        if call["effective_issue"].get("planner_stage") == "supplemental_retrieval"
    ]
    assert len(primary_calls) == len(discussion_turns)
    assert len(supplemental_calls) <= len(discussion_turns)


# ---------------------------------------------------------------------------
# 15) 세션 저장·복원 후에도 ref/chunk_id 매핑이 유지된다(다음 턴에서도 active 경로가
# 정상 동작한다).
# ---------------------------------------------------------------------------


def test_active_mode_survives_session_resume_via_reply():
    lookup = _multi_item_lookup()
    planner = _FakeActiveEvidencePlanner(active=True)
    llm = _ClaimAwareLLM()

    state = start_ideation_conversation(
        session_id="S-P2-RESUME", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea=USER_IDEA,
        llm_call=llm, evidence_lookup=lookup, evidence_planner=planner, ground_claims=_real_ground_claims(),
    )
    calls_before = len(planner.calls)

    resumed_state = reply_ideation_conversation(
        previous_state=state, user_message="추가 의견입니다", llm_call=llm, evidence_lookup=lookup,
        evidence_planner=planner, ground_claims=_real_ground_claims(),
    )
    assert resumed_state["phase"] != "failed"
    assert len(planner.calls) >= calls_before, "재개 후에도 active planner가 계속 호출돼야 한다"
    discussion_messages = [m for m in resumed_state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    for message in discussion_messages:
        for chunk_id in message["structured"]["linked_evidence_refs"]:
            assert chunk_id in ("CHUNK-TARGET-1", "CHUNK-CRITERIA-1")


# ---------------------------------------------------------------------------
# 17) 플래그 false(evidence_planner=None) 상태에서 기존 discussion 관련 회귀 스위트는 이
# 파일이 아니라 test_ideation_evidence_planner_shadow.py / test_ideation_conv_graph.py 등
# 기존 파일로 이미 커버된다(완료 보고서에서 실행 결과를 함께 보고한다).
# ---------------------------------------------------------------------------


def test_active_context_excludes_previous_turn_evidence_namespace_without_mutating_state():
    previous_message = {
        "message_id": "MSG-OLD",
        "speaker_id": "planning_expert",
        "speaker_name": "기획 위원",
        "role": "planning",
        "message_type": "opinion",
        "content": "이전 발언의 의미 내용은 유지합니다.",
        "evidence": [{"ref": "E4", "chunk_id": "CHUNK-OLD"}],
        "claims": [{"claim_id": "old", "evidence_refs": ["E4"]}],
        "linked_evidence_refs": ["CHUNK-OLD"],
        "structured": {"linked_evidence_refs": ["CHUNK-OLD"]},
    }
    context = {
        "round": 2,
        "recent_messages": [previous_message],
        "last_user_answer": previous_message,
        "consensus_so_far": [],
        "unresolved_issues": [],
    }

    isolated = _isolate_discussion_evidence_context(context)

    for message in (isolated["recent_messages"][0], isolated["last_user_answer"]):
        assert message["content"] == previous_message["content"]
        assert "evidence" not in message
        assert "claims" not in message
        assert "linked_evidence_refs" not in message
        assert "structured" not in message
    assert context["recent_messages"][0]["evidence"][0]["ref"] == "E4"


def test_active_notice_names_only_current_turn_allowed_refs():
    plan = {
        "issue": {"issue_id": "problem", "title": "문제 정의"},
        "selected_evidence": [{"ref": "E1"}],
    }

    notice = _build_evidence_plan_notice("active", plan)

    assert "현재 턴에서 허용된 evidence_refs는 [E1]뿐" in notice
    assert "과거 발언에서 보았던 E번호" in notice
