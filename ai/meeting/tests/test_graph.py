# 작성자: 경이
# 목적: LangGraph 회의 그래프(M4) 검증 — rubric 변환, raw->v2 위원/위원장 변환,
#       전체 그래프 실행(위원 병렬 → 점수 → 위원장 종합)이 review_output.schema.json
#       v2 계약을 만족하는 결과를 만드는지 확인한다. 실제 LLM 대신 고정 응답을 돌려주는
#       stub을 쓴다.
# import: 표준 라이브러리 json/pathlib, pytest, jsonschema; ai/meeting/graph 패키지.

import json
import sys
from pathlib import Path

import jsonschema

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
REPO_ROOT = MEETING_DIR.parents[1]
sys.path.insert(0, str(MEETING_DIR))

from graph import (  # noqa: E402
    assemble_meeting_graph,
    build_routing,
    build_rubric,
    initial_state,
    make_openai_llm_call,
    rerun_reviewer,
    run_meeting,
)
from graph.transform import raw_chair_to_v2, raw_reviewer_to_v2  # noqa: E402
from graph.evidence import EvidencePool  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "contracts" / "schemas" / "review_output.schema.json"
COMPETITION_MAPPING_PATH = MEETING_DIR / "personas" / "rubric_mapping_competition.json"


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate_against_defs(instance: dict, def_name: str) -> None:
    schema = _load_schema()
    sub_schema = {**schema["$defs"][def_name], "$defs": schema["$defs"]}
    jsonschema.Draft202012Validator(sub_schema).validate(instance)


# ---------------------------------------------------------------------------
# rubric.py
# ---------------------------------------------------------------------------


def test_build_rubric_from_competition_mapping_matches_v2_rubric_schema():
    mapping = json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))
    rubric = build_rubric(mapping)
    _validate_against_defs(rubric, "rubric")
    assert rubric["total_max_score"] == 100
    assert all(c["required"] is True for c in rubric["criteria"])


def test_build_routing_maps_every_criterion_to_primary_persona():
    mapping = json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))
    routing = build_routing(mapping)
    assert routing["creativity_appropriateness"]["primary"] == "planning_expert"
    assert routing["feasibility"]["primary"] == "dev_expert"
    assert set(routing) == {c["criterion_id"] for c in mapping["rubric"]}


# ---------------------------------------------------------------------------
# transform.py (raw -> v2)
# ---------------------------------------------------------------------------

_RAW_REVIEWER = {
    "review_id": "REV-TEST-001",
    "meeting_id": "MTG-TEST-001",
    "persona_id": "business_strategy",
    "persona_name": "사업전략 전문가",
    "review_round": 1,
    "review_summary": "타깃 고객은 명확하나 시장 검증 근거가 부족하다.",
    "review_items": [
        {
            "criterion_id": "marketability",
            "criterion_name": "시장성",
            "max_score": 30,
            "score_recommendation": 20,
            "judgment": "needs_improvement",
            "confidence": "high",
            "strengths": ["타깃 고객이 구체적이다."],
            "weaknesses": ["고객 검증 자료가 없다."],
            "evidence_refs": [
                {
                    "source_id": "document_001",
                    "chunk_id": "chunk_001",
                    "source_type": "submission",
                    "page": 3,
                    "quote": "초기 창업자는 지원사업 정보를 찾는 데 많은 시간을 쓴다.",
                    "relevance": "문제 정의 근거",
                }
            ],
            "improvement_actions": ["고객 인터뷰 결과를 추가한다."],
        },
        {
            "criterion_id": "differentiation",
            "criterion_name": "차별성",
            "max_score": 20,
            "score_recommendation": None,
            "judgment": "not_applicable",
            "confidence": "medium",
            "strengths": [],
            "weaknesses": [],
            "evidence_refs": [],
            "improvement_actions": [],
        },
    ],
    "cross_reviews": [],
    "priority_actions": [
        {"priority": 1, "criterion_id": "marketability", "action": "고객 인터뷰 보강", "reason": "근거 부족"}
    ],
    "out_of_scope": [
        {"topic": "AI 모델 구현", "reason": "기술 검토 필요", "handoff_persona_id": "technical_feasibility"}
    ],
}


def test_raw_reviewer_to_v2_excludes_not_applicable_and_matches_schema():
    pool = EvidencePool("business_strategy")
    result = raw_reviewer_to_v2(_RAW_REVIEWER, pool)
    _validate_against_defs(result, "reviewerResult")

    scored_ids = {s["criterion_id"] for s in result["rubric_scores"]}
    assert scored_ids == {"marketability"}, "not_applicable 항목은 채점 결과에서 빠져야 한다"
    assert result["rubric_scores"][0]["judgment"] == "needs_improvement"
    assert result["rubric_scores"][0]["evidence_ids"] == ["EV-business_strategy-001"]
    assert result["out_of_scope"][0]["topic"] == "AI 모델 구현"


_RAW_CHAIR = {
    "chair_id": "review_chair",
    "overall_assessment": "타깃 고객은 명확하나 시장 검증이 부족하다.",
    "consensus": ["문제 정의는 명확하다."],
    "disagreements": [],
    "top_strengths": ["타깃 고객이 구체적이다."],
    "top_risks": ["시장 검증 근거 부족"],
    "final_priority_actions": [
        {
            "priority": 1,
            "title": "고객 검증 자료 보강",
            "target": "시장성 섹션",
            "reason": "고객 인터뷰 결과가 없다.",
            "action": "고객 인터뷰 10건 이상을 추가한다.",
            "related_criteria": ["marketability"],
            "evidence_ids": ["EV-business_strategy-001"],
        }
    ],
    "final_decision": None,
    "decision_note": "합격 기준이 입력되지 않아 판정하지 않는다.",
}


def test_raw_chair_to_v2_matches_schema():
    chair_summary, top_revisions = raw_chair_to_v2(_RAW_CHAIR)
    _validate_against_defs(chair_summary, "chairSummary")
    for revision in top_revisions:
        _validate_against_defs(revision, "revision")
    assert top_revisions[0]["title"] == "고객 검증 자료 보강"


# ---------------------------------------------------------------------------
# 그래프 전체 실행 (stub LLM)
# ---------------------------------------------------------------------------

_TECHNICAL_RAW = {
    "review_id": "REV-TEST-002",
    "meeting_id": "MTG-TEST-001",
    "persona_id": "technical_feasibility",
    "persona_name": "기술·실현가능성 전문가",
    "review_round": 1,
    "review_summary": "일정이 촉박해 위험이 크다.",
    "review_items": [
        {
            "criterion_id": "technical_feasibility",
            "criterion_name": "기술·실현 가능성",
            "max_score": 30,
            "score_recommendation": 18,
            "judgment": "needs_improvement",
            "confidence": "high",
            "strengths": ["핵심 입력 데이터가 명시되어 있다."],
            "weaknesses": ["데이터 확보 방법이 없다."],
            "evidence_refs": [],
            "improvement_actions": ["데이터 확보 계획을 추가한다."],
        },
        {
            "criterion_id": "execution_plan",
            "criterion_name": "실행계획",
            "max_score": 20,
            "score_recommendation": 12,
            "judgment": "critical_risk",
            "confidence": "high",
            "strengths": ["개발 기간이 명시되어 있다."],
            "weaknesses": ["병렬 작업이 과도하다."],
            "evidence_refs": [],
            "improvement_actions": ["MVP 범위를 축소한다."],
        },
    ],
    "cross_reviews": [
        {
            "target_persona_id": "business_strategy",
            "relation": "supplement",
            "target_criterion_id": "marketability",
            "comment": "시장 검증과 함께 기술 검증 계획도 필요하다.",
            "evidence_refs": [],
        }
    ],
    "priority_actions": [],
    "out_of_scope": [],
}

_RUBRIC = {
    "rubric_id": "RUBRIC-TEST",
    "total_max_score": 100,
    "criteria": [
        {"criterion_id": "marketability", "criterion_name": "시장성", "max_score": 30, "required": True},
        {
            "criterion_id": "technical_feasibility",
            "criterion_name": "기술·실현 가능성",
            "max_score": 30,
            "required": True,
        },
        {"criterion_id": "differentiation", "criterion_name": "차별성", "max_score": 20, "required": False},
        {"criterion_id": "execution_plan", "criterion_name": "실행계획", "max_score": 20, "required": True},
    ],
}

_RETRIEVED_EVIDENCE = [
    {
        "chunk_id": "chunk_001",
        "document_name": "사업계획서.pdf",
        "page": 3,
        "section": "문제 정의",
        "text": "초기 창업자는 지원사업 정보를 찾는 데 평균적으로 많은 시간을 사용한다.",
        "score": 0.87,
    }
]


def _make_stub_llm(raw_by_marker: dict[str, dict]):
    def stub_llm(prompt: str) -> str:
        for marker, raw in raw_by_marker.items():
            if marker in prompt:
                return json.dumps(raw, ensure_ascii=False)
        raise AssertionError(f"stub_llm: 프롬프트에서 마커를 찾지 못함(markers={list(raw_by_marker)})")

    return stub_llm


def test_full_meeting_graph_produces_v2_compliant_result():
    committee = ["business_strategy", "technical_feasibility"]
    llm_call = _make_stub_llm(
        {
            "사업전략 전문가입니다": _RAW_REVIEWER,
            "기술·실현가능성 전문가입니다": _TECHNICAL_RAW,
            "위원장(review_chair)입니다": _RAW_CHAIR,
        }
    )
    graph = assemble_meeting_graph(committee, llm_call)

    state = initial_state(
        meeting_id="MTG-TEST-001",
        domain="startup",
        rubric=_RUBRIC,
        submission={"document_name": "테스트 사업계획서.pdf", "text": "..."},
        committee=committee,
        retrieved_evidence=_RETRIEVED_EVIDENCE,
    )
    result = graph.invoke(state)

    # MTG-001: 두 위원 결과가 모두, 서로 덮어쓰지 않고 남아 있어야 한다
    assert set(result["reviewer_results"]) == {"business_strategy", "technical_feasibility"}
    # MTG-006: 마지막 노드(chair)까지 지나 완료 단계여야 한다
    assert result["stage"] == "완료"
    # MTG-003: 점수 엔진(M2) 결과가 결정론적으로 채워졌는지
    assert result["score_result"]["total_score"] == 20 + 18 + 12  # differentiation은 not_applicable로 미채점
    # MTG-002/004
    assert result["chair_summary"]["chair_id"] == "review_chair"
    assert 1 <= len(result["top_revisions"]) <= 5

    document = {
        "schema_version": "2.0.0",
        "meeting_id": result["meeting_id"],
        "project_id": "PRJ-TEST-001",
        "document_id": "DOC-TEST-001",
        "title": "테스트 회의",
        "status": "completed",
        "domain": result["domain"],
        "rubric": result["rubric"],
        "reviewer_results": list(result["reviewer_results"].values()),
        "score_result": result["score_result"],
        "chair_summary": result["chair_summary"],
        "top_revisions": result["top_revisions"],
        "evidence": result["evidence"],
        "media_script": [],
    }
    jsonschema.Draft202012Validator(_load_schema()).validate(document)


# ---------------------------------------------------------------------------
# run.py — 실제 rubric_mapping(가은 PER-001)으로 엔트리포인트 전체 실행
# ---------------------------------------------------------------------------

_COMPETITION_PERSONA_NAMES = {
    "planning_expert": "기획 전문가",
    "dev_expert": "개발 전문가",
}


def _make_raw_reviewer(persona_id: str, persona_name: str, owned_criteria: list[dict]) -> dict:
    return {
        "review_id": f"REV-{persona_id}",
        "meeting_id": "MTG-RUN-TEST-001",
        "persona_id": persona_id,
        "persona_name": persona_name,
        "review_round": 1,
        "review_summary": f"{persona_name} 검토 요약",
        "review_items": [
            {
                "criterion_id": c["criterion_id"],
                "criterion_name": c["criterion_name"],
                "max_score": c["max_score"],
                "score_recommendation": c["max_score"] - 5,
                "judgment": "strong",
                "confidence": "high",
                "strengths": ["근거가 충분하다."],
                "weaknesses": [],
                "evidence_refs": [],
                "improvement_actions": [],
            }
            for c in owned_criteria
        ],
        "cross_reviews": [],
        "priority_actions": [],
        "out_of_scope": [],
    }


_RAW_CHAIR_FOR_RUN = {
    "chair_id": "review_chair",
    "overall_assessment": "전반적으로 양호하다.",
    "consensus": ["아이디어가 참신하다."],
    "disagreements": [],
    "top_strengths": ["아이디어가 참신하다."],
    "top_risks": ["실현 가능성 검증이 더 필요하다."],
    "final_priority_actions": [
        {
            "priority": 1,
            "title": "실현 가능성 보강",
            "target": "실현 가능성 섹션",
            "reason": "구체적 실행 계획이 부족하다.",
            "action": "단계별 실행 계획을 추가한다.",
            "related_criteria": ["feasibility"],
            "evidence_ids": [],
        }
    ],
    "final_decision": None,
    "decision_note": None,
}


def _competition_raw_by_marker(mapping: dict) -> dict:
    """공모전 매핑(기획·개발 전문가 2인) 기준으로 각 위원이 자기 주담당 항목 전부를
    채점한 raw 응답을 프롬프트 마커별로 구성한다."""
    routing = build_routing(mapping)
    criteria_by_id = {c["criterion_id"]: c for c in mapping["rubric"]}

    raw_by_marker = {}
    for persona_id in mapping["committee"]:
        owned = [
            {
                "criterion_id": cid,
                "criterion_name": criteria_by_id[cid]["criterion_name"],
                "max_score": criteria_by_id[cid]["max_score"],
            }
            for cid, r in routing.items()
            if r["primary"] == persona_id
        ]
        persona_name = _COMPETITION_PERSONA_NAMES[persona_id]
        raw_by_marker[f"{persona_name}입니다"] = _make_raw_reviewer(persona_id, persona_name, owned)
    raw_by_marker["위원장(review_chair)입니다"] = _RAW_CHAIR_FOR_RUN
    return raw_by_marker


def test_run_meeting_with_real_competition_mapping_produces_v2_document():
    mapping = json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))
    raw_by_marker = _competition_raw_by_marker(mapping)

    document = run_meeting(
        meeting_id="MTG-RUN-TEST-001",
        project_id="PRJ-RUN-TEST-001",
        document_id="DOC-RUN-TEST-001",
        title="공모전 테스트 회의",
        rubric_mapping=mapping,
        submission={"document_name": "test.pdf", "text": "..."},
        retrieved_evidence=[],
        llm_call=_make_stub_llm(raw_by_marker),
    )

    jsonschema.Draft202012Validator(_load_schema()).validate(document)
    assert document["domain"] == "competition"
    assert len(document["reviewer_results"]) == len(mapping["committee"])
    # 각 항목은 주담당 위원 1명이 max-5점으로 채점한다(위원 수가 아니라 항목 수 기준).
    assert document["score_result"]["total_score"] == document["score_result"]["max_score"] - 5 * len(
        mapping["rubric"]
    )


# ---------------------------------------------------------------------------
# llm.py — make_openai_llm_call (실제 API 호출 없이 가짜 client로 검증)
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = type("_Message", (), {"content": self._content})()
        choice = type("_Choice", (), {"message": message})()
        return type("_Response", (), {"choices": [choice]})()


class _FakeOpenAIClient:
    def __init__(self, content: str):
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(content)})()


def test_make_openai_llm_call_sends_given_model_and_returns_content():
    fake_client = _FakeOpenAIClient('{"ok": true}')
    llm_call = make_openai_llm_call(model="gpt-test-model", client=fake_client)

    text = llm_call("아무 프롬프트")

    assert text == '{"ok": true}'
    assert fake_client.chat.completions.last_kwargs["model"] == "gpt-test-model"


# ---------------------------------------------------------------------------
# MTG-006: 진행 상태 통지 + 실패 노드부터 재시도
# ---------------------------------------------------------------------------


def test_run_meeting_reports_progress_until_completed():
    """on_progress가 단계마다 호출되고, 위원 완료 수가 단조 증가하며, 마지막에 완료된다."""
    mapping = json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))
    raw_by_marker = _competition_raw_by_marker(mapping)

    events: list[dict] = []
    run_meeting(
        meeting_id="MTG-PROGRESS-001",
        project_id="PRJ-PROGRESS-001",
        document_id="DOC-PROGRESS-001",
        title="진행률 테스트",
        rubric_mapping=mapping,
        submission={"document_name": "test.pdf", "text": "..."},
        retrieved_evidence=[],
        llm_call=_make_stub_llm(raw_by_marker),
        on_progress=events.append,
    )

    assert events, "진행 콜백이 한 번도 호출되지 않았다"
    dones = [e["reviews_done"] for e in events]
    assert dones == sorted(dones), "위원 완료 수는 단조 증가해야 한다"

    last = events[-1]
    assert last["stage"] == "완료"
    assert last["reviews_done"] == last["reviews_total"] == len(mapping["committee"])
    assert last["score_done"] is True
    assert last["chair_done"] is True


def test_meeting_graph_resumes_from_failed_chair_without_rerunning_reviewers():
    """chair 노드가 처음 한 번 실패해도, 같은 thread_id로 재개하면 성공한 위원 노드는
    다시 돌지 않고 chair 부터 이어서 완료된다(MTG-006 실패 노드 재시도)."""
    import pytest
    from langgraph.checkpoint.memory import MemorySaver

    committee = ["business_strategy", "technical_feasibility"]
    base_stub = _make_stub_llm(
        {
            "사업전략 전문가입니다": _RAW_REVIEWER,
            "기술·실현가능성 전문가입니다": _TECHNICAL_RAW,
            "위원장(review_chair)입니다": _RAW_CHAIR,
        }
    )

    reviewer_calls = {"count": 0}
    chair_should_fail = {"fail": True}

    def flaky_llm(prompt: str) -> str:
        if "위원장(review_chair)입니다" in prompt:
            if chair_should_fail["fail"]:
                chair_should_fail["fail"] = False
                raise RuntimeError("chair 첫 실행 실패(테스트)")
        else:
            reviewer_calls["count"] += 1
        return base_stub(prompt)

    saver = MemorySaver()
    graph = assemble_meeting_graph(committee, flaky_llm, checkpointer=saver)
    state = initial_state(
        meeting_id="MTG-RESUME-001",
        domain="startup",
        rubric=_RUBRIC,
        submission={"document_name": "test.pdf", "text": "..."},
        committee=committee,
        retrieved_evidence=_RETRIEVED_EVIDENCE,
    )
    config = {"configurable": {"thread_id": "MTG-RESUME-001"}}

    with pytest.raises(RuntimeError):
        graph.invoke(state, config)
    assert reviewer_calls["count"] == 2, "1차 실행에서 위원 2명이 각각 한 번씩 돌아야 한다"

    # 재개: 입력 None + 같은 thread_id → 실패 지점(chair)부터 이어서 실행
    final = graph.invoke(None, config)

    assert final["stage"] == "완료"
    assert final["chair_summary"]["chair_id"] == "review_chair"
    assert reviewer_calls["count"] == 2, "재개 시 위원 노드는 다시 실행되지 않아야 한다"


# ---------------------------------------------------------------------------
# 회귀 가드: LLM이 지어낸 persona_id를 믿지 않고 committee 키로 교정
# (가은이 실제 OpenAI 호출로 발견한 버그 — assemble_document가 딕셔너리 값의
#  persona_id가 아니라 키를 신뢰. 이 값이 틀리면 rerun_reviewer 필터가 깨져
#  재평가마다 위원이 교체되지 않고 계속 추가된다.)
# ---------------------------------------------------------------------------


def test_document_uses_committee_key_even_if_llm_fabricates_persona_id():
    mapping = json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))
    routing = build_routing(mapping)
    criteria_by_id = {c["criterion_id"]: c for c in mapping["rubric"]}
    # 위원별 대표 항목 1개(주담당 항목 중 마지막) — 이 테스트는 persona_id 교정만 보므로
    # 전 항목을 채점할 필요가 없다.
    owned = {r["primary"]: cid for cid, r in routing.items()}

    # 각 위원 raw 의 persona_id 를 실제 committee id 와 다른 값으로 지어낸다.
    raw_by_marker = {}
    for i, (pid, cid) in enumerate(owned.items()):
        persona_name = _COMPETITION_PERSONA_NAMES[pid]
        raw = _make_raw_reviewer(
            pid,
            persona_name,
            [{"criterion_id": cid, "criterion_name": criteria_by_id[cid]["criterion_name"], "max_score": criteria_by_id[cid]["max_score"]}],
        )
        raw["persona_id"] = f"LLM-FAKE-{i:02d}"  # LLM이 지어낸(신뢰 불가) 값
        raw_by_marker[f"{persona_name}입니다"] = raw
    raw_by_marker["위원장(review_chair)입니다"] = _RAW_CHAIR_FOR_RUN

    document = run_meeting(
        meeting_id="MTG-PID-001",
        project_id="PRJ-PID-001",
        document_id="DOC-PID-001",
        title="persona_id 회귀 가드",
        rubric_mapping=mapping,
        submission={"document_name": "t.pdf", "text": "..."},
        retrieved_evidence=[],
        llm_call=_make_stub_llm(raw_by_marker),
    )

    result_ids = {r["persona_id"] for r in document["reviewer_results"]}
    assert result_ids == set(mapping["committee"]), "reviewer_results persona_id는 실제 committee 키여야 한다"
    assert not any(r["persona_id"].startswith("LLM-FAKE") for r in document["reviewer_results"])

    # rerun_reviewer가 교정된 persona_id로 위원을 '교체'(추가 아님)하는지 — 위원 수 유지
    target = "planning_expert"
    tcid = owned[target]
    rerun_markers = {
        f"{_COMPETITION_PERSONA_NAMES[target]}입니다": _make_raw_reviewer(
            target,
            _COMPETITION_PERSONA_NAMES[target],
            [{"criterion_id": tcid, "criterion_name": criteria_by_id[tcid]["criterion_name"], "max_score": criteria_by_id[tcid]["max_score"]}],
        ),
        "위원장(review_chair)입니다": _RAW_CHAIR_FOR_RUN,
    }
    updated = rerun_reviewer(
        previous_document=document,
        persona_id=target,
        rubric_mapping=mapping,
        submission={"document_name": "t.pdf", "text": "..."},
        retrieved_evidence=[],
        llm_call=_make_stub_llm(rerun_markers),
    )
    assert len(updated["reviewer_results"]) == len(mapping["committee"]), "재평가로 위원이 늘어나면 안 된다"
    assert {r["persona_id"] for r in updated["reviewer_results"]} == set(mapping["committee"])
