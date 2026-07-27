# 작성자: 경이
# 목적: RAG(용준) 연동 경로 검증 — evidence_context의 사전 prompt_guard 삽입, EvidencePool의
#       (document_id, chunk_id) 역조회, 콜백 최종 sufficiency 게이팅(allow_numeric_score=False
#       → 미채점), A안(근거를 RAG-004 linked_evidence_refs로 교체)이 v2 계약을 만족하는지 확인.
# import: 표준 라이브러리 json/sys/pathlib, pytest, jsonschema; ai/meeting graph/prompts 패키지.

import json
import sys
from pathlib import Path

import jsonschema

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
REPO_ROOT = MEETING_DIR.parents[1]
sys.path.insert(0, str(MEETING_DIR))

from graph import build_routing, run_meeting  # noqa: E402
from graph.evidence import EvidencePool  # noqa: E402
from graph.transform import raw_reviewer_to_v2  # noqa: E402
from prompts import build_reviewer_prompt  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "contracts" / "schemas" / "review_output.schema.json"
COMPETITION_MAPPING_PATH = MEETING_DIR / "personas" / "rubric_mapping_competition.json"
RAG_SAMPLES_PATH = MEETING_DIR / "tests" / "fixtures" / "rag_adapter_samples.json"

_PERSONA_NAMES = {
    "creativity_originality": "창의성·독창성 전문가",
    "technical_feasibility": "기술·실현가능성 전문가",
    "business_strategy": "사업전략 전문가",
    "presentation_completeness": "완성도·전달력 전문가",
}


# ---------------------------------------------------------------------------
# prompt_guard 삽입
# ---------------------------------------------------------------------------


def test_build_reviewer_prompt_inserts_criterion_guards():
    guards = [
        ("creativity_appropriateness", "근거가 부족하니 확정적 표현과 점수를 피하세요."),
        ("feasibility", "근거 충분: 통상 평가하세요."),
    ]
    prompt = build_reviewer_prompt(
        "creativity_originality",
        {"rubric_id": "R", "total_max_score": 100, "criteria": []},
        {"text": "..."},
        [],
        evidence_guards=guards,
    )
    assert "[creativity_appropriateness] 근거가 부족하니" in prompt
    assert "[feasibility] 근거 충분" in prompt
    assert "<<" not in prompt


def test_build_reviewer_prompt_without_guards_has_default_notice():
    prompt = build_reviewer_prompt(
        "creativity_originality",
        {"rubric_id": "R", "total_max_score": 100, "criteria": []},
        {"text": "..."},
        [],
    )
    assert "판정이 제공되지 않았습니다" in prompt


# ---------------------------------------------------------------------------
# EvidencePool.register_linked — (document_id, chunk_id) 역조회 + 원문 보강
# ---------------------------------------------------------------------------


def test_register_linked_dedupes_by_doc_chunk_and_backfills_text():
    retrieved = [
        {"chunk_id": "c1", "document_id": "D1", "text": "청크1 원문", "score": 0.9},
    ]
    pool = EvidencePool("business_strategy", retrieved)

    ref = {
        "document_id": "D1",
        "chunk_id": "c1",
        "quote": "짧은 인용",
        "document_name": "제출문서.pdf",
        "section": "문제 정의",
        "page": 3,
        "final_score": 0.87,
    }
    eid1 = pool.register_linked(ref)
    eid2 = pool.register_linked(ref)  # 같은 (D1,c1) → 같은 id
    assert eid1 == eid2 == "EV-business_strategy-001"

    item = pool.as_list()[0]
    assert item["chunk_id"] == "c1"
    assert item["text"] == "청크1 원문"  # retrieved 풀에서 원문 보강
    assert item["quote"] == "짧은 인용"
    assert item["score"] == 0.87
    assert item["document_name"] == "제출문서.pdf"


# ---------------------------------------------------------------------------
# transform 게이팅 (allow_numeric_score)
# ---------------------------------------------------------------------------

_RAW = {
    "review_id": "REV-1",
    "persona_id": "business_strategy",
    "persona_name": "사업전략 전문가",
    "review_round": 1,
    "review_summary": "요약",
    "review_items": [
        {
            "criterion_id": "A",
            "criterion_name": "항목A",
            "max_score": 50,
            "score_recommendation": 40,
            "judgment": "strong",
            "confidence": "high",
            "strengths": [],
            "weaknesses": [],
            "evidence_refs": [],
            "improvement_actions": [],
        },
        {
            "criterion_id": "B",
            "criterion_name": "항목B",
            "max_score": 50,
            "score_recommendation": 30,
            "judgment": "adequate",
            "confidence": "high",
            "strengths": [],
            "weaknesses": [],
            "evidence_refs": [],
            "improvement_actions": [],
        },
    ],
    "cross_reviews": [],
    "out_of_scope": [],
}


def test_transform_gates_criterion_when_numeric_score_blocked():
    criterion_evidence = {
        "A": {
            "linked_evidence_refs": [
                {"document_id": "D1", "chunk_id": "cA", "quote": "A 근거", "final_score": 0.8}
            ],
            "sufficiency": {"allow_numeric_score": True, "allow_definitive_judgment": True},
        },
        "B": {  # 최종 판정: 숫자 점수 차단 → 미채점
            "linked_evidence_refs": [],
            "sufficiency": {"allow_numeric_score": False, "allow_definitive_judgment": False},
        },
    }
    pool = EvidencePool("business_strategy", [])
    result = raw_reviewer_to_v2(_RAW, pool, criterion_evidence=criterion_evidence)

    scored = {s["criterion_id"] for s in result["rubric_scores"]}
    assert scored == {"A"}, "allow_numeric_score=False인 B는 미채점되어야 한다"
    a = result["rubric_scores"][0]
    assert a["evidence_ids"] == ["EV-business_strategy-001"]  # RAG-004 근거로 발급(A안)
    assert a["evidence_status"] == "sufficient"


# ---------------------------------------------------------------------------
# 전체 run_meeting — evidence_context + callback 게이팅 + A안 근거
# ---------------------------------------------------------------------------


def _make_raw_reviewer(persona_id: str, cid: str, cname: str, score: int) -> dict:
    return {
        "review_id": f"REV-{persona_id}",
        "persona_id": persona_id,
        "persona_name": _PERSONA_NAMES[persona_id],
        "review_round": 1,
        "review_summary": f"{_PERSONA_NAMES[persona_id]} 요약",
        "review_items": [
            {
                "criterion_id": cid,
                "criterion_name": cname,
                "max_score": 25,
                "score_recommendation": score,
                "judgment": "adequate",
                "confidence": "high",
                "strengths": ["근거 있음"],
                "weaknesses": [],
                "evidence_refs": [],
                "improvement_actions": [],
            }
        ],
        "cross_reviews": [],
        "out_of_scope": [],
    }


_RAW_CHAIR = {
    "chair_id": "review_chair",
    "overall_assessment": "종합",
    "consensus": [],
    "disagreements": [],
    "top_strengths": [],
    "top_risks": [],
    "final_priority_actions": [],
    "final_decision": None,
    "decision_note": None,
}


def test_run_meeting_with_evidence_context_gates_and_uses_linked_evidence():
    mapping = json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))
    routing = build_routing(mapping)
    criteria_by_id = {c["criterion_id"]: c for c in mapping["rubric"]}
    owned = {r["primary"]: cid for cid, r in routing.items()}  # 1인 1항목

    gated_persona = "presentation_completeness"
    gated_criterion = owned[gated_persona]

    # 위원 stub: 전원 20점
    raw_by_marker = {
        f"{_PERSONA_NAMES[pid]}입니다": _make_raw_reviewer(
            pid, cid, criteria_by_id[cid]["criterion_name"], 20
        )
        for pid, cid in owned.items()
    }
    raw_by_marker["위원장(review_chair)입니다"] = _RAW_CHAIR

    def stub_llm(prompt: str) -> str:
        for marker, raw in raw_by_marker.items():
            if marker in prompt:
                return json.dumps(raw, ensure_ascii=False)
        raise AssertionError("마커 못 찾음")

    # evidence_context: (persona, criterion)별 근거 + 사전 판정
    evidence_context = [
        {
            "persona_id": pid,
            "criterion_id": cid,
            "retrieved_evidence": [
                {"chunk_id": f"chunk-{cid}", "document_id": "DOC-1", "text": f"{cid} 근거 원문", "score": 0.85}
            ],
            "sufficiency": {"prompt_guard": f"[{cid}] 안내", "allow_numeric_score": True, "allow_definitive_judgment": True},
        }
        for pid, cid in owned.items()
    ]

    # 콜백: gated_criterion만 최종 숫자 점수 차단, 나머지는 RAG-004 근거 1건 반환
    def evidence_callback(persona_id: str, criterion_id: str, review_item: dict) -> dict:
        allow = criterion_id != gated_criterion
        refs = (
            [{"document_id": "DOC-1", "chunk_id": f"chunk-{criterion_id}", "quote": "인용", "final_score": 0.85}]
            if allow
            else []
        )
        return {
            "linked_evidence_refs": refs,
            "sufficiency": {"allow_numeric_score": allow, "allow_definitive_judgment": allow},
        }

    document = run_meeting(
        meeting_id="MTG-EV-001",
        project_id="PRJ-EV-001",
        document_id="DOC-EV-001",
        title="근거 연동 테스트",
        rubric_mapping=mapping,
        submission={"document_name": "t.pdf", "text": "..."},
        retrieved_evidence=[],
        llm_call=stub_llm,
        evidence_context=evidence_context,
        evidence_callback=evidence_callback,
    )

    jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(document)

    # 게이팅: 3개 항목만 채점(각 20) → 총점 60, gated 항목은 raw 0
    assert document["score_result"]["total_score"] == 60
    breakdown = {b["criterion_id"]: b for b in document["score_result"]["breakdown"]}
    assert breakdown[gated_criterion]["raw_score"] == 0
    assert breakdown[gated_criterion]["source_review_ids"] == []
    # gated 위원은 채점 결과가 비어야 한다
    gated_result = next(r for r in document["reviewer_results"] if r["persona_id"] == gated_persona)
    assert gated_result["rubric_scores"] == []
    # 근거는 RAG-004 linked ref에서 발급된 EV-*(persona별) 3건
    assert len(document["evidence"]) == 3
    assert all(e["evidence_id"].startswith("EV-") for e in document["evidence"])


# ---------------------------------------------------------------------------
# v2.1.0: similar_success_cases (RAG-006) pass-through
# ---------------------------------------------------------------------------


def _simple_competition_stub(mapping: dict):
    routing = build_routing(mapping)
    criteria_by_id = {c["criterion_id"]: c for c in mapping["rubric"]}
    owned = {r["primary"]: cid for cid, r in routing.items()}
    raw_by_marker = {
        f"{_PERSONA_NAMES[pid]}입니다": _make_raw_reviewer(
            pid, cid, criteria_by_id[cid]["criterion_name"], 20
        )
        for pid, cid in owned.items()
    }
    raw_by_marker["위원장(review_chair)입니다"] = _RAW_CHAIR

    def stub(prompt: str) -> str:
        for marker, raw in raw_by_marker.items():
            if marker in prompt:
                return json.dumps(raw, ensure_ascii=False)
        raise AssertionError("마커 못 찾음")

    return stub


def _run_simple(mapping, **kwargs):
    return run_meeting(
        meeting_id="MTG-SC-001",
        project_id="PRJ-SC-001",
        document_id="DOC-SC-001",
        title="유사사례 테스트",
        rubric_mapping=mapping,
        submission={"document_name": "t.pdf", "text": "..."},
        retrieved_evidence=[],
        llm_call=_simple_competition_stub(mapping),
        **kwargs,
    )


def test_run_meeting_emits_latest_schema_and_passes_similar_success_cases():
    mapping = json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    latest_version = schema["properties"]["schema_version"]["enum"][-1]  # 계약 최신 버전

    cases = {
        "results": [
            {
                "case_id": "C1",
                "title": "수상작 예시",
                "case_type": "award_winner",
                "similarity_score": 0.83,
                "reference_only": True,
            }
        ],
        "total_results": 1,
        "comparison_mode": "selected_case_gap",
        "reference_only": True,
    }

    # RAG-006 결과가 있을 때: 그대로 pass-through
    doc = _run_simple(mapping, similar_success_cases=cases)
    validator.validate(doc)
    assert doc["schema_version"] == latest_version  # 계약 버전이 올라가도 최신값을 발행
    assert doc["similar_success_cases"] == cases
    # 평가 결과엔 영향 없음(4항목 전부 채점되어 총점 80)
    assert doc["score_result"]["total_score"] == 80

    # RAG-006 미실행: null 이어도 v2.1.0 스키마 유효
    doc_none = _run_simple(mapping)
    validator.validate(doc_none)
    assert doc_none["similar_success_cases"] is None


# ---------------------------------------------------------------------------
# 용준 실제 어댑터 출력 샘플로 통합 검증 (rag_adapter_samples.json)
# ---------------------------------------------------------------------------


def _rag_samples() -> dict:
    return json.loads(RAG_SAMPLES_PATH.read_text(encoding="utf-8"))


def test_real_linked_ref_sample_maps_to_v2_evidence():
    """용준 to_linked_evidence_refs() 실제 출력 1건이 v2 evidence[]로 정확히 매핑되는지.
    linked ref엔 text가 없어 retrieved_evidence 샘플에서 원문을 보강한다."""
    samples = _rag_samples()
    retrieved = samples["retrieved_evidence"]
    linked = samples["linked_evidence_refs"][0]

    pool = EvidencePool("business_strategy", retrieved)
    eid = pool.register_linked(linked)
    item = pool.as_list()[0]

    assert eid == "EV-business_strategy-001"
    assert item["chunk_id"] == "CHUNK-014"
    assert item["document_name"] == "사업계획서.pdf"
    assert item["page"] == 6
    assert item["section"] == "시장 분석"
    assert item["quote"] == linked["quote"]
    # text는 retrieved 샘플에서 보강(linked엔 없음)
    assert item["text"] == retrieved[0]["text"]
    # score는 linked.final_score
    assert item["score"] == 0.86


def test_real_retrieved_sample_flows_through_run_meeting():
    """용준 build_meeting_retrieved_evidence() 실제 shape로 evidence_context를 구성해
    run_meeting을 돌리면, business_strategy 근거가 실제 샘플의 원문/출처로 조립된다."""
    samples = _rag_samples()
    retrieved_sample = samples["retrieved_evidence"]
    linked_sample = samples["linked_evidence_refs"]

    mapping = json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))
    routing = build_routing(mapping)
    criteria_by_id = {c["criterion_id"]: c for c in mapping["rubric"]}
    owned = {r["primary"]: cid for cid, r in routing.items()}
    bs_criterion = owned["business_strategy"]

    raw_by_marker = {
        f"{_PERSONA_NAMES[pid]}입니다": _make_raw_reviewer(
            pid, cid, criteria_by_id[cid]["criterion_name"], 20
        )
        for pid, cid in owned.items()
    }
    raw_by_marker["위원장(review_chair)입니다"] = _RAW_CHAIR

    def stub(prompt: str) -> str:
        for marker, raw in raw_by_marker.items():
            if marker in prompt:
                return json.dumps(raw, ensure_ascii=False)
        raise AssertionError("마커 못 찾음")

    # evidence_context: business_strategy는 실제 샘플 근거, 나머지는 근거 없음
    evidence_context = []
    for pid, cid in owned.items():
        evidence_context.append(
            {
                "persona_id": pid,
                "criterion_id": cid,
                "retrieved_evidence": retrieved_sample if pid == "business_strategy" else [],
                "sufficiency": {"prompt_guard": "근거 충분", "allow_numeric_score": True, "allow_definitive_judgment": True},
            }
        )

    def evidence_callback(persona_id, criterion_id, review_item):
        refs = linked_sample if persona_id == "business_strategy" else []
        return {"linked_evidence_refs": refs, "sufficiency": {"allow_numeric_score": True, "allow_definitive_judgment": True}}

    document = run_meeting(
        meeting_id="MTG-RAGE2E-001",
        project_id="PRJ-RAGE2E-001",
        document_id="DOC-001",
        title="RAG 실제 샘플 통합",
        rubric_mapping=mapping,
        submission={"document_name": "사업계획서.pdf", "text": "..."},
        retrieved_evidence=[],
        llm_call=stub,
        evidence_context=evidence_context,
        evidence_callback=evidence_callback,
    )

    jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(document)

    # 전원 채점(20 x 4 = 80)
    assert document["score_result"]["total_score"] == 80
    # business_strategy 근거가 실제 샘플에서 조립됨
    bs = next(r for r in document["reviewer_results"] if r["persona_id"] == "business_strategy")
    ev_ids = bs["rubric_scores"][0]["evidence_ids"]
    assert ev_ids == ["EV-business_strategy-001"]
    ev = next(e for e in document["evidence"] if e["evidence_id"] == ev_ids[0])
    assert ev["chunk_id"] == "CHUNK-014"
    assert ev["text"] == retrieved_sample[0]["text"]
    assert ev["score"] == 0.86


# ---------------------------------------------------------------------------
# 항목별 적정 위원만 채점(경이 확정 2026-07-27) — 배정(expected_criterion_ids) 밖 항목은
# 근거 게이트를 통과해도 점수·지적이 집계에서 완전히 제외된다. 이전에는 비담당 위원
# 점수가 calculator 평균에 섞여 배정과 무관한 3명 평균(1/3 단위 점수)이 생겼다.
# ---------------------------------------------------------------------------


def test_out_of_scope_criterion_is_dropped_even_if_evidence_gate_allows():
    raw = {
        "review_id": "REV-1",
        "persona_id": "business_strategy",
        "review_items": [
            {
                "criterion_id": "my_axis",
                "criterion_name": "담당 항목",
                "judgment": "adequate",
                "score_recommendation": 20,
                "max_score": 25,
                "weaknesses": [],
            },
            {
                "criterion_id": "other_axis",
                "criterion_name": "남의 항목",
                "judgment": "adequate",
                "score_recommendation": 18,
                "max_score": 25,
                "weaknesses": [],
            },
        ],
    }
    pool = EvidencePool("business_strategy", [])
    # 근거 게이트가 두 항목 모두 점수를 허용해도(비담당 항목까지 콜백이 통과시켜도)
    criterion_evidence = {
        cid: {
            "linked_evidence_refs": [],
            "sufficiency": {"allow_numeric_score": True, "allow_definitive_judgment": True},
        }
        for cid in ("my_axis", "other_axis")
    }

    result = raw_reviewer_to_v2(
        raw,
        pool,
        criterion_evidence=criterion_evidence,
        expected_criterion_ids={"my_axis"},
    )

    scored = [s["criterion_id"] for s in result["rubric_scores"]]
    assert scored == ["my_axis"]  # 배정된 항목만 집계
    assert all(u["criterion_id"] == "my_axis" for u in result.get("unscored_criteria", []))


def test_legacy_path_without_assignment_keeps_all_items():
    """expected_criterion_ids=None(레거시 flat 경로)은 기존처럼 전 항목을 유지한다."""
    raw = {
        "review_id": "REV-1",
        "persona_id": "business_strategy",
        "review_items": [
            {
                "criterion_id": "a",
                "criterion_name": "A",
                "judgment": "adequate",
                "score_recommendation": 20,
                "max_score": 25,
                "weaknesses": [],
                "evidence_refs": [],
            },
            {
                "criterion_id": "b",
                "criterion_name": "B",
                "judgment": "adequate",
                "score_recommendation": 18,
                "max_score": 25,
                "weaknesses": [],
                "evidence_refs": [],
            },
        ],
    }
    pool = EvidencePool("business_strategy", [])
    result = raw_reviewer_to_v2(raw, pool, expected_criterion_ids=None)
    assert [s["criterion_id"] for s in result["rubric_scores"]] == ["a", "b"]
