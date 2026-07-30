"""
Claim-Evidence Semantic Alignment (Shadow Mode) Unit Tests
================================================================
semantic_alignment.py는 claim_grounding.py::ground_claims()의 키워드 stem 겹침 검사(1차)를
대체하지 않고, 그 검사를 통과한 (claim, evidence) 쌍에만 LLM judge를 한 번 더 돌려
entailed/partially_entailed/topic_only/contradicted를 판정한다 — 현재는 shadow mode라
판정 결과가 ground_claims()의 반환값을 절대 바꾸지 않는다. 이 테스트는 실제 LLM을 쓰지
않고 canned JSON을 반환하는 fake llm_call로 검증한다.
"""

import json

from ai.rag.evidence_linking.semantic_alignment import (
    build_pairs,
    ground_claims_with_alignment_shadow,
    judge_claim_evidence_alignment,
)

EVIDENCE = [
    {
        "ref": "E1",
        "chunk_id": "C1",
        "document_id": "DOC-1",
        "document_name": "공공부문 AI 도입·활용 가이드",
        "section": None,
        "text": (
            "이 문서는 'AI를 도입하자'는 제안서가 아니라, 서비스가 계속할 가치가 있는지 "
            "판단하는 안전장치다. AI는 도구일 뿐이며 기준은 '문제가 실제로 줄어들었는가'다."
        ),
        "source_type": "official_report",
    },
]

CLAIMS = [
    {
        "claim_id": "claim_1",
        "text": "AI 모델 선택 근거를 문서화하지 않으면 신뢰성을 잃는다.",
        "claim_type": "document_fact",
        "evidence_refs": ["E1"],
    }
]


# shadow wrapper 테스트 전용 — lexical 1차 검사를 통과하도록 명확히 겹치는 fixture
# (test_claim_grounding.py의 EVIDENCE와 같은 스타일).
CLEAN_EVIDENCE = [
    {
        "ref": "E1",
        "chunk_id": "C1",
        "document_id": "DOC-2",
        "document_name": "WSCE2026 공고문",
        "section": "평가 기준",
        "text": "본 사업은 실현 가능성과 경제성을 중점적으로 평가한다.",
        "source_type": "criteria",
    },
]
CLEAN_CLAIMS = [
    {
        "claim_id": "claim_1",
        "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
        "claim_type": "document_fact",
        "evidence_refs": ["E1"],
    }
]


def _fake_llm_call(response_payload: dict):
    def call(prompt: str) -> str:
        return json.dumps(response_payload, ensure_ascii=False)

    return call


def _claim_evidence_links_for_c1():
    return [{"claim_id": "claim_1", "evidence_refs": ["E1"], "chunk_ids": ["C1"]}]


def test_build_pairs_only_includes_actually_linked_evidence():
    links = _claim_evidence_links_for_c1()
    pairs = build_pairs(links, CLAIMS, EVIDENCE)
    assert len(pairs) == 1
    assert pairs[0]["claim_id"] == "claim_1"
    assert pairs[0]["chunk_id"] == "C1"
    assert pairs[0]["claim_text"] == CLAIMS[0]["text"]


def test_build_pairs_skips_unlinked_claims():
    """검색됐지만 인용되지 않은 청크는 대상에 포함하지 않는다."""
    pairs = build_pairs([], CLAIMS, EVIDENCE)
    assert pairs == []


def test_judge_returns_entailed_label():
    pairs = build_pairs(_claim_evidence_links_for_c1(), CLAIMS, EVIDENCE)
    llm_call = _fake_llm_call(
        {"judgments": [{"claim_id": "claim_1", "chunk_id": "C1", "label": "entailed", "reason": "직접 뒷받침"}]}
    )
    judgments = judge_claim_evidence_alignment(pairs, llm_call, model="test-model")
    assert len(judgments) == 1
    assert judgments[0]["label"] == "entailed"


def test_judge_returns_topic_only_label_for_generic_word_overlap():
    """실측 사례 재현 — "AI"/"문서" 같은 범용 단어만 겹치고 결론 방향이 다른 쌍."""
    pairs = build_pairs(_claim_evidence_links_for_c1(), CLAIMS, EVIDENCE)
    llm_call = _fake_llm_call(
        {
            "judgments": [
                {
                    "claim_id": "claim_1",
                    "chunk_id": "C1",
                    "label": "topic_only",
                    "reason": "AI/문서라는 범용 단어만 겹치고 근거는 서비스 지속 여부 판단 기준을 말할 뿐",
                }
            ]
        }
    )
    judgments = judge_claim_evidence_alignment(pairs, llm_call, model="test-model")
    assert judgments[0]["label"] == "topic_only"


def test_judge_returns_partially_entailed_label():
    pairs = build_pairs(_claim_evidence_links_for_c1(), CLAIMS, EVIDENCE)
    llm_call = _fake_llm_call(
        {
            "judgments": [
                {"claim_id": "claim_1", "chunk_id": "C1", "label": "partially_entailed", "reason": "일부만 뒷받침"}
            ]
        }
    )
    judgments = judge_claim_evidence_alignment(pairs, llm_call, model="test-model")
    assert judgments[0]["label"] == "partially_entailed"


def test_judge_returns_contradicted_label():
    pairs = build_pairs(_claim_evidence_links_for_c1(), CLAIMS, EVIDENCE)
    llm_call = _fake_llm_call(
        {"judgments": [{"claim_id": "claim_1", "chunk_id": "C1", "label": "contradicted", "reason": "반대 결론"}]}
    )
    judgments = judge_claim_evidence_alignment(pairs, llm_call, model="test-model")
    assert judgments[0]["label"] == "contradicted"


def test_judge_falls_back_to_topic_only_on_parse_failure():
    """판정 실패 시 낙관적으로 entailed 처리하지 않고 topic_only로 보수적으로 강등한다."""
    pairs = build_pairs(_claim_evidence_links_for_c1(), CLAIMS, EVIDENCE)

    def broken_llm_call(prompt: str) -> str:
        return "이것은 JSON이 아닙니다"

    judgments = judge_claim_evidence_alignment(pairs, broken_llm_call, model="test-model")
    assert len(judgments) == 1
    assert judgments[0]["label"] == "topic_only"


def test_judge_falls_back_to_topic_only_on_invalid_label():
    pairs = build_pairs(_claim_evidence_links_for_c1(), CLAIMS, EVIDENCE)
    llm_call = _fake_llm_call(
        {"judgments": [{"claim_id": "claim_1", "chunk_id": "C1", "label": "확실함", "reason": "..."}]}
    )
    judgments = judge_claim_evidence_alignment(pairs, llm_call, model="test-model")
    assert judgments[0]["label"] == "topic_only"


def test_judge_with_no_pairs_returns_empty_without_calling_llm():
    calls = {"n": 0}

    def counting_llm_call(prompt: str) -> str:
        calls["n"] += 1
        return "{}"

    judgments = judge_claim_evidence_alignment([], counting_llm_call, model="test-model")
    assert judgments == []
    assert calls["n"] == 0


def test_shadow_wrapper_never_changes_ground_claims_result():
    """shadow 보장 — llm_call이 있어도 ground_claims()의 반환값(근거 표시)은 그대로다.
    build_pairs 전용 EVIDENCE/CLAIMS는 C-1(범용어 보강)로 lexical 검사 자체를 통과하지
    못하므로(의도된 동작), 여기서는 명확히 연결되는 별도 fixture를 쓴다."""
    llm_call = _fake_llm_call(
        {"judgments": [{"claim_id": "claim_1", "chunk_id": "C1", "label": "topic_only", "reason": "무관"}]}
    )
    result = ground_claims_with_alignment_shadow(
        CLEAN_CLAIMS, CLEAN_EVIDENCE, llm_call=llm_call, model="test-model", trace_sink=None
    )
    assert result["linked_evidence_refs"] == ["C1"]
    assert result["evidence_status"] == "grounded"


def test_shadow_wrapper_reports_judgments_via_trace_sink():
    llm_call = _fake_llm_call(
        {"judgments": [{"claim_id": "claim_1", "chunk_id": "C1", "label": "topic_only", "reason": "무관"}]}
    )
    captured = []
    ground_claims_with_alignment_shadow(
        CLEAN_CLAIMS, CLEAN_EVIDENCE, llm_call=llm_call, model="test-model", trace_sink=captured.append
    )
    assert len(captured) == 1
    payload = captured[0]
    assert payload["event"] == "IDEATION_CLAIM_EVIDENCE_ALIGNMENT_SHADOW"
    assert payload["pair_count"] == 1
    assert payload["label_counts"] == {"topic_only": 1}


def test_shadow_wrapper_without_llm_call_is_pure_passthrough():
    """llm_call=None(플래그 꺼짐)이면 judge를 전혀 호출하지 않고 ground_claims() 그대로."""
    result = ground_claims_with_alignment_shadow(CLEAN_CLAIMS, CLEAN_EVIDENCE, llm_call=None, trace_sink=None)
    assert result["linked_evidence_refs"] == ["C1"]


def test_shadow_wrapper_with_no_linked_evidence_skips_judge_call():
    """claim_evidence_links가 비어 있으면(모든 근거가 lexical 검사에서 이미 탈락) judge를
    호출하지 않는다."""
    calls = {"n": 0}

    def counting_llm_call(prompt: str) -> str:
        calls["n"] += 1
        return "{}"

    unrelated_claims = [
        {
            "claim_id": "claim_1",
            "text": "이 아이디어와 전혀 무관한 완전히 다른 주장입니다",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        }
    ]
    ground_claims_with_alignment_shadow(
        unrelated_claims, EVIDENCE, llm_call=counting_llm_call, model="test-model", trace_sink=None
    )
    assert calls["n"] == 0
