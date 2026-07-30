# 작성자: 용준/Claude(2026-07-23, 요청: RAG 근거 실제 활용 강화 — evidence 참조 안정화)
# 목적: call_evidence_lookup이 evidence_lookup의 반환값 각 항목에 안정적인 순번 참조
#       ("ref": "E1", "E2", ...)를 부여하는지 확인한다. chunk_id(해시 20자 안팎)를 LLM이
#       그대로 베껴 써야 하는 부담을 줄이기 위한 필드이므로, 기존 chunk_id는 그대로 남아
#       있어야 하고(순수 추가), evidence_lookup이 None이거나 결과가 없는 경우도 안전해야
#       한다.

import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting

sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_nodes import call_evidence_lookup  # noqa: E402


def test_call_evidence_lookup_assigns_sequential_refs_preserving_chunk_id():
    def lookup(persona_id, query):
        return [
            {"chunk_id": "chk_aaaa1111", "text": "첫 번째 청크"},
            {"chunk_id": "chk_bbbb2222", "text": "두 번째 청크"},
        ]

    result = call_evidence_lookup(lookup, "planning_expert", "query")
    assert [item["ref"] for item in result] == ["E1", "E2"]
    assert result[0]["chunk_id"] == "chk_aaaa1111"
    assert result[1]["chunk_id"] == "chk_bbbb2222"


def test_call_evidence_lookup_does_not_overwrite_existing_ref():
    def lookup(persona_id, query):
        return [{"chunk_id": "chk_1", "ref": "CUSTOM"}]

    result = call_evidence_lookup(lookup, "planning_expert", "query")
    assert result[0]["ref"] == "CUSTOM"


def test_call_evidence_lookup_none_returns_empty_list():
    assert call_evidence_lookup(None, "planning_expert", "query") == []


def test_call_evidence_lookup_empty_results_stay_empty():
    assert call_evidence_lookup(lambda p, q: [], "planning_expert", "query") == []


def test_call_evidence_lookup_with_runtime_scope_still_assigns_refs():
    def lookup(persona_id, query, *, runtime_scope=None):
        assert runtime_scope == {"session_id": "S1"}
        return [{"chunk_id": "chk_1", "text": "t"}]

    result = call_evidence_lookup(lookup, "dev_expert", "query", runtime_scope={"session_id": "S1"})
    assert result[0]["ref"] == "E1"


def test_call_evidence_lookup_excludes_user_session_answer_chunks():
    """용준/Claude(2026-07-30, 요청: "사용자 답변은 문서 기반 grounded evidence로 사용하지
    마세요") — ideation_source_type="user_session_answer"인 항목은 검색 결과 최상위에
    있어도(top-1) 이 지점에서 제외돼야, 이후 프롬프트의 retrieved_evidence/turn_evidence·
    ground_claims 검증 대상·claim_evidence_links·evidence 버킷 어디에도 등장하지 않는다."""

    def lookup(persona_id, query):
        return [
            {"chunk_id": "chk_answer1", "document_role": "target", "ideation_source_type": "user_session_answer", "text": "사용자 답변 원문"},
            {"chunk_id": "chk_candidate1", "document_role": "target", "ideation_source_type": "ideation_candidate", "text": "선택 아이디어"},
            {"chunk_id": "chk_criteria1", "document_role": "criteria", "text": "공고문 조항"},
        ]

    result = call_evidence_lookup(lookup, "planning_expert", "query")
    chunk_ids = [item["chunk_id"] for item in result]
    assert "chk_answer1" not in chunk_ids
    assert chunk_ids == ["chk_candidate1", "chk_criteria1"]
    assert [item["ref"] for item in result] == ["E1", "E2"]
