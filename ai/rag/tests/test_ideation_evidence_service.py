"""
Ideation Evidence Service — Role Composition Tests
=======================================================
용준/Claude(2026-07-22, 요청: 역할별 검색 데이터 구성 검토). planning_expert/dev_expert가
동일한 semantic 검색 풀에서 대부분 같은(가장 점수가 높은) 공고문 청크만 받던 문제를
검증한다 — _compose_by_document_role이 실제 metadata.document_role("criteria"/"target")
값만으로 역할별 쿼터를 구성하고, 존재하지 않는 role은 임의로 채우지 않는지 확인한다.

용준/Claude(2026-07-23, 요청: stale closure + target starvation 수정 검증) — 아래
TestTargetStarvationDirectSearch/TestRuntimeScopeOverride는 실 사이트에서 확인된 버그
(criteria 청크 100개가 top-N을 모두 차지해 target이 project-wide 검색에 전혀 없어도,
selected_candidate_document_id를 알고 있으면 직접 검색으로 target을 확보해야 한다 +
evidence_lookup 호출 시점의 runtime_scope가 lookup 생성 시점의 closure 스냅샷보다
우선해야 한다)를 검증한다.
"""

from ai.rag.orchestration.ideation_evidence_service import (
    _build_issue_focused_query,
    _compose_by_document_role,
    _scope_target_evidence,
    make_ideation_evidence_lookup,
    search_ideation_evidence,
)
from ai.rag.retrieval.schemas import SearchResult
from ai.rag.role_retrieval.roles import RoleRegistry
from ai.rag.role_retrieval.service import RoleAwareRetrievalService


def _item(chunk_id: str, document_role: str | None, score: float = 0.9) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": f"DOC-{chunk_id}",
        "document_role": document_role,
        "final_score": score,
        "text": f"content {chunk_id}",
    }


def test_planning_expert_prioritizes_criteria_over_target():
    items = [
        _item("C1", "criteria", 0.95),
        _item("C2", "criteria", 0.9),
        _item("C3", "criteria", 0.85),
        _item("C4", "criteria", 0.8),
        _item("T1", "target", 0.99),
        _item("T2", "target", 0.7),
    ]
    composed, missing = _compose_by_document_role(items, persona_id="planning_expert", top_k=5)
    roles = [item["document_role"] for item in composed]
    assert roles.count("criteria") == 3
    assert roles.count("target") == 2
    assert missing == []
    assert len(composed) == 5


def test_dev_expert_prioritizes_target_over_criteria():
    items = [
        _item("C1", "criteria", 0.95),
        _item("C2", "criteria", 0.9),
        _item("C3", "criteria", 0.85),
        _item("T1", "target", 0.99),
        _item("T2", "target", 0.8),
        _item("T3", "target", 0.7),
    ]
    composed, missing = _compose_by_document_role(items, persona_id="dev_expert", top_k=5)
    roles = [item["document_role"] for item in composed]
    assert roles.count("target") == 3
    assert roles.count("criteria") == 2
    assert missing == []


def test_missing_document_role_is_reported_not_backfilled_with_wrong_label():
    """target 문서가 프로젝트에 전혀 없으면(공고문만 있는 프로젝트) missing_document_roles에
    "target"이 기록되고, 부족분은 criteria로만 채워진다 — "domain" 같은 존재하지 않는 role을
    지어내 채우지 않는다."""
    items = [
        _item("C1", "criteria", 0.95),
        _item("C2", "criteria", 0.9),
        _item("C3", "criteria", 0.85),
        _item("C4", "criteria", 0.8),
    ]
    composed, missing = _compose_by_document_role(items, persona_id="dev_expert", top_k=5)
    assert missing == ["target"]
    assert all(item["document_role"] == "criteria" for item in composed)
    assert len(composed) == 4  # 실제로 검색된 후보(4개)보다 더 채워 넣지 않는다.


def test_unclassified_documents_only_used_as_last_resort_fill():
    """document_role 메타데이터가 없는(구버전 색인) 후보는 쿼터를 못 채운 나머지만 채운다."""
    items = [
        _item("C1", "criteria", 0.95),
        _item("U1", None, 0.99),  # 구버전 색인(역할 미분류) — 점수는 가장 높다.
    ]
    composed, missing = _compose_by_document_role(items, persona_id="planning_expert", top_k=5)
    assert missing == ["target"]
    roles = [item["document_role"] for item in composed]
    assert roles.count("criteria") == 1
    assert None in roles  # 부족분을 미분류 후보로 보충했다.


def test_no_quota_configured_for_unknown_persona_returns_items_unchanged():
    items = [_item("C1", "criteria"), _item("T1", "target")]
    composed, missing = _compose_by_document_role(items, persona_id="ideation_facilitator", top_k=5)
    assert composed == items
    assert missing == []


def test_problem_issue_builds_compact_criteria_query():
    query = _build_issue_focused_query(
        "침수 안전 서비스 | 현재 쟁점: 문제 정의 | 검토 관점: 기술 구조와 구현 가능성"
    )
    assert query is not None
    assert query.startswith("문제 정의 ")
    assert "도시 문제의 설정" in query
    assert "구현 가능성" not in query


def test_unknown_issue_does_not_add_supplemental_search():
    assert _build_issue_focused_query("아이디어 | 현재 쟁점: 임의 자유 토론") is None


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-22, 요청: 세션 범위 검색 + 후보 변경 시 이전 candidate target 제외).
# ---------------------------------------------------------------------------


def _ideation_item(chunk_id: str, *, ideation_source_type=None, document_id=None, session_id=None) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id or f"DOC-{chunk_id}",
        "document_role": "target",
        "ideation_source_type": ideation_source_type,
        "session_id": session_id,
        "final_score": 0.9,
        "text": f"content {chunk_id}",
    }


def test_general_project_documents_always_pass_scope_filter():
    """ideation_source_type이 없는(일반 project criteria/target 문서) 항목은 session_id나
    selected_candidate_document_id와 무관하게 항상 통과한다."""
    items = [_item("C1", "criteria"), _item("T1", "target")]
    scoped = _scope_target_evidence(items, session_id="S1", selected_candidate_document_id=None)
    assert scoped == items


def test_other_sessions_user_answer_is_excluded():
    """다른 회의 세션의 사용자 답변이 현재 회의에 섞이면 안 된다(요청 5번)."""
    items = [
        _ideation_item("A1", ideation_source_type="user_session_answer", session_id="S1"),
        _ideation_item("A2", ideation_source_type="user_session_answer", session_id="OTHER_SESSION"),
    ]
    scoped = _scope_target_evidence(items, session_id="S1", selected_candidate_document_id=None)
    chunk_ids = [item["chunk_id"] for item in scoped]
    assert chunk_ids == ["A1"]


def test_previous_candidate_target_excluded_after_reselection():
    """사용자가 후보를 다시 선택/결합하면 이전 후보의 target은 현재 근거로 검색되지 않는다
    (요청 17-5번) — Chroma에는 회의 이력으로 남아있어도 검색 결과에서는 제외된다."""
    items = [
        _ideation_item("OLD1", ideation_source_type="ideation_candidate", document_id="ideation-target::P1::S1::candidate_1"),
        _ideation_item("NEW1", ideation_source_type="ideation_candidate", document_id="ideation-target::P1::S1::candidate_2"),
    ]
    scoped = _scope_target_evidence(
        items, session_id="S1", selected_candidate_document_id="ideation-target::P1::S1::candidate_2"
    )
    chunk_ids = [item["chunk_id"] for item in scoped]
    assert chunk_ids == ["NEW1"]


def test_no_selected_candidate_document_id_excludes_all_candidate_target_items():
    """아직 후보 색인이 완료되지 않았거나 selected_candidate_document_id를 모르면(색인 실패
    등) candidate target은 안전하게 전부 제외한다 — 가짜로 통과시키지 않는다."""
    items = [_ideation_item("X1", ideation_source_type="ideation_candidate", document_id="ideation-target::P1::S1::c1")]
    scoped = _scope_target_evidence(items, session_id="S1", selected_candidate_document_id=None)
    assert scoped == []


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-23, 요청: target starvation 보강 — 필수 테스트 1/2/8번). 실 사이트
# 실측: criteria 청크가 project-wide top-N 후보 풀을 모두 차지하면 target이 검색 결과에
# 전혀 없다. selected_candidate_document_id를 알고 있으므로 project-wide semantic 순위와
# 무관하게 그 document_id를 직접 검색해야 한다.
# ---------------------------------------------------------------------------


class _FakeRoleRetrievalBackend:
    """RAGIndexingService.search()와 동일한 시그니처의 fake. document_id가 주어지면 그
    document_id 전용 레코드만(project-wide 후보 풀과 별도로) 반환해, 실제 Chroma의
    document_id 필터 동작을 흉내낸다."""

    def __init__(self, project_wide_records: list[SearchResult], by_document_id: dict[str, list[SearchResult]]):
        self._project_wide_records = project_wide_records
        self._by_document_id = by_document_id
        self.calls: list[dict] = []

    def search(self, query, project_id, document_id=None, top_k=5):
        self.calls.append({"query": query, "project_id": project_id, "document_id": document_id, "top_k": top_k})
        if document_id is not None:
            return list(self._by_document_id.get(document_id, []))[:top_k]
        return list(self._project_wide_records)[:top_k]


def _search_result(record_id: str, document_id: str, *, document_role: str, score: float = 0.9, **extra_metadata) -> SearchResult:
    return SearchResult(
        record_id=record_id,
        chunk_id=record_id,
        document_id=document_id,
        content=f"content {record_id}",
        distance=1.0 - score,
        score=score,
        metadata={"document_role": document_role, **extra_metadata},
    )


class TestTargetStarvationDirectSearch:
    def test_starved_target_is_found_via_direct_document_id_search(self):
        """criteria 청크 100개가 top-N을 모두 차지해도(project-wide 검색에 target이 전혀
        없어도), selected_candidate_document_id로 직접 검색해 target을 확보한다."""
        target_document_id = "ideation-target::P1::S1::candidate_1"
        criteria_records = [
            _search_result(f"C{i}", "doc-criteria", document_role="criteria", score=0.99 - i * 0.001)
            for i in range(100)
        ]
        target_records = [
            _search_result(
                "T1", target_document_id, document_role="target",
                ideation_source_type="ideation_candidate", session_id="S1",
            )
        ]
        fake = _FakeRoleRetrievalBackend(criteria_records, {target_document_id: target_records})
        service = RoleAwareRetrievalService(retrieval_service=fake, role_registry=RoleRegistry())

        composed = search_ideation_evidence(
            "dev_expert", "쿼리", "P1", service, top_k=5,
            session_id="S1", selected_candidate_document_id=target_document_id,
        )

        target_items = [item for item in composed if item.get("document_role") == "target"]
        assert len(target_items) >= 1
        assert any(item.get("document_id") == target_document_id for item in target_items)
        # project-wide 검색(document_id=None)에서는 target이 전혀 없었음을 재확인한다 —
        # 즉 direct search가 없었다면 이 테스트는 target_items == []가 됐을 것이다.
        project_wide_calls = [c for c in fake.calls if c["document_id"] is None]
        assert project_wide_calls
        assert all(r.metadata["document_role"] == "criteria" for r in criteria_records[: project_wide_calls[0]["top_k"]])

    def test_without_selected_candidate_document_id_no_direct_search_and_no_fabricated_target(self):
        """selected_candidate_document_id가 없으면 direct search 자체를 하지 않고, 없는
        target을 가짜로 만들어내지도 않는다."""
        criteria_records = [_search_result(f"C{i}", "doc-criteria", document_role="criteria") for i in range(10)]
        fake = _FakeRoleRetrievalBackend(criteria_records, {})
        service = RoleAwareRetrievalService(retrieval_service=fake, role_registry=RoleRegistry())

        composed = search_ideation_evidence(
            "dev_expert", "쿼리", "P1", service, top_k=5, session_id="S1", selected_candidate_document_id=None
        )

        assert all(item.get("document_role") != "target" for item in composed)
        assert all(c["document_id"] is None for c in fake.calls)  # document_id 직접 검색이 호출되지 않았다.

    def test_direct_search_excludes_other_document_ids(self):
        """direct search가 요청한 document_id와 다른 문서를 반환하면(방어적 상황) 결과에
        섞이지 않는다."""
        target_document_id = "ideation-target::P1::S1::candidate_2"
        other_document_id = "ideation-target::P1::S1::candidate_1"
        fake = _FakeRoleRetrievalBackend(
            [],
            {target_document_id: [_search_result("WRONG", other_document_id, document_role="target")]},
        )
        service = RoleAwareRetrievalService(retrieval_service=fake, role_registry=RoleRegistry())

        composed = search_ideation_evidence(
            "dev_expert", "쿼리", "P1", service, top_k=5,
            session_id="S1", selected_candidate_document_id=target_document_id,
        )
        assert composed == []


class TestRuntimeScopeOverride:
    """make_ideation_evidence_lookup이 반환하는 lookup은 closure 생성 시점의
    selected_candidate_document_id(스냅샷)보다, 호출 시점에 전달된 runtime_scope를
    우선해야 한다 — 이것이 stale closure 버그의 직접적인 수정 대상이다."""

    def test_runtime_scope_overrides_closure_snapshot(self):
        target_document_id = "ideation-target::P1::S1::candidate_1"
        target_records = [
            _search_result(
                "T1", target_document_id, document_role="target",
                ideation_source_type="ideation_candidate", session_id="S1",
            )
        ]
        fake = _FakeRoleRetrievalBackend([], {target_document_id: target_records})
        service = RoleAwareRetrievalService(retrieval_service=fake, role_registry=RoleRegistry())

        # closure는 요청 시작 시점의 값(아직 후보 선택 전)이라 selected_candidate_document_id
        # 없이 lookup을 만든다 — /reply 시작 시점의 previous_state를 흉내낸다.
        lookup = make_ideation_evidence_lookup(
            project_id="P1", role_retrieval_service=service, top_k=5,
            session_id="S1", selected_candidate_document_id=None,
        )

        # runtime_scope 없이 부르면(배치형 호출자와 동일) closure 스냅샷(None)만 쓰이므로
        # target을 못 찾는다 — 수정 전 버그와 동일한 상황을 재현한다.
        without_runtime_scope = lookup("dev_expert", "쿼리")
        assert all(item.get("document_role") != "target" for item in without_runtime_scope)

        # 같은 lookup 인스턴스를, 그래프 노드가 evidence_lookup을 호출하는 순간의 최신
        # state(candidate_selection 직후 갱신된 selected_idea_document_id)로 다시 부르면
        # target을 확보한다 — closure를 다시 만들지 않고도(같은 /reply 안에서) 최신 값이
        # 반영되는 것이 이번 수정의 핵심이다.
        with_runtime_scope = lookup(
            "dev_expert", "쿼리",
            runtime_scope={"session_id": "S1", "selected_candidate_document_id": target_document_id},
        )
        target_items = [item for item in with_runtime_scope if item.get("document_role") == "target"]
        assert len(target_items) >= 1
        assert target_items[0]["document_id"] == target_document_id

    def test_runtime_scope_reselection_excludes_previous_candidate(self):
        """같은 요청 안에서 후보 A -> B로 재선택되면, runtime_scope가 B로 갱신된 이후의
        호출은 A target을 더 이상 포함하지 않는다(요청 7번 후보 재선택 처리)."""
        doc_a = "ideation-target::P1::S1::candidate_1"
        doc_b = "ideation-target::P1::S1::candidate_2"
        fake = _FakeRoleRetrievalBackend(
            [],
            {
                doc_a: [_search_result("A1", doc_a, document_role="target", ideation_source_type="ideation_candidate", session_id="S1")],
                doc_b: [_search_result("B1", doc_b, document_role="target", ideation_source_type="ideation_candidate", session_id="S1")],
            },
        )
        service = RoleAwareRetrievalService(retrieval_service=fake, role_registry=RoleRegistry())
        lookup = make_ideation_evidence_lookup(
            project_id="P1", role_retrieval_service=service, top_k=5, session_id="S1", selected_candidate_document_id=doc_a
        )

        result_b = lookup("dev_expert", "쿼리", runtime_scope={"session_id": "S1", "selected_candidate_document_id": doc_b})
        document_ids = {item["document_id"] for item in result_b if item.get("document_role") == "target"}
        assert document_ids == {doc_b}
