"""
Unit Tests for ai.rag.integration.meeting_evidence_adapter
(mock RoleSearchResponse/LinkedEvaluation만 사용 — 실제 Chroma/KURE/LLM/LangGraph 없음)
"""

import copy

from ai.rag.evidence_linking.schemas import EvidenceSource, LinkedEvaluation
from ai.rag.integration.meeting_evidence_adapter import (
    build_meeting_retrieved_evidence,
    to_linked_evidence_refs,
    to_retrieved_evidence,
)
from ai.rag.integration.schemas import PersonaRoleSearchResponse
from ai.rag.role_retrieval.schemas import RoleSearchResponse, RoleSearchResult


def _result(
    chunk_id="chunk-1",
    document_id="doc-1",
    content="본문 내용",
    semantic_score=0.8,
    role_score=0.4,
    final_score=0.7,
    metadata=None,
) -> RoleSearchResult:
    return RoleSearchResult(
        record_id=f"{document_id}::{chunk_id}",
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        semantic_score=semantic_score,
        role_score=role_score,
        final_score=final_score,
        role_id="finance",
        metadata=metadata if metadata is not None else {},
    )


def _response(results, role_id="finance", role_name="재무 심사위원", project_id="p1") -> RoleSearchResponse:
    return RoleSearchResponse(
        query="예산",
        expanded_query="재무 심사 관점에서 예산",
        role_id=role_id,
        role_name=role_name,
        project_id=project_id,
        document_id=None,
        results=results,
        result_count=len(results),
        warnings=[],
    )


class TestBasicFieldConversion:
    def test_chunk_id_preserved(self):
        response = _response([_result(chunk_id="chunk-017")])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["chunk_id"] == "chunk-017"

    def test_document_id_preserved(self):
        response = _response([_result(document_id="doc-001")])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["document_id"] == "doc-001"

    def test_persona_id_reflects_input(self):
        response = _response([_result()])
        items = to_retrieved_evidence(response, persona_id="technology")
        assert items[0]["persona_id"] == "technology"

    def test_role_id_reflects_explicit_argument(self):
        response = _response([_result()], role_id="finance")
        items = to_retrieved_evidence(response, persona_id="finance", role_id="finance")
        assert items[0]["role_id"] == "finance"

    def test_role_id_falls_back_to_response_role_id_when_not_given(self):
        response = _response([_result()], role_id="marketing")
        items = to_retrieved_evidence(response, persona_id="marketing")
        assert items[0]["role_id"] == "marketing"

    def test_content_maps_to_text(self):
        response = _response([_result(content="총사업비는 5,000만 원이다.")])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["text"] == "총사업비는 5,000만 원이다."

    def test_document_title_maps_to_document_name(self):
        response = _response([_result(metadata={"document_title": "사업계획서.pdf"})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["document_name"] == "사업계획서.pdf"

    def test_section_title_maps_to_section(self):
        response = _response([_result(metadata={"section_title": "예산 계획"})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["section"] == "예산 계획"

    def test_location_number_maps_to_page(self):
        response = _response([_result(metadata={"location_number": 7})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["page"] == 7

    def test_location_number_preserved(self):
        response = _response([_result(metadata={"location_number": 7})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["location_number"] == 7

    def test_location_type_preserved(self):
        response = _response([_result(metadata={"location_type": "page"})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["location_type"] == "page"

    def test_document_role_preserved(self):
        """용준/Claude(2026-07-22, 요청: 역할별 검색 데이터 구성) — 색인 시점의
        IndexingContext.document_role("criteria"/"target")이 그대로 노출돼야
        ideation_evidence_service._compose_by_document_role이 역할별 쿼터를 계산할 수 있다."""
        response = _response([_result(metadata={"document_role": "criteria"})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["document_role"] == "criteria"

    def test_document_role_missing_is_none(self):
        response = _response([_result(metadata={})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["document_role"] is None

    def test_origin_type_and_file_name_and_source_url_preserved(self):
        """용준/Claude(2026-07-30, 요청: criteria 근거의 출처(URL 본문/URL 첨부파일/직접
        업로드) 구분) — 청킹 단계가 채운 chunk 메타데이터의 source_type/source_filename/
        source_url이 화면까지 흘러가려면 이 어댑터가 먼저 노출해야 한다."""
        response = _response(
            [
                _result(
                    metadata={
                        "source_type": "url_attachment",
                        "source_filename": "공고_별첨.hwp",
                        "source_url": "https://example.go.kr/notice/attachment.hwp",
                    }
                )
            ]
        )
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["origin_type"] == "url_attachment"
        assert items[0]["file_name"] == "공고_별첨.hwp"
        assert items[0]["source_url"] == "https://example.go.kr/notice/attachment.hwp"

    def test_origin_type_missing_is_none(self):
        response = _response([_result(metadata={})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["origin_type"] is None
        assert items[0]["file_name"] is None
        assert items[0]["source_url"] is None


class TestScoreConversion:
    # RoleSearchResult.final_score는 필수 필드(float, None 불가)라 이 입력 경로에서는
    # final_score가 항상 채워져 있다 — 그래서 score == final_score 케이스만 RoleSearchResult로
    # 직접 재현 가능하다. final_score/semantic_score 둘 다 없는 경우의 폴백 체인은
    # resolve_score()(ai/rag/evidence_linking/linker.py, RAG-004 기존 유닛 테스트가 이미
    # 검증)를 그대로 재사용하므로 여기서 다시 검증하지 않는다.
    def test_score_equals_final_score_when_present(self):
        response = _response([_result(final_score=0.73, semantic_score=0.82)])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["score"] == 0.73

    def test_score_uses_final_score_even_when_semantic_score_missing(self):
        response = _response([_result(final_score=0.68, semantic_score=None)])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["score"] == 0.68

    def test_raw_score_fields_preserved(self):
        response = _response([_result(semantic_score=0.82, role_score=0.45, final_score=0.73)])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["semantic_score"] == 0.82
        assert items[0]["role_score"] == 0.45
        assert items[0]["final_score"] == 0.73


class TestMetadataEdgeCases:
    def test_metadata_empty_dict_does_not_raise(self):
        response = _response([_result(metadata={})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["document_name"] is None

    def test_metadata_empty_dict(self):
        response = _response([_result(metadata={})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["document_name"] is None
        assert items[0]["section"] is None
        assert items[0]["page"] is None
        assert items[0]["location_number"] is None
        assert items[0]["location_type"] is None

    def test_document_title_missing(self):
        response = _response([_result(metadata={"section_title": "예산"})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["document_name"] is None

    def test_section_title_missing(self):
        response = _response([_result(metadata={"document_title": "doc.pdf"})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["section"] is None

    def test_location_number_missing(self):
        response = _response([_result(metadata={})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["location_number"] is None

    def test_location_type_missing(self):
        response = _response([_result(metadata={})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["location_type"] is None

    def test_location_number_string_digit_is_converted(self):
        response = _response([_result(metadata={"location_number": "7"})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["location_number"] == 7

    def test_invalid_location_number_does_not_raise_and_yields_none(self):
        response = _response([_result(metadata={"location_number": "not-a-number"})])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["location_number"] is None


class TestFlatListHandling:
    def test_order_preserved(self):
        response = _response([_result(chunk_id="c1"), _result(chunk_id="c2"), _result(chunk_id="c3")])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert [i["chunk_id"] for i in items] == ["c1", "c2", "c3"]

    def test_multiple_personas_merged_into_one_flat_list(self):
        finance_response = _response([_result(chunk_id="c1", document_id="doc-1")])
        tech_response = _response([_result(chunk_id="c2", document_id="doc-1")], role_id="technology")
        combined = build_meeting_retrieved_evidence([
            PersonaRoleSearchResponse(persona_id="finance", response=finance_response, role_id="finance"),
            PersonaRoleSearchResponse(persona_id="technology", response=tech_response, role_id="technology"),
        ])
        assert len(combined) == 2
        assert {item["persona_id"] for item in combined} == {"finance", "technology"}

    def test_persona_id_correct_per_item(self):
        finance_response = _response([_result(chunk_id="c1")])
        tech_response = _response([_result(chunk_id="c2")], role_id="technology")
        combined = build_meeting_retrieved_evidence([
            PersonaRoleSearchResponse(persona_id="finance", response=finance_response, role_id="finance"),
            PersonaRoleSearchResponse(persona_id="technology", response=tech_response, role_id="technology"),
        ])
        by_chunk = {item["chunk_id"]: item["persona_id"] for item in combined}
        assert by_chunk == {"c1": "finance", "c2": "technology"}

    def test_same_chunk_different_persona_both_kept(self):
        finance_response = _response([_result(chunk_id="chunk-1", document_id="doc-1")])
        tech_response = _response([_result(chunk_id="chunk-1", document_id="doc-1")], role_id="technology")
        combined = build_meeting_retrieved_evidence([
            PersonaRoleSearchResponse(persona_id="finance", response=finance_response, role_id="finance"),
            PersonaRoleSearchResponse(persona_id="technology", response=tech_response, role_id="technology"),
        ])
        assert len(combined) == 2

    def test_empty_response_returns_empty_list(self):
        response = _response([])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items == []


class TestDuplicateHandling:
    def test_same_persona_duplicate_document_chunk_removed(self):
        response = _response([
            _result(chunk_id="c1", document_id="doc-1"),
            _result(chunk_id="c1", document_id="doc-1"),
        ])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert len(items) == 1

    def test_first_occurrence_order_kept(self):
        first = _result(chunk_id="c1", document_id="doc-1", content="첫 번째")
        second = _result(chunk_id="c1", document_id="doc-1", content="두 번째")
        response = _response([first, second])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert items[0]["text"] == "첫 번째"

    def test_same_chunk_id_different_document_id_both_kept(self):
        response = _response([
            _result(chunk_id="c1", document_id="doc-1"),
            _result(chunk_id="c1", document_id="doc-2"),
        ])
        items = to_retrieved_evidence(response, persona_id="finance")
        assert len(items) == 2

    def test_same_document_and_chunk_different_persona_both_kept(self):
        finance_response = _response([_result(chunk_id="c1", document_id="doc-1")])
        tech_response = _response([_result(chunk_id="c1", document_id="doc-1")], role_id="technology")
        combined = build_meeting_retrieved_evidence([
            PersonaRoleSearchResponse(persona_id="finance", response=finance_response, role_id="finance"),
            PersonaRoleSearchResponse(persona_id="technology", response=tech_response, role_id="technology"),
        ])
        assert len(combined) == 2


class TestLinkedEvidenceConversion:
    def _linked(self, has_evidence=True, evidence=None):
        return LinkedEvaluation(
            opinion="인건비 비중이 높아 예산 구조의 위험이 있습니다.",
            role_id="finance",
            role_name="재무 심사위원",
            has_evidence=has_evidence,
            evidence=evidence or [],
        )

    def _source(self, **overrides):
        defaults = dict(
            document_id="doc-001",
            chunk_id="chunk-017",
            document_title="사업계획서.pdf",
            page_number=7,
            section_title="예산 계획",
            content_kind="body",
            quote="총사업비는 5,000만 원이다.",
            semantic_score=0.82,
            role_score=0.45,
            final_score=0.73,
        )
        defaults.update(overrides)
        return EvidenceSource(**defaults)

    def test_evidence_source_converted_to_ref_dict(self):
        linked = self._linked(evidence=[self._source()])
        refs = to_linked_evidence_refs(linked)
        assert len(refs) == 1

    def test_document_id_and_chunk_id_preserved(self):
        linked = self._linked(evidence=[self._source(document_id="doc-9", chunk_id="chunk-9")])
        refs = to_linked_evidence_refs(linked)
        assert refs[0]["document_id"] == "doc-9"
        assert refs[0]["chunk_id"] == "chunk-9"

    def test_quote_preserved(self):
        linked = self._linked(evidence=[self._source(quote="인용문 원문")])
        refs = to_linked_evidence_refs(linked)
        assert refs[0]["quote"] == "인용문 원문"

    def test_document_name_section_page_mapped(self):
        linked = self._linked(evidence=[self._source(
            document_title="사업계획서.pdf", section_title="예산 계획", page_number=7,
        )])
        refs = to_linked_evidence_refs(linked)
        assert refs[0]["document_name"] == "사업계획서.pdf"
        assert refs[0]["section"] == "예산 계획"
        assert refs[0]["page"] == 7

    def test_score_fields_preserved(self):
        linked = self._linked(evidence=[self._source(semantic_score=0.82, role_score=0.45, final_score=0.73)])
        refs = to_linked_evidence_refs(linked)
        assert refs[0]["semantic_score"] == 0.82
        assert refs[0]["role_score"] == 0.45
        assert refs[0]["final_score"] == 0.73

    def test_no_evidence_returns_empty_list(self):
        linked = self._linked(has_evidence=False, evidence=[])
        refs = to_linked_evidence_refs(linked)
        assert refs == []

    def test_no_evidence_id_field_generated(self):
        linked = self._linked(evidence=[self._source()])
        refs = to_linked_evidence_refs(linked)
        assert "evidence_id" not in refs[0]


class TestInputImmutability:
    def test_role_search_response_not_mutated(self):
        response = _response([_result(metadata={"document_title": "doc.pdf"})])
        snapshot = response.model_copy(deep=True)
        to_retrieved_evidence(response, persona_id="finance")
        assert response == snapshot

    def test_role_search_result_metadata_not_mutated(self):
        result = _result(metadata={"document_title": "doc.pdf"})
        response = _response([result])
        metadata_snapshot = copy.deepcopy(result.metadata)
        to_retrieved_evidence(response, persona_id="finance")
        assert result.metadata == metadata_snapshot

    def test_linked_evaluation_not_mutated(self):
        linked = LinkedEvaluation(
            opinion="의견",
            role_id="finance",
            role_name="재무 심사위원",
            has_evidence=True,
            evidence=[EvidenceSource(
                document_id="doc-1", chunk_id="c1", quote="인용문",
                document_title="doc.pdf", section_title="예산", page_number=1,
            )],
        )
        snapshot = linked.model_copy(deep=True)
        to_linked_evidence_refs(linked)
        assert linked == snapshot


def _has_meeting_import(module) -> bool:
    with open(module.__file__, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("import ai.meeting") or stripped.startswith("from ai.meeting"):
                return True
    return False


class TestScopeBoundary:
    def test_adapter_module_does_not_import_meeting_graph(self):
        import ai.rag.integration.meeting_evidence_adapter as adapter_module

        assert not _has_meeting_import(adapter_module)

    def test_schemas_module_does_not_import_meeting_graph(self):
        import ai.rag.integration.schemas as schemas_module

        assert not _has_meeting_import(schemas_module)
