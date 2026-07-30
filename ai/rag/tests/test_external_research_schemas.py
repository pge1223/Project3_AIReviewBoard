"""
Unit Tests for ai.rag.external_research.schemas (RAG-007)
================================================================
"""

import math

import pytest

from ai.rag.external_research.exceptions import ExternalResearchValidationError
from ai.rag.external_research.schemas import (
    ExternalEvidenceDocument,
    ExternalEvidenceType,
    ExternalResearchRequest,
)


def _doc(**overrides) -> ExternalEvidenceDocument:
    base = dict(
        source_id="KOSIS-POP-2025",
        document_id="DOC-001",
        chunk_id="CHUNK-001",
        title="연령별 인구 통계",
        evidence_type=ExternalEvidenceType.STATISTICS,
        publisher="통계청",
        source_url="https://kosis.kr/example",
        domain="public_service",
        content="2025년 12월 기준 전국 인구는 5,170만 명입니다.",
    )
    base.update(overrides)
    return ExternalEvidenceDocument(**base)


def _request(**overrides) -> ExternalResearchRequest:
    base = dict(
        domain="공공 AI 서비스",
        evaluation_criteria=["시장성", "정책 적합성"],
        reviewer_role="planning",
    )
    base.update(overrides)
    return ExternalResearchRequest(**base)


class TestExternalEvidenceDocument:
    def test_valid_document_created(self):
        doc = _doc()
        assert doc.evidence_type == ExternalEvidenceType.STATISTICS
        assert doc.source_url == "https://kosis.kr/example"

    def test_missing_source_url_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _doc(source_url="")

    def test_missing_publisher_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _doc(publisher="")

    def test_missing_document_id_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _doc(document_id="")

    def test_missing_chunk_id_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _doc(chunk_id="")

    def test_empty_content_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _doc(content="")

    def test_blank_title_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _doc(title="   ")

    def test_invalid_evidence_type_rejected(self):
        with pytest.raises(ValueError):
            ExternalEvidenceDocument(
                source_id="s", document_id="d", chunk_id="c", title="t",
                evidence_type="not_a_real_type", publisher="p", source_url="https://x",
                domain="dom", content="본문",
            )

    def test_valid_reference_date_accepted(self):
        doc = _doc(reference_date="2025-12-31")
        assert doc.reference_date == "2025-12-31"

    def test_invalid_reference_date_format_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _doc(reference_date="2025/12/31")


class TestExternalEvidenceDocumentNewQualityFields:
    """용준/Claude(2026-07-30, 요청: RAG-007 색인 청크 품질 정제) — 신규 필드가 전부
    Optional/기본값을 가져 기존 호출부(신규 필드를 넘기지 않는 코드)와 하위 호환되는지
    확인한다."""

    def test_existing_style_call_without_new_fields_still_works(self):
        doc = _doc()
        assert doc.file_hash is None
        assert doc.source_content_hash is None
        assert doc.chunk_content_hash is None
        assert doc.normalized_content_hash is None
        assert doc.page_start is None
        assert doc.page_end is None
        assert doc.content_level is None
        assert doc.url_verified is None
        assert doc.direct_file_url is None
        assert doc.allow_grounded_claim is True

    def test_new_fields_can_be_set_explicitly(self):
        doc = _doc(
            file_hash="abc123",
            source_content_hash="abc123",
            chunk_content_hash="def456",
            normalized_content_hash="def456",
            page_start=3,
            page_end=4,
            content_level="pdf_attachment",
            url_verified=True,
            direct_file_url="https://example.org/file.pdf",
            allow_grounded_claim=False,
        )
        assert doc.file_hash == "abc123"
        assert doc.page_start == 3
        assert doc.page_end == 4
        assert doc.content_level == "pdf_attachment"
        assert doc.url_verified is True
        assert doc.direct_file_url == "https://example.org/file.pdf"
        assert doc.allow_grounded_claim is False

    def test_invalid_published_at_format_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _doc(published_at="Dec 2025")

    def test_none_dates_allowed(self):
        doc = _doc(reference_date=None, published_at=None, retrieved_at=None)
        assert doc.reference_date is None

    def test_metric_fields_default_none(self):
        doc = _doc()
        assert doc.metric_name is None
        assert doc.metric_value is None
        assert doc.metric_unit is None

    def test_metric_fields_pass_through_when_provided(self):
        doc = _doc(metric_name="총인구", metric_value=51700000, metric_unit="명")
        assert doc.metric_value == 51700000


class TestExternalResearchRequest:
    def test_valid_request_constructed(self):
        request = _request()
        assert request.top_k == 5

    def test_blank_domain_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(domain="  ")

    def test_empty_evaluation_criteria_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(evaluation_criteria=[])

    def test_blank_evaluation_criteria_entry_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(evaluation_criteria=["시장성", "  "])

    def test_blank_reviewer_role_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(reviewer_role="")

    def test_zero_top_k_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(top_k=0)

    def test_negative_top_k_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(top_k=-3)

    def test_excessive_top_k_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(top_k=10_000)

    def test_nan_min_score_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(min_score=math.nan)

    def test_infinite_min_score_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(min_score=math.inf)

    def test_unrecognized_evidence_type_rejected(self):
        with pytest.raises(ValueError):
            ExternalResearchRequest(
                domain="d", evaluation_criteria=["c"], reviewer_role="planning",
                evidence_types=["not_a_real_type"],
            )

    def test_valid_evidence_types_accepted(self):
        request = _request(evidence_types=[ExternalEvidenceType.STATISTICS, ExternalEvidenceType.MARKET])
        assert request.evidence_types == [ExternalEvidenceType.STATISTICS, ExternalEvidenceType.MARKET]

    def test_invalid_reference_date_format_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(reference_date="31-12-2025")

    def test_excessively_long_query_context_rejected(self):
        with pytest.raises(ExternalResearchValidationError):
            _request(query_context="가" * 3000)

    def test_reasonable_query_context_accepted(self):
        request = _request(query_context="공공기관 사업계획서 자동 평가 서비스")
        assert request.query_context is not None
