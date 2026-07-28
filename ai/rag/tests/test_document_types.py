from ai.rag.domain.document_types import (
    ANNOUNCEMENT,
    APPLICATION_FORM_BEST_PRACTICE,
    APPLICATION_FORM_POC,
    EVALUATION_CRITERIA,
    normalize_document_type,
    preferred_document_types,
)


def test_legacy_korean_types_are_normalized():
    assert normalize_document_type("공고문") == ANNOUNCEMENT
    assert normalize_document_type("평가기준") == EVALUATION_CRITERIA


def test_application_form_is_split_by_content():
    assert (
        normalize_document_type(
            "신청서양식",
            document_name="붙임2.hwpx",
            text="실증·PoC 수행보고서 작성 요령",
        )
        == APPLICATION_FORM_POC
    )
    assert (
        normalize_document_type(
            "신청서양식",
            document_name="붙임3.hwp",
            text="AI 활용 우수사례 신청서 작성 요령",
        )
        == APPLICATION_FORM_BEST_PRACTICE
    )


def test_evaluation_intent_precedes_poc_form_intent():
    preferences = preferred_document_types("실증·PoC 심사 평가 항목과 배점")
    assert preferences[:2] == (EVALUATION_CRITERIA, ANNOUNCEMENT)
    assert APPLICATION_FORM_POC in preferences


def test_announcement_precedes_best_practice_form_for_submission_conditions():
    preferences = preferred_document_types(
        "AI 활용 우수사례 부문에 제출할 수 있는 운영 사례의 조건은 무엇인가요?"
    )
    assert preferences[0] == ANNOUNCEMENT
    assert APPLICATION_FORM_BEST_PRACTICE in preferences


def test_best_practice_form_stays_first_for_writing_question():
    preferences = preferred_document_types(
        "AI 활용 우수사례 수행보고서는 어떤 내용을 작성해야 하나요?"
    )
    assert preferences[0] == APPLICATION_FORM_BEST_PRACTICE


def test_ai_adoption_necessity_routes_to_poc_form():
    preferences = preferred_document_types(
        "AI 도입 필요성과 적절성을 설명할 때 어떤 내용을 구체화해야 하나요?"
    )
    assert preferences[0] == APPLICATION_FORM_POC
