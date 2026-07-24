# 작성자: 가은/Claude(2026-07-18)
# 목적: PER-002 동적 rubric(공고문에서 LLM으로 추출한 평가항목을 rubric_mapping 형태로
#       병합하는 build_dynamic_rubric_mapping(), ai/meeting/graph/rubric.py) 검증.
#       경이 리뷰(팀 승인 답변)에서 명시적으로 요청한 4가지: 배점 합계 재계산, 중복
#       criterion_id 거부, committee/perspective 화이트리스트 검증, 검증 실패 시 폴백
#       (폴백 자체는 backend/app/api/routes/meetings.py의 호출부 책임이라 여기선
#       build_dynamic_rubric_mapping()이 ValueError를 던지는 지점까지만 확인한다).
# import: 표준 라이브러리 json/pathlib, pytest; ai/meeting/graph 패키지.

import json
import sys
from pathlib import Path

import pytest

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph import build_dynamic_rubric_mapping, build_rubric, combine_criteria_documents  # noqa: E402

COMPETITION_MAPPING_PATH = MEETING_DIR / "personas" / "rubric_mapping_competition.json"
PERSONA_CARDS_PATH = MEETING_DIR / "personas" / "persona_cards.json"


def _load_base_mapping() -> dict:
    return json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))


def _load_persona_cards() -> dict[str, dict]:
    cards = json.loads(PERSONA_CARDS_PATH.read_text(encoding="utf-8"))["personas"]
    return {c["persona_id"]: c for c in cards}


def _valid_items() -> list[dict]:
    """competition 4개 persona 중 두 명(creativity_originality, technical_feasibility)의
    실제 perspective_id를 써서 만든 2개짜리 정상 추출 결과."""
    return [
        {
            "criterion_id": "idea_novelty_axis",
            "criterion_name": "아이디어 참신성",
            "max_score": 60,
            "required": True,
            "primary_persona_id": "creativity_originality",
            "primary_perspective_id": "idea_novelty",
            "secondary_persona_id": None,
        },
        {
            "criterion_id": "feasibility_axis",
            "criterion_name": "구현 가능성",
            "max_score": 40,
            "required": True,
            "primary_persona_id": "technical_feasibility",
            "primary_perspective_id": "implementation_feasibility",
            "secondary_persona_id": "creativity_originality",
        },
    ]


def test_total_max_score_is_recomputed_from_items_not_trusted_from_llm():
    """total_max_score는 base_mapping이나 LLM 출력이 아니라 최종 max_score 합으로
    서버에서 재계산된다(팀 요청: 배점 합계 검증) — weights.total_max_score()가 선언값과
    계산값 불일치 시 예외를 던지므로 여기서 항상 일치하게 만들어야 한다."""
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    merged = build_dynamic_rubric_mapping(
        base_mapping=base_mapping,
        extracted_items=_valid_items(),
        source_document_id="doc-123",
        persona_cards=persona_cards,
    )
    assert merged["total_max_score"] == 100
    assert merged["total_max_score"] == sum(item["max_score"] for item in merged["rubric"])
    # base_mapping의 100(4항목 x 25)이 아니라 실제 추출된 2항목(60+40)의 합이어야 한다.
    assert base_mapping["total_max_score"] == 100
    assert len(merged["rubric"]) == 2


def test_source_document_id_and_meta_dynamic_flag_are_set():
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    merged = build_dynamic_rubric_mapping(
        base_mapping=base_mapping,
        extracted_items=_valid_items(),
        source_document_id="doc-abc",
        persona_cards=persona_cards,
    )
    assert merged["meta"]["source_document_id"] == "doc-abc"
    assert merged["meta"]["dynamic"] is True
    # committee/default_supplementary_perspectives는 base_mapping 그대로 유지된다.
    assert merged["committee"] == base_mapping["committee"]


def test_explicit_required_keywords_are_preserved_for_score_calibration():
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    items = _valid_items()
    items[0]["description"] = "공고문에 적힌 세부 평가내용"
    items[0]["required_keywords"] = ["공공데이터 활용"]
    items[0]["required_keyword_groups"] = [["웹", "모바일"]]

    merged = build_dynamic_rubric_mapping(
        base_mapping=base_mapping,
        extracted_items=items,
        source_document_id="doc-required",
        persona_cards=persona_cards,
    )

    assert merged["rubric"][0]["required_keywords"] == ["공공데이터 활용"]
    assert merged["rubric"][0]["required_keyword_groups"] == [["웹", "모바일"]]
    assert merged["rubric"][0]["description"] == "공고문에 적힌 세부 평가내용"
    runtime_rubric = build_rubric(merged)
    assert runtime_rubric["criteria"][0]["description"] == "공고문에 적힌 세부 평가내용"
    assert runtime_rubric["criteria"][0]["required_keywords"] == ["공공데이터 활용"]
    assert runtime_rubric["criteria"][0]["required_keyword_groups"] == [["웹", "모바일"]]


def test_dynamic_bonus_rules_are_preserved_separately_from_base_score():
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    rules = [
        {
            "bonus_id": "priority_data",
            "name": "국가중점데이터 활용",
            "points": 2,
            "description": "주최기관이 지정한 국가중점데이터를 활용한 경우",
            "evidence_keywords": ["국가중점데이터"],
            "requires_verification": True,
        }
    ]
    merged = build_dynamic_rubric_mapping(
        base_mapping=base_mapping,
        extracted_items=_valid_items(),
        source_document_id="doc-bonus",
        persona_cards=persona_cards,
        bonus_rules=rules,
        bonus_max_score=2,
    )
    runtime_rubric = build_rubric(merged)

    assert merged["meta"]["rubric_extraction_version"] == 2
    assert runtime_rubric["total_max_score"] == 100
    assert runtime_rubric["bonus_max_score"] == 2
    assert runtime_rubric["bonus_rules"] == rules


def test_multiple_criteria_documents_share_prompt_budget_and_keep_all_source_ids():
    documents = [
        {
            "_id": "url-doc",
            "source_url": "https://contest.example/notice",
            "parsed_text": "웹 공고 " + "가" * 5000,
        },
        {
            "_id": "table-doc",
            "original_filename": "평가기준.docx",
            "parsed_text": "배점표 AI 혁신성 25점 데이터 활용성 20점",
        },
    ]
    combined, source_ids = combine_criteria_documents(documents, max_chars=1000)

    assert source_ids == ["url-doc", "table-doc"]
    assert "웹 공고" in combined
    assert "배점표 AI 혁신성 25점" in combined
    assert len(combined) <= 1000


def test_duplicate_criterion_id_is_rejected():
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    items = _valid_items()
    items[1]["criterion_id"] = items[0]["criterion_id"]  # 중복 유발
    with pytest.raises(ValueError, match="중복"):
        build_dynamic_rubric_mapping(
            base_mapping=base_mapping,
            extracted_items=items,
            source_document_id="doc-123",
            persona_cards=persona_cards,
        )


def test_primary_persona_id_outside_committee_is_rejected():
    """새 위원을 만들 수 없다는 팀 요구사항 — committee(고정 4인)에 없는 persona_id는
    거부된다."""
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    items = _valid_items()
    items[0]["primary_persona_id"] = "policy_fit"  # competition committee엔 없는 persona
    with pytest.raises(ValueError, match="committee"):
        build_dynamic_rubric_mapping(
            base_mapping=base_mapping,
            extracted_items=items,
            source_document_id="doc-123",
            persona_cards=persona_cards,
        )


def test_secondary_persona_id_outside_committee_is_rejected():
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    items = _valid_items()
    items[1]["secondary_persona_id"] = "policy_fit"
    with pytest.raises(ValueError, match="committee"):
        build_dynamic_rubric_mapping(
            base_mapping=base_mapping,
            extracted_items=items,
            source_document_id="doc-123",
            persona_cards=persona_cards,
        )


def test_perspective_id_not_in_persona_whitelist_is_rejected():
    """LLM이 그 위원의 실제 evaluation_perspectives에 없는 값을 지어내면 거부된다."""
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    items = _valid_items()
    items[0]["primary_perspective_id"] = "no_such_perspective"
    with pytest.raises(ValueError, match="evaluation_perspectives"):
        build_dynamic_rubric_mapping(
            base_mapping=base_mapping,
            extracted_items=items,
            source_document_id="doc-123",
            persona_cards=persona_cards,
        )


def test_perspective_id_belonging_to_a_different_persona_is_rejected():
    """다른 위원의 perspective_id를 잘못 배정한 경우도 화이트리스트 위반으로 거부된다
    (예: technical_feasibility 담당인데 business_strategy의 perspective_id를 씀)."""
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    items = _valid_items()
    items[1]["primary_perspective_id"] = "marketability"  # business_strategy 소속 perspective
    with pytest.raises(ValueError, match="evaluation_perspectives"):
        build_dynamic_rubric_mapping(
            base_mapping=base_mapping,
            extracted_items=items,
            source_document_id="doc-123",
            persona_cards=persona_cards,
        )


def test_empty_extracted_items_is_rejected():
    """빈 리스트(공고문에서 아무것도 못 찾은 경우)도 거부되어 호출부가 정적 템플릿으로
    폴백하게 한다."""
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    with pytest.raises(ValueError):
        build_dynamic_rubric_mapping(
            base_mapping=base_mapping,
            extracted_items=[],
            source_document_id="doc-123",
            persona_cards=persona_cards,
        )


def test_invalid_max_score_is_rejected():
    base_mapping = _load_base_mapping()
    persona_cards = _load_persona_cards()
    items = _valid_items()
    items[0]["max_score"] = 0
    with pytest.raises(ValueError, match="max_score"):
        build_dynamic_rubric_mapping(
            base_mapping=base_mapping,
            extracted_items=items,
            source_document_id="doc-123",
            persona_cards=persona_cards,
        )
