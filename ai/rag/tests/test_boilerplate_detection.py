from ai.rag.preprocessing.boilerplate_detection import (
    extract_boundary_lines,
    find_repeated_boundary_texts,
    is_boilerplate_text,
)

# 실측(reports/rag007_e2e_20260730/search_top5.json)에서 top-5에 그대로 노출된
# pipc.go.kr 사이트 메뉴 청크는 줄바꿈 없이 하나의 긴 문자열(120자 초과)로 추출돼
# 있어, 첫/마지막 "줄" 기준인 이 알고리즘의 경계 탐지 대상에서 애초에 벗어난다 —
# 이는 알려진 한계다(완료 보고에 명시). 아래 테스트는 그 대신 실제 메뉴 항목이
# 줄바꿈으로 구분되는(HTML 리스트/PDF 텍스트에서 흔한) 현실적인 형태로 검증한다.
_PIPC_MENU_LINES = "위원회 소식\n공지사항\n채용공시\n보도‧해명자료\n정책 · 법령\n개인정보 보호 캠페인\n마이데이터 제도"


def test_extract_boundary_lines_filters_by_length():
    text = "짧음\n" + "가" * 8 + "\n" + "나" * 121 + "\n본문"
    boundaries = extract_boundary_lines(text)
    assert "가" * 8 in boundaries
    assert "가" * 121 not in boundaries
    assert "짧음" not in boundaries


def test_find_repeated_boundary_texts_requires_min_group_count():
    items = [
        ("doc1", "공통 안내 문구입니다 안내"),
        ("doc2", "공통 안내 문구입니다 안내"),
    ]
    # 2개 문서에만 등장 -> 기본 min_group_count=3 미만이라 탐지되지 않는다.
    assert find_repeated_boundary_texts(items, min_group_count=3) == set()

    items.append(("doc3", "공통 안내 문구입니다 안내"))
    repeated = find_repeated_boundary_texts(items, min_group_count=3)
    assert "공통 안내 문구입니다 안내" in repeated


def test_find_repeated_boundary_texts_pdf_page_scope_within_one_document():
    """PDF 머리글/바닥글: 같은 document_id 안에서 페이지 번호를 group_key로 쓴다."""
    header = "2026년도 공공 AI 활용 가이드라인 보고서"
    items = [
        ("1", f"{header}\n본문 내용 1"),
        ("2", f"{header}\n본문 내용 2"),
        ("3", f"{header}\n본문 내용 3"),
    ]
    repeated = find_repeated_boundary_texts(items, min_group_count=3)
    assert header in repeated


def test_is_boilerplate_text_matches_repeated_menu():
    repeated = find_repeated_boundary_texts(
        [("doc1", _PIPC_MENU_LINES), ("doc2", _PIPC_MENU_LINES), ("doc3", _PIPC_MENU_LINES)],
        min_group_count=3,
    )
    assert is_boilerplate_text(_PIPC_MENU_LINES, repeated)


def test_is_boilerplate_text_false_for_unrelated_text():
    repeated = {"어떤 반복 문구"}
    assert not is_boilerplate_text("전혀 관련 없는 본문 내용입니다", repeated)


def test_is_boilerplate_text_empty_repeated_set_never_matches():
    assert not is_boilerplate_text("아무 텍스트", set())
