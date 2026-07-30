from ai.rag.preprocessing.short_chunk_policy import merge_or_filter_short_chunks


def _chunk(chunk_id, document_id, content, *, page=1, section="섹션A"):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "content": content,
        "page": page,
        "section": section,
        "metadata": {},
    }


def test_short_chunk_merges_into_previous_when_adjacent_and_same_section():
    long_chunk = _chunk("C1", "D1", "본문 " * 60)  # >= 100자
    short_chunk = _chunk("C2", "D1", "짧은 꼬리 문장입니다.")
    result = merge_or_filter_short_chunks([long_chunk, short_chunk])

    assert len(result.kept) == 1
    assert "짧은 꼬리 문장입니다." in result.kept[0]["content"]
    assert result.merged_count == 1
    assert result.kept[0]["metadata"]["merged_from_chunk_ids"] == ["C2"]


def test_short_chunk_merges_forward_when_no_previous_candidate():
    short_chunk = _chunk("C1", "D1", "짧은 도입부입니다.")
    long_chunk = _chunk("C2", "D1", "본문 " * 60)
    result = merge_or_filter_short_chunks([short_chunk, long_chunk])

    assert len(result.kept) == 1
    assert result.kept[0]["chunk_id"] == "C2"
    assert "짧은 도입부입니다." in result.kept[0]["content"]
    assert result.merged_count == 1


def test_short_chunk_not_merged_across_different_sections():
    short_chunk = _chunk("C1", "D1", "짧은 문장입니다.", section="섹션A")
    other_section_long = _chunk("C2", "D1", "본문 " * 60, section="섹션B")
    result = merge_or_filter_short_chunks([short_chunk, other_section_long])

    # 병합 불가 + 보존 패턴 미해당 -> 제거된다.
    dropped_ids = [c["chunk_id"] for c in result.dropped]
    assert "C1" in dropped_ids
    assert len(result.kept) == 1
    assert result.kept[0]["chunk_id"] == "C2"


def test_short_chunk_not_merged_when_page_gap_too_large():
    short_chunk = _chunk("C1", "D1", "짧은 문장입니다.", page=1)
    far_page_long = _chunk("C2", "D1", "본문 " * 60, page=10)
    result = merge_or_filter_short_chunks([short_chunk, far_page_long])

    dropped_ids = [c["chunk_id"] for c in result.dropped]
    assert "C1" in dropped_ids


def test_legal_article_number_preserved_even_when_short():
    lonely_short = _chunk("C1", "D1", "제10조 (정의)")
    result = merge_or_filter_short_chunks([lonely_short])

    assert len(result.kept) == 1
    assert result.preserved_short and result.preserved_short[0]["chunk_id"] == "C1"
    assert result.dropped == []


def test_statistics_figure_preserved_even_when_short():
    lonely_short = _chunk("C1", "D1", "고령층 접근성 지표는 62.3%입니다.")
    result = merge_or_filter_short_chunks([lonely_short])

    assert result.kept[0]["chunk_id"] == "C1"
    assert result.preserved_short


def test_menu_like_short_chunk_without_preserve_pattern_is_dropped():
    lonely_short = _chunk("C1", "D1", "홈으로 이동")
    result = merge_or_filter_short_chunks([lonely_short])

    assert result.kept == []
    assert result.dropped and result.dropped[0]["chunk_id"] == "C1"


def test_does_not_merge_when_combined_length_exceeds_max_chars():
    short_chunk = _chunk("C1", "D1", "짧은 문장.")
    huge_chunk = _chunk("C2", "D1", "본문 " * 500)  # 훨씬 큼
    result = merge_or_filter_short_chunks([short_chunk, huge_chunk], max_chars=800)

    dropped_ids = [c["chunk_id"] for c in result.dropped]
    assert "C1" in dropped_ids
    assert result.kept[0]["chunk_id"] == "C2"


def test_long_chunks_pass_through_unmodified():
    long_chunk = _chunk("C1", "D1", "본문 " * 60)
    result = merge_or_filter_short_chunks([long_chunk])

    assert result.kept == [long_chunk]
    assert result.merged_count == 0
    assert result.dropped == []
    assert result.preserved_short == []


def test_different_documents_do_not_merge_with_each_other():
    short_a = _chunk("C1", "D1", "짧은 문장입니다.")
    long_b = _chunk("C2", "D2", "본문 " * 60)
    result = merge_or_filter_short_chunks([short_a, long_b])

    dropped_ids = [c["chunk_id"] for c in result.dropped]
    assert "C1" in dropped_ids
    assert [c["chunk_id"] for c in result.kept] == ["C2"]
