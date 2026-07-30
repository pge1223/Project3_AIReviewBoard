from ai.rag.preprocessing.content_hashing import (
    compute_chunk_content_hash,
    compute_normalized_content_hash,
    compute_source_content_hash,
    deduplicate_chunks,
    normalize_for_content_hash,
)


def test_source_content_hash_is_deterministic():
    assert compute_source_content_hash("동일한 원문") == compute_source_content_hash("동일한 원문")


def test_source_content_hash_differs_for_different_text():
    assert compute_source_content_hash("원문 A") != compute_source_content_hash("원문 B")


def test_normalize_collapses_whitespace_but_keeps_digits_and_punctuation():
    normalized = normalize_for_content_hash("법령 제10조   1항\n\n실업률은 3.5%입니다.")
    assert normalized == "법령 제10조 1항 실업률은 3.5%입니다."


def test_normalized_content_hash_ignores_whitespace_differences():
    a = compute_normalized_content_hash("실업률은  3.5%입니다.")
    b = compute_normalized_content_hash("실업률은\n3.5%입니다.")
    assert a == b


def test_normalized_content_hash_differs_when_numbers_differ():
    a = compute_normalized_content_hash("실업률은 3.5%입니다.")
    b = compute_normalized_content_hash("실업률은 4.5%입니다.")
    assert a != b


def test_chunk_content_hash_is_exact_not_normalized():
    a = compute_chunk_content_hash("동일 내용")
    b = compute_chunk_content_hash("동일  내용")
    assert a != b


def test_deduplicate_chunks_keeps_first_and_records_dropped():
    chunks = [
        {"chunk_id": "C1", "document_id": "D1", "page": 1, "section": "S1", "normalized_content_hash": "H1"},
        {"chunk_id": "C2", "document_id": "D1", "page": 2, "section": "S1", "normalized_content_hash": "H1"},
        {"chunk_id": "C3", "document_id": "D2", "page": 1, "section": "S2", "normalized_content_hash": "H2"},
    ]
    kept, dropped = deduplicate_chunks(chunks)

    assert [c["chunk_id"] for c in kept] == ["C1", "C3"]
    assert [c["chunk_id"] for c in dropped] == ["C2"]
    assert dropped[0]["duplicate_of_chunk_id"] == "C1"


def test_deduplicate_chunks_preserves_merged_source_ranges_on_kept_item():
    chunks = [
        {"chunk_id": "C1", "document_id": "D1", "page": 1, "section": "S1", "normalized_content_hash": "H1"},
        {"chunk_id": "C2", "document_id": "D1", "page": 2, "section": "S1", "normalized_content_hash": "H1"},
    ]
    kept, _ = deduplicate_chunks(chunks)

    ranges = kept[0]["metadata"]["merged_source_ranges"]
    assert ranges == [{"document_id": "D1", "chunk_id": "C2", "page": 2, "section": "S1"}]


def test_deduplicate_chunks_no_hash_key_passes_through_unchanged():
    chunks = [{"chunk_id": "C1", "document_id": "D1"}]
    kept, dropped = deduplicate_chunks(chunks)
    assert kept == chunks
    assert dropped == []
