import asyncio

from app.api.routes import documents


def test_refetch_failure_reuses_saved_parsed_text(monkeypatch):
    captured = {}

    async def fake_find_by_id(document_id):
        return {"parsed_text": "이미 수집해 둔 공고문 본문"}

    def fail_refetch(url):
        raise ConnectionError("temporary network failure")

    async def fake_index_webpage_background(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(documents.document_repo, "find_by_id", fake_find_by_id)
    monkeypatch.setattr(documents, "load_from_url", fail_refetch)
    monkeypatch.setattr(documents, "_index_webpage_background", fake_index_webpage_background)

    asyncio.run(
        documents._refetch_and_index_webpage_background(
            document_id="doc-1",
            project_id="project-1",
            url="https://official.example/notice",
            title="공식 공고문",
        )
    )

    assert captured["parsed_text"] == "이미 수집해 둔 공고문 본문"
    assert captured["cleaned"].fallback_used is True
    assert captured["cleaned"].cleaned_blocks[0].content == "이미 수집해 둔 공고문 본문"
    assert captured["cleaned"].cleaned_blocks[0].metadata["reindex_source"] == "saved_parsed_text"


def test_refetch_failure_without_saved_text_keeps_failure(monkeypatch):
    updates = []

    async def fake_find_by_id(document_id):
        return {"parsed_text": ""}

    def fail_refetch(url):
        raise ConnectionError("temporary network failure")

    async def fake_update_fields(document_id, patch):
        updates.append((document_id, patch))

    monkeypatch.setattr(documents.document_repo, "find_by_id", fake_find_by_id)
    monkeypatch.setattr(documents.document_repo, "update_fields", fake_update_fields)
    monkeypatch.setattr(documents, "load_from_url", fail_refetch)

    asyncio.run(
        documents._refetch_and_index_webpage_background(
            document_id="doc-2",
            project_id="project-1",
            url="https://official.example/notice",
            title="공식 공고문",
        )
    )

    assert updates[0][0] == "doc-2"
    assert updates[0][1]["status"] == "indexing_failed"
    assert updates[0][1]["indexing_error"]["error_type"] == "ConnectionError"
