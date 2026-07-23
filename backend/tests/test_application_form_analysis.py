# 작성자: 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입 → 업로드 영역 통합)
# 목적: POST /documents/{project_id}/application-form-analysis 검증 — "공모전 공고 · 평가기준 ·
#       신청서 양식"을 한 업로드 영역(document_role="criteria")으로 합친 뒤이므로, 이
#       엔드포인트는 announcement-analysis와 같은 문서 풀을 신청서 양식 관점으로 다시
#       읽는다(별도 document_role 없음). announcement-analysis와 같은 정책(문서 없으면
#       LLM 호출 안 함/프로젝트당 1회 캐시/양식에 없는 값을 지어내지 않음)을 그대로
#       지키는지 확인한다. document_repo/project_repo(둘 다 documents.py 모듈 싱글턴
#       인스턴스)의 메서드를 monkeypatch로 직접 대체해 실제 MongoDB 없이 라우트를
#       검증한다(backend/tests/test_ideation_conversation_llm_call_budget.py가 OpenAI
#       클라이언트를 대체하는 것과 같은 원칙).
# import: fastapi.testclient(conftest client/auth_header 픽스처 재사용), app.api.routes.documents.

import json

import pytest

import app.api.routes.documents as documents_route


# 가은/Claude(2026-07-24, dev 병합 후 회귀 수정) — application-form-analysis에
# analysis_version 필드(_APPLICATION_FORM_ANALYSIS_CACHE_VERSION)와 전역 공고문 캐시
# 조회(notice_cache_repo)가 새로 붙으면서, 이 파일이 지키던 "실제 MongoDB 없이 라우트를
# 검증한다" 원칙이 notice_cache_repo 쪽에서 깨졌었다 — project_repo/document_repo처럼
# 자동으로 모든 테스트에 mock을 씌운다(전역 캐시는 항상 미스로 취급, 이 파일의 테스트
# 관심사가 아니므로).
@pytest.fixture(autouse=True)
def _patch_notice_cache(monkeypatch):
    async def _find_by_cache_key(cache_key, analysis_kind):
        return None

    async def _upsert(model):
        return None

    monkeypatch.setattr(documents_route.notice_cache_repo, "find_by_cache_key", _find_by_cache_key)
    monkeypatch.setattr(documents_route.notice_cache_repo, "upsert", _upsert)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responder, counter: list[int]):
        self._responder = responder
        self._counter = counter

    def create(self, *, model, messages, response_format=None, temperature=None):
        self._counter[0] += 1
        prompt = messages[0]["content"]
        return _FakeCompletion(self._responder(prompt))


class _FakeChat:
    def __init__(self, responder, counter):
        self.completions = _FakeCompletions(responder, counter)


def _make_fake_openai(responder, counter: list[int]):
    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = _FakeChat(responder, counter)

    return _FakeOpenAI


def _project(cache: dict | None = None) -> dict:
    return {"_id": "PROJ-1", "user_email": "tester@example.com", "application_form_analysis_cache": cache}


def _patch_project(monkeypatch, project: dict):
    async def _find_by_id_and_user(project_id, user_email):
        return project

    captured_updates = {}

    async def _update_project(project_id, update_data):
        captured_updates.update(update_data)
        project.update(update_data)
        return project

    monkeypatch.setattr(documents_route.project_repo, "find_by_id_and_user", _find_by_id_and_user)
    monkeypatch.setattr(documents_route.project_repo, "update_project", _update_project)
    return captured_updates


def _patch_documents(monkeypatch, docs: list[dict]):
    async def _find_by_project_id(project_id):
        return docs

    monkeypatch.setattr(documents_route.document_repo, "find_by_project_id", _find_by_project_id)


def test_no_criteria_document_skips_llm_and_returns_false(client, auth_header, monkeypatch):
    project = _project()
    _patch_project(monkeypatch, project)
    _patch_documents(monkeypatch, [])  # 문서 자체가 없음
    call_counter = [0]
    monkeypatch.setattr(documents_route, "OpenAI", _make_fake_openai(lambda p: "{}", call_counter))

    resp = client.post("/documents/PROJ-1/application-form-analysis", headers=auth_header)

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_application_form"] is False
    assert body["items"] == []
    assert call_counter[0] == 0, "문서가 없으면 LLM을 호출하지 않아야 한다"


def test_only_target_documents_present_skips_llm(client, auth_header, monkeypatch):
    """criteria 문서가 하나도 없고 target(평가 대상 문서/기획서)만 있으면 LLM을 호출하지
    않는다 — 업로드 영역 통합 후에도 role 필터링(criteria만 읽음)이 정확히 동작하는지 확인."""
    project = _project()
    _patch_project(monkeypatch, project)
    _patch_documents(
        monkeypatch,
        [{"document_role": "target", "parsed_text": "기획서 내용", "original_filename": "plan.pdf"}],
    )
    call_counter = [0]
    monkeypatch.setattr(documents_route, "OpenAI", _make_fake_openai(lambda p: "{}", call_counter))

    resp = client.post("/documents/PROJ-1/application-form-analysis", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["has_application_form"] is False
    assert call_counter[0] == 0


def test_extracts_items_from_shared_criteria_pool_and_caches_result(client, auth_header, monkeypatch):
    """가은/Claude(2026-07-22, 요청: 업로드 영역 통합) — 공고문과 신청서 양식이 같은
    document_role="criteria" 풀에 섞여 있어도(실제 EntryScreen 업로드 흐름 그대로) 항목
    추출이 정상 동작해야 한다."""
    project = _project()
    updates = _patch_project(monkeypatch, project)
    _patch_documents(
        monkeypatch,
        [
            {"document_role": "criteria", "parsed_text": "본 공모전은 실현가능성을 평가한다.", "original_filename": "공고문.pdf"},
            {"document_role": "criteria", "parsed_text": "1. 문제 정의(300자 이내)\n2. 차별성", "original_filename": "신청서.hwp"},
        ],
    )
    call_counter = [0]

    def _responder(prompt: str) -> str:
        assert "1. 문제 정의" in prompt and "실현가능성" in prompt  # 두 문서 원문이 모두 프롬프트에 들어갔는지
        return json.dumps(
            {
                "items": [
                    {"field_name": "문제 정의", "description": "해결하려는 문제", "char_limit": 300},
                    {"field_name": "차별성", "description": "", "char_limit": None},
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(documents_route, "OpenAI", _make_fake_openai(_responder, call_counter))

    resp = client.post("/documents/PROJ-1/application-form-analysis", headers=auth_header)

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_application_form"] is True
    assert call_counter[0] == 1
    assert [item["field_name"] for item in body["items"]] == ["문제 정의", "차별성"]
    assert body["items"][0]["char_limit"] == 300
    assert body["items"][1]["char_limit"] is None  # 양식에 없는 값을 지어내지 않는다
    assert body["source_document_names"] == ["공고문.pdf", "신청서.hwp"]
    # 캐시에 그대로 저장됐는지
    assert updates["application_form_analysis_cache"]["has_application_form"] is True


def test_criteria_documents_without_actual_form_yields_empty_items(client, auth_header, monkeypatch):
    """공고문만 올리고 실제 신청서 양식은 안 올렸어도(흔한 경우) has_application_form은
    True(문서는 있어 분석을 시도했으므로)지만 items는 빈 배열이어야 한다 — 신청서 양식이
    실제로 있었는지는 items로 판단한다(스키마 주석 참고)."""
    project = _project()
    _patch_project(monkeypatch, project)
    _patch_documents(
        monkeypatch,
        [{"document_role": "criteria", "parsed_text": "실현가능성과 차별성을 평가한다.", "original_filename": "공고문.pdf"}],
    )
    call_counter = [0]
    monkeypatch.setattr(
        documents_route, "OpenAI", _make_fake_openai(lambda p: json.dumps({"items": []}), call_counter)
    )

    resp = client.post("/documents/PROJ-1/application-form-analysis", headers=auth_header)

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_application_form"] is True
    assert body["items"] == []
    assert call_counter[0] == 1


def test_cached_result_skips_second_llm_call(client, auth_header, monkeypatch):
    cached = {
        "has_application_form": True,
        "items": [{"field_name": "이미 캐시된 항목", "description": "", "char_limit": None}],
        "source_document_names": ["old.hwp"],
        # 가은/Claude(2026-07-24): analysis_version이 없으면(레거시 캐시) 이제 의도적으로
        # 재계산한다 — 버전 없는 캐시를 "새 프롬프트로도 검증 안 된 오래된 값"으로 보고
        # 신뢰하지 않는 게 이번에 고친 버그였다. 이 테스트는 "제대로 캐싱된" 경우를
        # 검증하는 것이므로 버전을 채워서 캐시 히트 경로를 탄다.
        "analysis_version": documents_route._APPLICATION_FORM_ANALYSIS_CACHE_VERSION,
    }
    project = _project(cache=cached)
    _patch_project(monkeypatch, project)
    _patch_documents(monkeypatch, [{"document_role": "criteria", "parsed_text": "무시돼야 함", "original_filename": "new.hwp"}])
    call_counter = [0]
    monkeypatch.setattr(documents_route, "OpenAI", _make_fake_openai(lambda p: "{}", call_counter))

    resp = client.post("/documents/PROJ-1/application-form-analysis", headers=auth_header)

    assert resp.status_code == 200
    assert resp.json()["items"][0]["field_name"] == "이미 캐시된 항목"
    assert call_counter[0] == 0, "캐시가 있으면 LLM을 다시 호출하지 않아야 한다"


def test_all_items_kept_without_count_cap_only_invalid_entries_filtered(client, auth_header, monkeypatch):
    """가은/Claude(2026-07-22, 요청: 6개 상한 제거 — "신청서 항목은 제대로 들어가야지.
    말하는 주제가 신청서 항목 방향으로만 치우쳐지면 안 된다는 거였어"): 양식에 실제로
    있는 항목은 몇 개든 전부 담아야 한다 — 회의 주제 쏠림 방지는 discussion 프롬프트의
    [신청양식 참고 규칙]이 담당하므로, 여기서 개수를 줄이지 않는다. field_name이 빈
    항목만 유효성 검증으로 걸러낸다."""
    project = _project()
    _patch_project(monkeypatch, project)
    _patch_documents(monkeypatch, [{"document_role": "criteria", "parsed_text": "신청서 양식", "original_filename": "form.hwp"}])
    call_counter = [0]

    def _responder(prompt: str) -> str:
        items = [{"field_name": f"항목{i}", "description": "", "char_limit": None} for i in range(8)]
        items.append({"field_name": "", "description": "빈 이름은 제외돼야 함", "char_limit": None})
        return json.dumps({"items": items}, ensure_ascii=False)

    monkeypatch.setattr(documents_route, "OpenAI", _make_fake_openai(_responder, call_counter))

    resp = client.post("/documents/PROJ-1/application-form-analysis", headers=auth_header)

    body = resp.json()
    assert len(body["items"]) == 8, "6개 상한 없이 유효한 항목 8개가 전부 담겨야 한다"
    assert all(item["field_name"] for item in body["items"])
    assert [item["field_name"] for item in body["items"]] == [f"항목{i}" for i in range(8)]


def test_malformed_llm_response_does_not_crash(client, auth_header, monkeypatch):
    """LLM이 JSON이 아닌 응답을 주면 빈 items로 안전하게 처리한다(500이 아니라 200 +
    빈 배열) — announcement-analysis의 동일 실패 처리 정책과 같다."""
    project = _project()
    _patch_project(monkeypatch, project)
    _patch_documents(monkeypatch, [{"document_role": "criteria", "parsed_text": "신청서 양식", "original_filename": "form.hwp"}])
    call_counter = [0]
    monkeypatch.setattr(documents_route, "OpenAI", _make_fake_openai(lambda p: "이것은 JSON이 아닙니다", call_counter))

    resp = client.post("/documents/PROJ-1/application-form-analysis", headers=auth_header)

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_application_form"] is True  # 문서는 있었으므로 True
    assert body["items"] == []
