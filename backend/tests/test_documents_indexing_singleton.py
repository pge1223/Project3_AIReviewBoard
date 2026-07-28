"""
Unit tests for the KUREEmbedder/chromadb.PersistentClient singleton wiring in
app.api.routes.documents._get_indexing_service() / _get_chroma_client() / _canonical_chroma_persist_dir().

2026-07-18, fetch-url 색인 5분+ hang 조사(용준/Claude) 중 확인한 두 가지를 회귀 테스트로 고정한다.

1) chromadb.PersistentClient(path=...)는 path 문자열 자체를 프로세스 전역 캐시
   (SharedSystemClient._identifier_to_system)의 key로 쓴다. meetings.py가 이전에
   chromadb.PersistentClient(path=str(Path(settings.CHROMA_PERSIST_DIR)))로 두 번째
   client를 만들고 있었는데, settings.CHROMA_PERSIST_DIR == "./chroma_db"일 때
   str(Path("./chroma_db")) == "chroma_db"라 documents.py가 쓰는 identifier
   "./chroma_db"와 달라진다 — 완전히 별개의 System(=별개의 SQLite/엔진 연결)이 같은
   물리 디렉터리에 서로 모른 채 동시 접근하게 됐다. Windows는 SQLite 파일 잠금이
   POSIX와 달리 mandatory라, 이 상태에서 한쪽이 쓰기 중일 때 다른 쪽이 파일을 열려고
   하면 즉시 에러 대신 무기한 대기로 이어질 수 있다(가장 유력한 hang 원인).
   -> 지금은 meetings.py가 documents.py의 싱글턴 client(_get_chroma_client())를 그대로
      재사용하도록 고쳤다(별도 client를 만들지 않음).

2) _get_indexing_service()의 지연 초기화(check-then-create)는 meetings.py가 앱 시작
   시점에(단일 스레드) 강제로 한 번 호출해줘서 지금은 실질적으로 레이스가 안 생기지만,
   그건 우연에 기댄 것이라 락을 추가했다 — 동시 첫 호출에도 KUREEmbedder/PersistentClient가
   한 번만 생성되어야 한다.

이 테스트들은 app.main(전체 FastAPI 앱, meetings.py 강제 임포트 포함)을 거치지 않고
app.api.routes.documents만 직접 import해서 실제 KURE-v1 모델 로딩이나 실제 chromadb
파일 접근, MongoDB 연결 없이 빠르게 돈다.
"""

import threading
from pathlib import Path

import pytest

import app.api.routes.documents as documents_module


@pytest.fixture
def reset_indexing_singleton():
    """다른 테스트 모듈이 이미 app.main을 import해서(그 과정에서 meetings.py가
    _get_indexing_service()를 강제 호출해서) 진짜 KUREEmbedder/chromadb client가 이미
    싱글턴에 들어있을 수 있다 — 테스트 전후로 저장/복원해 다른 테스트에 영향을 주지 않는다."""
    original = documents_module._indexing_service
    documents_module._indexing_service = None
    yield
    documents_module._indexing_service = original


class _FakeEmbedder:
    """실제 SentenceTransformer 로딩 없이 생성 횟수만 센다."""

    instances = 0
    last_config = None

    def __init__(self, config=None, *args, **kwargs):
        type(self).instances += 1
        type(self).last_config = config
        self.model_name = "fake-model"
        self.embedding_dimension = 4


class _FakeCollection:
    metadata: dict = {}

    def get(self, *args, **kwargs):
        return {"ids": []}


class _FakeChromaClient:
    """실제 chromadb 파일 접근 없이 생성 횟수만 센다."""

    instances = 0

    def __init__(self, *args, **kwargs):
        type(self).instances += 1

    def get_or_create_collection(self, *args, **kwargs):
        return _FakeCollection()


@pytest.fixture
def fake_dependencies(monkeypatch):
    _FakeEmbedder.instances = 0
    _FakeEmbedder.last_config = None
    _FakeChromaClient.instances = 0
    monkeypatch.setattr(documents_module, "KUREEmbedder", _FakeEmbedder)
    monkeypatch.setattr(documents_module.chromadb, "PersistentClient", _FakeChromaClient)
    return _FakeEmbedder, _FakeChromaClient


class TestSingletonConcurrency:
    def test_concurrent_first_calls_create_exactly_one_instance(
        self, reset_indexing_singleton, fake_dependencies
    ):
        fake_embedder_cls, fake_client_cls = fake_dependencies
        results: list[object] = []

        def _call():
            results.append(documents_module._get_indexing_service())

        threads = [threading.Thread(target=_call) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not any(t.is_alive() for t in threads), "일부 스레드가 10초 내에 끝나지 않았습니다"
        assert fake_embedder_cls.instances == 1, (
            "동시 첫 호출인데도 KUREEmbedder가 두 번 이상 생성됨 (TOCTOU 레이스 재발)"
        )
        assert fake_client_cls.instances == 1, (
            "동시 첫 호출인데도 chromadb.PersistentClient가 두 번 이상 생성됨"
        )
        assert len(results) == 16
        assert len({id(r) for r in results}) == 1, "스레드마다 서로 다른 RAGIndexingService 인스턴스를 받음"

    def test_second_call_reuses_existing_instance(self, reset_indexing_singleton, fake_dependencies):
        fake_embedder_cls, fake_client_cls = fake_dependencies

        first = documents_module._get_indexing_service()
        second = documents_module._get_indexing_service()

        assert first is second
        assert fake_embedder_cls.instances == 1
        assert fake_client_cls.instances == 1

    def test_embedding_runtime_settings_are_passed_to_embedder(
        self, reset_indexing_singleton, fake_dependencies, monkeypatch
    ):
        fake_embedder_cls, _ = fake_dependencies
        monkeypatch.setattr(documents_module.settings, "RAG_EMBEDDING_BATCH_SIZE", 16)
        monkeypatch.setattr(documents_module.settings, "RAG_TORCH_NUM_THREADS", 4)

        documents_module._get_indexing_service()

        assert fake_embedder_cls.last_config.batch_size == 16
        assert fake_embedder_cls.last_config.cpu_threads == 4


class TestCanonicalChromaPersistDir:
    def test_relative_path_is_resolved_from_repository_root(self, monkeypatch):
        monkeypatch.setattr(documents_module.settings, "CHROMA_PERSIST_DIR", "./chroma_db")
        monkeypatch.chdir(Path(__file__).parent)

        result1 = documents_module._canonical_chroma_persist_dir()
        result2 = documents_module._canonical_chroma_persist_dir()

        expected = (documents_module._REPOSITORY_ROOT / "chroma_db").resolve()
        assert result1 == result2 == str(expected)

    def test_absolute_path_is_preserved(self, monkeypatch):
        configured = documents_module._REPOSITORY_ROOT / ".test-custom-chroma"
        monkeypatch.setattr(documents_module.settings, "CHROMA_PERSIST_DIR", str(configured))

        assert documents_module._canonical_chroma_persist_dir() == str(configured.resolve())

    def test_relative_and_pathlib_normalized_forms_collide_before_fix(self):
        """회귀 방지용 증거: "./chroma_db"(원본 설정값)와 meetings.py가 예전에 쓰던
        str(Path("./chroma_db"))는 서로 다른 문자열이었다 — 이게 바로 chromadb
        SharedSystemClient가 서로 다른 identifier로 인식해 별개 System을 만들던 원인이다.
        지금은 documents.py/meetings.py 둘 다 _canonical_chroma_persist_dir()(절대경로)
        하나만 쓰므로 이 불일치 자체가 코드 경로에서 사라졌다 — 이 테스트는 왜 절대경로
        정규화가 필요했는지의 근거를 남겨둔다."""
        raw = "./chroma_db"
        old_meetings_style = str(Path(raw))
        assert raw != old_meetings_style  # "./chroma_db" != "chroma_db"


class TestGetChromaClientReusesSingleton:
    def test_returns_same_client_as_indexing_service_vector_store(
        self, reset_indexing_singleton, fake_dependencies
    ):
        _, fake_client_cls = fake_dependencies

        service = documents_module._get_indexing_service()
        client = documents_module._get_chroma_client()

        assert client is service.vector_store.client
        assert fake_client_cls.instances == 1, "meetings.py가 재사용해야 할 client가 중복 생성됨"


class TestMeetingsModuleDoesNotCreateDuplicateClient:
    def test_meetings_module_has_no_own_persistent_client_or_embedder(self):
        """meetings.py 소스에 chromadb.PersistentClient(...)나 KUREEmbedder()를 직접
        만드는 코드가 남아있지 않은지 정적으로 확인한다(2026-07-18, 중복 client/embedder
        제거 — documents.py의 싱글턴을 재사용하도록 고침)."""
        meetings_path = (
            Path(__file__).resolve().parents[2] / "backend" / "app" / "api" / "routes" / "meetings.py"
        )
        source = meetings_path.read_text(encoding="utf-8")
        live_lines = [
            line for line in source.splitlines() if not line.strip().startswith("#")
        ]
        live_source = "\n".join(live_lines)

        assert "chromadb.PersistentClient(" not in live_source
        assert "KUREEmbedder()" not in live_source
        assert "_get_chroma_client()" in live_source
        assert "_get_indexing_service().embedder" in live_source
