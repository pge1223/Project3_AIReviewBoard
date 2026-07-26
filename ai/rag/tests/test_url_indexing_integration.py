"""
실제 URL(외부 네트워크) + 실제 KURE-v1 모델을 사용하는 fetch-url 색인 파이프라인
통합 테스트
============================================================================
기본 pytest 실행에서는 수집되지 않도록(외부 네트워크 요청이 CI/기본 테스트 스위트에서
자동으로 나가지 않게) 모듈 최상단에서 RUN_URL_INTEGRATION 환경변수를 확인해 skip한다.

2026-07-18, "/documents/fetch-url이 project_id와 함께 오면 색인 단계에서 5분+ 멈춘다"는
버그 조사(용준/Claude) 중 만든 실제 repro다. 원인 조사 결과 요약은
ai/rag/retrieval/exceptions.py, diagnostics.py, backend/app/api/routes/documents.py의
_canonical_chroma_persist_dir()/_get_chroma_client() 주석 참고 — 확인된 가장 유력한
원인은 documents.py와 meetings.py가 같은 CHROMA_PERSIST_DIR을 서로 다른 identifier
문자열("./chroma_db" vs str(Path(...))=="chroma_db")로 열어 chromadb가 별개의 System
(=별개의 엔진/SQLite 연결)을 두 개 만들던 것 — 이 통합 테스트 자체는 그 픽스처의
결과물(실제 URL에서 만들어진 실제 청크/임베딩)이 정상적으로 색인되는지를 실제 KURE-v1로
한 번 더 확인하는 안전망이다(빠른 단위 테스트는 test_indexing_service.py의
TestShortKoreanWebpageIndexing이 fake 임베더로 이미 커버한다).

실행 (PowerShell):
    $env:RUN_URL_INTEGRATION="1"
    python -m pytest ai/rag/tests/test_url_indexing_integration.py -v -m url_integration

첫 실행은 huggingface_hub가 nlpai-lab/KURE-v1을 다운로드할 수 있고(캐시 없으면),
실제 리포트된 repro URL로 실제 HTTP 요청을 보낸다.
"""

import multiprocessing
import os
import traceback

import pytest

pytestmark = pytest.mark.url_integration

if os.environ.get("RUN_URL_INTEGRATION") != "1":
    pytest.skip(
        "RUN_URL_INTEGRATION=1일 때만 실행 (실제 URL 네트워크 요청 + 실제 KURE-v1 모델 로딩)",
        allow_module_level=True,
    )

from ai.rag.chunking.chunker import chunk_document
from ai.rag.chunking.schemas import ChunkSourceContext, SourceType
from ai.rag.domain import IndexingContext
from ai.rag.domain.config import DEFAULT_COLLECTION_NAME
from ai.rag.embedding.kure_embedder import KUREEmbedder
from ai.rag.embedding.schemas import EmbeddingConfig
from ai.rag.loaders.url_loader import load_from_url
from ai.rag.preprocessing.html_cleaner import clean_page_content
from ai.rag.retrieval.chroma_store import ChromaVectorStore, create_persistent_client
from ai.rag.retrieval.service import RAGIndexingService

# 버그 리포트에 첨부된 실제 repro URL (사업 공고 에필로그 결과보기 페이지, 첨부 2개:
# 포스터 이미지 + HWPX(미지원, 정상적으로 건너뜀))
_REPRO_URL = (
    "https://sotong.go.kr/front/epilogue/epilogueRsltViewPage.do"
    "?bbs_id=60caf1aeb54b45748c80d1652e706438&searchkey=&searchtxt=&miv_pageNo="
)

_COLLECTION = DEFAULT_COLLECTION_NAME

# 2026-07-18 PR 리뷰 지적사항 대응: 이 값이 선언만 되고 실제로 타임아웃에 적용되지
# 않아, hang이 재발하면 이 opt-in 테스트 자체도 무한 대기했다(단순 buggy no-op).
# 실제 색인을 별도 spawn 프로세스에서 실행한다. 시간 초과 시 부모 pytest 프로세스는
# 그대로 둔 채 자식만 terminate/kill하고 pytest.fail()로 정상적인 실패 결과를 남긴다.
# 스레드는 안전하게 강제 종료할 수 없고, os._exit()로 pytest 전체를 종료하면 JUnit 결과와
# fixture cleanup까지 유실되므로 프로세스 격리를 사용한다.
_TIMEOUT_SECONDS = 60
_PROCESS_SHUTDOWN_GRACE_SECONDS = 5


def _run_indexing(chroma_path: str, result_connection) -> None:
    try:
        result = load_from_url(_REPRO_URL)
        assert result.page_content is not None

        cleaned = clean_page_content(result.page_content)
        chunk_context = ChunkSourceContext(
            document_id="repro-real-url-1",
            source_type=SourceType.URL_WEBPAGE,
            source_url=_REPRO_URL,
            document_title=result.page_content.title or _REPRO_URL,
        )
        chunking_result = chunk_document(cleaned, chunk_context)
        assert chunking_result.chunk_count > 0

        embedder = KUREEmbedder(EmbeddingConfig(device="cpu"))
        client = create_persistent_client(path=chroma_path)
        vector_store = ChromaVectorStore(
            client=client,
            collection_name=_COLLECTION,
            embedding_model=embedder.model_name,
            embedding_dimension=embedder.embedding_dimension,
            embedding_version="embedding_v1",
        )
        service = RAGIndexingService(embedder, vector_store)
        context = IndexingContext(
            project_id="repro-real-url-project",
            document_id="repro-real-url-1",
            document_title=result.page_content.title or _REPRO_URL,
        )

        indexing_result = service.index_chunking_result(chunking_result, context)
        result_connection.send(("ok", indexing_result.status.value))
    except BaseException as exc:  # noqa: BLE001 — 자식 실패를 부모 pytest에 전달하기 위해 포착
        # 예외 객체 자체는 pickling이 불가능할 수 있어 문자열과 traceback만 전달한다.
        result_connection.send(("error", type(exc).__name__, str(exc), traceback.format_exc()))
    finally:
        result_connection.close()


class TestRealUrlIndexing:
    def test_repro_url_indexes_without_hanging(self, tmp_path):
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        worker = context.Process(
            target=_run_indexing,
            args=(str(tmp_path / "chroma_data"), child_connection),
            daemon=False,
        )
        worker.start()
        child_connection.close()
        worker.join(timeout=_TIMEOUT_SECONDS)

        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=_PROCESS_SHUTDOWN_GRACE_SECONDS)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=_PROCESS_SHUTDOWN_GRACE_SECONDS)
            parent_connection.close()
            pytest.fail(
                f"실URL 색인이 {_TIMEOUT_SECONDS}초 안에 끝나지 않아 자식 프로세스를 "
                "종료했습니다 (hang 재발 의심)."
            )

        if not parent_connection.poll(timeout=1):
            exitcode = worker.exitcode
            parent_connection.close()
            pytest.fail(f"색인 자식 프로세스가 결과 없이 종료됐습니다: exitcode={exitcode}")

        result = parent_connection.recv()
        parent_connection.close()
        if result[0] == "error":
            _, error_type, message, child_traceback = result
            pytest.fail(
                f"색인 자식 프로세스에서 {error_type}가 발생했습니다: "
                f"{message}\n{child_traceback}"
            )

        _, status = result
        assert worker.exitcode == 0
        assert status in ("success", "partial", "empty")
