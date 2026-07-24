import asyncio
from datetime import date
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool
from jose import jwt, JWTError
from openai import OpenAI

from app.core.llm import trace_openai_client

from ai.rag.converters import (
    ConversionStatus,
    DocumentConversionError,
    build_conversion_metadata,
    cleanup_converted_file,
    convert_if_needed,
)
from ai.rag.loaders.url_loader import load_from_url
from ai.rag.loaders.schemas import UrlExtractionResult, WebPageContent
from ai.rag.loaders.exceptions import (
    InvalidUrlError,
    BlockedUrlError,
    UrlFetchError,
    TooManyRedirectsError,
    DownloadSizeLimitExceededError,
)
from ai.rag.preprocessing.html_cleaner import clean_page_content
from ai.rag.preprocessing.schemas import CleanedWebContent
from ai.rag.parsers import extract_document
from ai.rag.parsers.html_render import render_blocks_to_html
from ai.rag.converters.preview_pdf_converter import convert_to_preview_pdf
from ai.rag.converters.config import HwpConversionConfig
from ai.rag.converters.exceptions import DocumentConversionError
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from ai.rag.chunking.chunker import chunk_document
from ai.rag.chunking.schemas import ChunkSourceContext, SourceType
from ai.rag.domain.schemas import IndexingContext
from ai.rag.domain.config import DEFAULT_COLLECTION_NAME
from ai.rag.embedding.kure_embedder import KUREEmbedder
from ai.rag.retrieval.chroma_store import ChromaVectorStore
from ai.rag.retrieval.service import RAGIndexingService
from ai.rag.embedding.config import EMBEDDING_VERSION
import chromadb

from app.common.exceptions import BadRequestException, InternalServerException
from app.repositories.project_repository import ProjectRepository
from app.repositories.contest_work_repository import ContestWorkRepository
from app.config import settings
from app.models.document import DocumentModel
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    AnnouncementAnalysisResponse,
    AnnouncementEvidence,
    ApplicationFormAnalysisResponse,
    ApplicationFormItem,
    ContestWorkDetail,
    ContestWorksByTitleResponse,
    DocumentResponse,
    FetchUrlRequest,
    FetchUrlResponse,
    OfficialFacts,
    ScheduleItem,
    SimilarWork,
    StrategicAnalysis,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])
document_repo = DocumentRepository()
project_repo = ProjectRepository()
contest_work_repo = ContestWorkRepository()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_GENERIC_ERROR_MESSAGE = "URL 문서를 처리하는 중 오류가 발생했습니다."

_indexing_service: RAGIndexingService | None = None
# 가은/Claude(2026-07-15) 시점엔 없었던 락. 용준/Claude(2026-07-18, fetch-url 색인 hang
# 조사): 이 함수는 meetings.py 모듈 임포트 시점(앱 시작, 단일 스레드)에 이미 강제
# 호출되어 정상 상황에선 아래 if _indexing_service is None 체크가 레이스에 걸릴 일이
# 없다 — 하지만 그건 "meetings.py가 지금 이 함수를 호출한다"는 우연에 기대는 것이라,
# 나중에 그 강제 호출이 없어지면 여러 요청이 동시에 처음 호출할 때 KUREEmbedder(모델
# 로딩 비용 큼)가 여러 번 생성되는 TOCTOU 레이스가 생긴다. 방어적으로 락을 건다
# (락 경합은 최초 1회 초기화 이후엔 없음 — 매 요청마다 비용 없음).
_indexing_service_lock = threading.Lock()


def _canonical_chroma_persist_dir() -> str:
    """chromadb.PersistentClient(path=...)의 캐시 키(SharedSystemClient._identifier_to_system)는
    path 문자열을 있는 그대로 딕셔너리 key로 쓴다 — "./chroma_db"와 "chroma_db"처럼 같은
    디렉터리를 가리켜도 문자열이 다르면 완전히 별개의 System(=별개의 SQLite/엔진 연결)이
    두 개 생긴다. 실제로 meetings.py가 str(Path(settings.CHROMA_PERSIST_DIR))로 두 번째
    PersistentClient를 만들고 있었던 게 그 사례였다(2026-07-18, fetch-url 색인 hang 조사
    중 발견 — Windows는 SQLite 파일 잠금이 POSIX와 달리 mandatory라, 서로 모르는 두
    엔진이 같은 물리 파일에 동시 접근하면 즉시 에러 대신 무기한 대기로 이어질 수 있다).
    이 함수로 절대경로로 정규화해 항상 같은 identifier를 쓰도록 강제한다."""
    return str(Path(settings.CHROMA_PERSIST_DIR).resolve())


def _get_indexing_service() -> RAGIndexingService:
    """KUREEmbedder는 모델 로딩 비용이 커서, 첫 호출 시 한 번만 만들어 재사용한다
    (앱 시작 시점에 매번 로딩하지 않도록 지연 초기화). meetings.py 등 다른 모듈은 이
    싱글턴을 직접 재사용해야 한다 — 새로 chromadb.PersistentClient(path=...)나
    KUREEmbedder()를 만들지 말 것 (아래 _get_chroma_client() 참고)."""
    global _indexing_service
    if _indexing_service is None:
        with _indexing_service_lock:
            if _indexing_service is None:
                embedder = KUREEmbedder()
                client = chromadb.PersistentClient(path=_canonical_chroma_persist_dir())
                vector_store = ChromaVectorStore(
                    client=client,
                    collection_name=DEFAULT_COLLECTION_NAME,
                    embedding_model=embedder.model_name,
                    embedding_dimension=embedder.embedding_dimension,
                    embedding_version=EMBEDDING_VERSION,
                )
                _indexing_service = RAGIndexingService(embedder, vector_store)
    return _indexing_service


def _get_chroma_client() -> chromadb.ClientAPI:
    """documents.py 싱글턴이 쓰는 chromadb client를 그대로 반환한다. 같은
    CHROMA_PERSIST_DIR을 가리키는 별도의 chromadb.PersistentClient를 프로세스 안에
    또 만들지 않기 위함 — 다른 컬렉션(예: similar_cases)을 쓰더라도 client(=엔진 연결)
    자체는 공유해야 한다(2026-07-18, meetings.py 중복 PersistentClient 조사 참고)."""
    return _get_indexing_service().vector_store.client


def _parse_chunk_and_index(
    document_id: str, project_id: str, file_path: str, filename: str, document_role: str = "target"
) -> tuple[int, str, dict]:
    """RAG-001~003: 파싱 -> 청킹 -> 임베딩 -> Chroma 저장까지 동기적으로 실행하고
    (색인된 청크 수, 원문 전체 텍스트, 변환 metadata)를 반환한다. CPU-bound라 호출부에서
    threadpool로 감싸 실행해야 한다.
    가은/Claude(2026-07-15): 원문 텍스트도 같이 반환하도록 확장 — analyze_project()가
    submission.text로 쓸 "문서 전체 원문"이 이전엔 어디에도 저장되지 않았다(Chroma는
    벡터/청크 단위라 전체 원문 조회에는 안 맞음).
    가은/Claude(2026-07-16): 용준의 ai/rag/converters(HWP/HWPX -> PDF) 통합
    (ai/rag/converters/INTEGRATION.md 1번 권장 지점 그대로 따름) — HWP/HWPX만 변환해서
    처리용 PDF 경로를 만들고, 그 외 형식은 convert_if_needed()가 None을 반환해 기존 경로를
    그대로 탄다. chunk_context/indexing_context의 filename은 원본 그대로 써서(파라미터
    file_path만 처리용 경로로 바뀜) 색인 메타데이터에 변환 파일명이 노출되지 않는다.
    변환 실패(DocumentConversionError)는 청킹/임베딩 없이 그대로 전파 — 호출부가
    conversion_status=failed 처리를 하도록."""
    source_path = Path(file_path)
    conversion_result = None
    processing_path = source_path
    conversion_metadata: dict = {
        "original_file_type": source_path.suffix.lstrip(".").lower(),
        "processing_file_type": source_path.suffix.lstrip(".").lower(),
        "conversion_status": ConversionStatus.NOT_REQUIRED.value,
        "conversion_error": None,
        "converter_name": None,
        "conversion_duration_ms": None,
    }

    try:
        conversion_result = convert_if_needed(source_path)
        if conversion_result is not None:
            processing_path = conversion_result.converted_path
            conversion_metadata = build_conversion_metadata(conversion_result).model_dump(mode="json")

        extraction = extract_document(processing_path)
        chunk_context = ChunkSourceContext(
            document_id=document_id,
            source_type=SourceType.FILE_UPLOAD,
            source_filename=filename,
        )
        chunking_result = chunk_document(extraction, chunk_context)

        indexing_context = IndexingContext(
            project_id=project_id,
            document_id=document_id,
            document_title=filename,
            document_role=document_role,
        )
        summary = _get_indexing_service().index_chunking_result_with_summary(chunking_result, indexing_context)
        parsed_text = "\n\n".join(block.content for block in extraction.blocks)
        return summary.stored_count, parsed_text, conversion_metadata
    finally:
        cleanup_converted_file(conversion_result)


def _chunk_and_index_webpage(
    document_id: str, project_id: str, url: str, title: str, cleaned: CleanedWebContent
) -> int:
    """가은/Claude(2026-07-15, "다 이어버리자" — 용준 확인 필요): fetch-url이 정제까지만
    하고 멈추던 걸 파일 업로드와 똑같이 청킹/색인까지 잇는다. chunk_document()가 원래부터
    CleanedWebContent를 받도록 설계돼 있어서(ai/rag/chunking/chunker.py) 새 파서/청커를
    만들 필요는 없었다 — SourceType만 URL_WEBPAGE로 바꿔서 그대로 재사용."""
    chunk_context = ChunkSourceContext(
        document_id=document_id,
        source_type=SourceType.URL_WEBPAGE,
        source_url=url,
        document_title=title,
    )
    chunking_result = chunk_document(cleaned, chunk_context)

    indexing_context = IndexingContext(
        project_id=project_id,
        document_id=document_id,
        document_title=title,
        document_role="criteria",
    )
    summary = _get_indexing_service().index_chunking_result_with_summary(chunking_result, indexing_context)
    return summary.stored_count


# 가은/Claude(2026-07-19, INF-007): 공고문 색인(청킹+임베딩+Chroma 저장)이 끝날 때까지
# fetch-url 응답 자체를 막고 있었다 — 정상 케이스도 수 초~수십 초가 걸리고, 예전엔 hang
# 버그(용준/Claude 2026-07-18, Chroma 이중 client 문제 — 지금은 고쳐짐)로 5분+ 무응답도
# 실측됐다. 근본 hang 원인은 고쳐졌지만, "색인이 오래 걸리는 것 자체"와 "타임아웃이
# 아예 없다"는 별개 문제라 — 색인을 백그라운드로 넘기고 타임아웃을 강제한다.
# GET /{project_id}/{document_id}/status(기존 DOC-004 엔드포인트)를 프론트가 폴링해서
# 완료 여부를 확인한다. 아래 상수는 그 백그라운드 색인의 최대 대기 시간(윤한/Claude
# 2026-07-18) — 재발 시 요청이 무기한 걸려있지 않도록 하는 안전장치다.
_WEBPAGE_INDEXING_TIMEOUT_SECONDS = 120


async def _index_webpage_background(
    *,
    document_id: str,
    project_id: str,
    url: str,
    title: str,
    cleaned: CleanedWebContent,
) -> None:
    """색인을 백그라운드로 돌리고 끝나면 documents 컬렉션의 status를 patch한다
    (meetings.py의 _synthesize_chair_background()와 동일 패턴). asyncio.wait_for로
    타임아웃을 걸지만, run_in_threadpool로 넘긴 실제 스레드 자체를 강제 종료하지는
    못한다 — 진짜 hang이면 그 스레드는 백그라운드에 남아있게 되고, 여기선 "기다리는 걸
    포기하고 실패로 기록"까지만 보장한다(요청/폴링 쪽을 무한 대기에서 풀어주는 게 목적)."""
    _index_started = time.time()
    logger.info("[fetch-url] 색인(백그라운드) 시작 document_id=%s url=%s", document_id, url)
    try:
        stored_count = await asyncio.wait_for(
            run_in_threadpool(_chunk_and_index_webpage, document_id, project_id, url, title, cleaned),
            timeout=_WEBPAGE_INDEXING_TIMEOUT_SECONDS,
        )
        status_value = "indexed" if stored_count > 0 else "indexed_empty"
        logger.info(
            "[fetch-url] === 색인(백그라운드) 완료 === document_id=%s elapsed=%.1fs stored_count=%d",
            document_id,
            time.time() - _index_started,
            stored_count,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[fetch-url] 색인(백그라운드) 타임아웃: document_id=%s (%ds 초과) elapsed=%.1fs",
            document_id,
            _WEBPAGE_INDEXING_TIMEOUT_SECONDS,
            time.time() - _index_started,
        )
        status_value = "indexing_timeout"
    except Exception:
        logger.exception(
            "[fetch-url] 색인(백그라운드) 실패: document_id=%s elapsed=%.1fs",
            document_id,
            time.time() - _index_started,
        )
        status_value = "indexing_failed"
    await document_repo.update_fields(document_id, {"status": status_value})


# 가은/Claude(2026-07-21): 실측 요청 — 파일 업로드(평가 대상 기획서 포함)가 느리다.
# 원인은 fetch-url(INF-007)과 달리 파일 업로드는 파싱→청킹→임베딩→Chroma 색인을 응답
# 전에 동기로 다 끝내고 있어서다. 같은 패턴으로 색인을 백그라운드로 넘긴다 — 업로드
# 응답은 즉시(status="indexing") 돌아가고, 프론트는 기존 status 엔드포인트(DOC-004)를
# 폴링한다. 타임아웃은 웹페이지보다 여유 있게 둔다(HWP→PDF 변환 + 대용량 파일 고려).
_FILE_INDEXING_TIMEOUT_SECONDS = 180


async def _index_file_background(
    *,
    document_id: str,
    project_id: str,
    file_path: str,
    filename: str,
    document_role: str,
) -> None:
    """파일 파싱+색인을 백그라운드로 돌리고 끝나면 documents의 status/parsed_text/
    conversion_metadata를 patch한다(_index_webpage_background와 동일 패턴). 동기 시절
    응답에 실어 보내던 HWP/HWPX 변환 실패(user_message)도 여기서 conversion_metadata에
    저장한다 — 프론트는 status 폴링 응답의 conversion_metadata로 같은 안내를 보여준다."""
    _index_started = time.time()
    logger.info("[upload] 색인(백그라운드) 시작 document_id=%s filename=%s", document_id, filename)
    try:
        stored_count, parsed_text, conversion_metadata = await asyncio.wait_for(
            run_in_threadpool(_parse_chunk_and_index, document_id, project_id, file_path, filename, document_role),
            timeout=_FILE_INDEXING_TIMEOUT_SECONDS,
        )
        await document_repo.update_fields(
            document_id,
            {
                "status": "indexed" if stored_count > 0 else "indexed_empty",
                "parsed_text": parsed_text,
                "conversion_metadata": conversion_metadata,
            },
        )
        logger.info(
            "[upload] === 색인(백그라운드) 완료 === document_id=%s elapsed=%.1fs stored_count=%d",
            document_id,
            time.time() - _index_started,
            stored_count,
        )
        return
    except asyncio.TimeoutError:
        logger.warning(
            "[upload] 색인(백그라운드) 타임아웃: document_id=%s (%ds 초과) elapsed=%.1fs",
            document_id,
            _FILE_INDEXING_TIMEOUT_SECONDS,
            time.time() - _index_started,
        )
        await document_repo.update_fields(document_id, {"status": "indexing_timeout"})
    except DocumentConversionError as exc:
        logger.warning("[upload] 문서 변환 실패(백그라운드): document_id=%s error=%s", document_id, exc)
        await document_repo.update_fields(
            document_id,
            {
                "status": "conversion_failed",
                "conversion_metadata": {
                    "original_file_type": os.path.splitext(filename)[1].lstrip(".").lower(),
                    "processing_file_type": None,
                    "conversion_status": "failed",
                    "conversion_error": exc.user_message,
                    "converter_name": None,
                    "conversion_duration_ms": None,
                },
            },
        )
    except Exception:
        logger.exception(
            "[upload] 색인(백그라운드) 실패: document_id=%s elapsed=%.1fs",
            document_id,
            time.time() - _index_started,
        )
        await document_repo.update_fields(document_id, {"status": "indexing_failed"})


# 가은/Claude (2026-07-15): 비회원 로그인은 Authorization 헤더 없이 그대로 들어온다 —
# 헤더가 없으면 401 대신 고정 게스트 사용자로 통과시킨다 (projects.py와 동일 컨벤션).
GUEST_USER_EMAIL = "guest@local"


def get_current_user(authorization: Optional[str]) -> str:
    if not authorization:
        return GUEST_USER_EMAIL
    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")


async def verify_project_owner(project_id: str, user_email: str):
    project = await project_repo.find_by_id_and_user(project_id, user_email)
    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")
    return project


def _apply_cleaning(page_content: WebPageContent) -> tuple[WebPageContent, CleanedWebContent]:
    """clean_page_content()는 CleanedWebContent(title/fetched_at/encoding 없음)를 반환하므로,
    원본 WebPageContent의 title/fetched_at/encoding/is_js_rendered_suspected는 그대로 두고
    blocks/text/text_length만 정제 결과로 교체한 새 WebPageContent를 만든다
    (기존 UrlExtractionResult.page_content 응답 계약 유지).
    가은/Claude(2026-07-15): 원본 CleanedWebContent도 같이 반환하도록 확장 — chunk_document()가
    바로 이 타입을 받도록 설계돼 있어서(merge된 WebPageContent가 아니라) 색인하려면 필요하다."""
    cleaned = clean_page_content(page_content)
    cleaned_text = "\n\n".join(block.content for block in cleaned.cleaned_blocks)
    merged = page_content.model_copy(update={
        "blocks": cleaned.cleaned_blocks,
        "text": cleaned_text,
        "text_length": len(cleaned_text),
    })
    return merged, cleaned


# DOC-004: URL 문서 수집
# 가은/Claude(2026-07-19, INF-007): 색인(청킹+임베딩)이 끝날 때까지 이 응답 자체를 막지
# 않는다 — project_id가 있으면 document_id를 즉시 만들어 반환하고, 색인은
# _index_webpage_background()로 넘긴다. 프론트는 응답에 담긴 page_content/attachments/
# warnings로 문서 행을 바로 그리고, document_status가 "indexing"이면
# GET /{project_id}/{document_id}/status(DOC-004, 기존 엔드포인트)를 폴링한다.
@router.post("/fetch-url", response_model=FetchUrlResponse)
async def fetch_url(
    request: FetchUrlRequest,
    authorization: Optional[str] = Header(None, alias="authorization"),
) -> FetchUrlResponse:
    user_email = get_current_user(authorization)

    try:
        result = await run_in_threadpool(load_from_url, request.url)
    except (InvalidUrlError, BlockedUrlError, TooManyRedirectsError) as exc:
        raise BadRequestException(detail=str(exc)) from exc
    except (UrlFetchError, DownloadSizeLimitExceededError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except Exception:
        logger.exception("URL 문서 수집 중 예상하지 못한 오류가 발생했습니다: url=%s", request.url)
        raise InternalServerException(detail=_GENERIC_ERROR_MESSAGE)

    document_id: Optional[str] = None
    document_status: Optional[str] = None

    # 데모 파이프라인 점검용 — fetch-url은 지금까지 실패(예외)만 로그에 남았다. 응답이
    # 200이어도 본문을 못 찾았거나(page_content=None) 첨부파일이 다 미지원/실패로
    # 빠졌을 수 있으므로, 성공 경로도 한 줄 남겨 "URL에서 실제로 뭘 가져왔는지"를 바로
    # 확인할 수 있게 한다.
    logger.info(
        "[fetch-url] 수집 완료 url=%s target_type=%s page_content=%s attachments=%d unsupported=%d failed=%d",
        request.url,
        result.fetch_target_type,
        "있음" if result.page_content is not None else "없음",
        len(result.attachments),
        len(result.unsupported_attachments),
        len(result.failed_attachments),
    )

    if result.page_content is not None:
        try:
            merged_page_content, cleaned = await run_in_threadpool(_apply_cleaning, result.page_content)
        except Exception:
            logger.exception("HTML 정제 중 예상하지 못한 오류가 발생했습니다: url=%s", request.url)
            raise InternalServerException(detail=_GENERIC_ERROR_MESSAGE)
        result = result.model_copy(update={"page_content": merged_page_content})

        # 가은/Claude(2026-07-15, "다 이어버리자"): project_id가 있으면 공고문도 기획서
        # 업로드와 동일하게 색인 + documents 컬렉션 저장까지 잇는다. 없으면(과거 호출
        # 호환) 조회만 하고 끝낸다 — 프론트가 project_id를 꼭 보내도록 같이 바꿨다.
        # 윤한/Claude(2026-07-18, INF-007): 색인을 이 요청 안에서 await하면 요청이 색인
        # 소요 시간만큼(수십 초~) 붙잡혀 있었다 — status="indexing"으로 저장만 해두고
        # asyncio.create_task()로 넘긴 뒤 즉시 응답한다(meetings.py의 위원장 종합 백그라운드
        # 이관과 동일 패턴). 프론트는 document_status="indexing"이면 document_id로
        # /{project_id}/{document_id}/status를 폴링해 최종 상태를 확인해야 한다.
        if request.project_id:
            title = merged_page_content.title or request.url
            document = DocumentModel(
                project_id=request.project_id,
                user_email=user_email,
                original_filename=title,
                stored_filename=request.url,
                file_path=request.url,
                file_size=merged_page_content.text_length,
                mime_type="text/html",
                status="indexing",
                source_type="url",
                document_role="criteria",
                parsed_text=merged_page_content.text,
                # 가은/Claude(2026-07-18): 실측(sotong.go.kr) — 평가기준이 본문이 아니라
                # HWP 요강 파일에만 있는 공고가 실제로 있었다. 재접속 후에도(getDocuments())
                # "직접 받아서 올려주세요" 안내를 다시 보여줄 수 있게 저장해둔다.
                unsupported_attachments=[
                    {"url": a.url, "file_name": a.file_name, "reason": a.reason}
                    for a in result.unsupported_attachments
                ]
                or None,
            )
            document_id = await document_repo.create(document)
            document_status = "indexing"
            asyncio.create_task(
                _index_webpage_background(
                    document_id=document_id,
                    project_id=request.project_id,
                    url=request.url,
                    title=title,
                    cleaned=cleaned,
                )
            )

    return FetchUrlResponse(**result.model_dump(), document_id=document_id, document_status=document_status)


# DOC-001: 문서 업로드
@router.post("/{project_id}", response_model=DocumentResponse)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    source_type: str = Form("pdf"),
    # 가은/Claude(2026-07-15): "target"(평가 대상 문서/기획서, 기본값 — 기존 호출 호환) |
    # "criteria"(공고문 파일 업로드 탭). DocumentUploadPage.jsx의 두 드롭존과 대응된다.
    document_role: str = Form("target"),
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    # DOC-005: 업로드 파일 검증
    # HWP/HWPX는 ai/rag/converters(#45)가 내부적으로 PDF로 변환해 처리한다
    # (ai/rag/converters/INTEGRATION.md 5번 참고) — 화이트리스트에도 포함해야 한다.
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".hwp", ".hwpx"}
    ALLOWED_MIME_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/x-hwp",
        "application/haansofthwp",
        "application/vnd.hancom.hwp",
        "application/haansofthwpx",
        "application/vnd.hancom.hwpx",
    }
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise BadRequestException(detail=f"지원하지 않는 파일 형식입니다. 허용: PDF, DOCX, PPTX, HWP, HWPX")

    # HWP/HWPX는 OS/브라우저에 등록된 표준 MIME이 없어 대부분 application/octet-stream으로
    # 전송된다 — 확장자 화이트리스트를 이미 통과했으므로 이 두 확장자는 MIME 검사를 건너뛴다.
    if file_ext not in {".hwp", ".hwpx"} and file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise BadRequestException(detail=f"지원하지 않는 MIME 타입입니다: {file.content_type}")

    content = await file.read()
    if len(content) == 0:
        raise BadRequestException(detail="빈 파일은 업로드할 수 없습니다")
    if len(content) > MAX_FILE_SIZE:
        raise BadRequestException(detail=f"파일 크기가 너무 큽니다. 최대 50MB까지 허용됩니다")

    stored_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    # DOC-002: 문서 메타데이터 저장
    document = DocumentModel(
        project_id=project_id,
        user_email=user_email,
        original_filename=file.filename,
        stored_filename=stored_filename,
        file_path=file_path,
        file_size=len(content),
        mime_type=file.content_type or "application/octet-stream",
        source_type=source_type,
        document_role=document_role,
        status="indexing",
    )

    result = await document_repo.create(document)

    # RAG-001~003: 파싱 -> 청킹 -> 임베딩 -> Chroma 색인.
    # 가은/Claude(2026-07-21): 색인이 끝날 때까지 응답을 막지 않는다(INF-007과 동일 패턴,
    # 위 _index_file_background 주석 참고) — status="indexing"으로 즉시 응답하고, 프론트가
    # GET /{project_id}/{document_id}/status(DOC-004)를 폴링해 완료를 확인한다. HWP/HWPX
    # 변환 실패(DocumentConversionError.user_message)도 이제 폴링 응답의
    # conversion_metadata로 전달된다.
    asyncio.create_task(
        _index_file_background(
            document_id=result,
            project_id=project_id,
            file_path=file_path,
            filename=file.filename,
            document_role=document_role,
        )
    )

    return DocumentResponse(
        id=result,
        project_id=document.project_id,
        user_email=document.user_email,
        original_filename=document.original_filename,
        stored_filename=document.stored_filename,
        file_path=document.file_path,
        file_size=document.file_size,
        mime_type=document.mime_type,
        source_type=document.source_type,
        status=document.status,
        created_at=document.created_at,
        updated_at=document.updated_at,
        document_role=document.document_role,
        conversion_metadata=document.conversion_metadata,
        unsupported_attachments=document.unsupported_attachments,
    )


# 가은/Claude(2026-07-21): 실측 요청 — "수상작·유사사례 경향" 카드에서 항목을 클릭하면
# 같은 공모전(contest_title)의 다른 수상작/후보작을 옆 패널에서 더 보여준다. contest_works는
# 프로젝트 소유권과 무관한 공개 아카이브라 project_id 없이 로그인 여부만 확인한다.
# 주의: "/{project_id}" GET(DOC-003, 바로 아래)과 둘 다 단일 경로 세그먼트라 FastAPI는
# 등록 순서로 매칭한다 — 이 라우트가 반드시 그보다 먼저 등록돼야
# "/documents/contest-works"가 project_id="contest-works"로 잘못 매칭되지 않는다.
@router.get("/contest-works", response_model=ContestWorksByTitleResponse)
async def get_contest_works_by_title(
    contest_title: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    get_current_user(authorization)
    docs = await contest_work_repo.find_by_contest_title(contest_title)
    works = [
        ContestWorkDetail(
            work_title=str(doc.get("work_title") or "").strip(),
            award_grade=str(doc.get("award_grade") or "").strip(),
            selection_status=str(doc.get("selection_status") or ""),
            images=[img.get("url") for img in (doc.get("images") or []) if img.get("url")],
            ocr_text=str(doc.get("ocr_text") or ""),
            source_url=str(doc.get("source_url") or ""),
        )
        for doc in docs
    ]
    return ContestWorksByTitleResponse(contest_title=contest_title, works=works)


# DOC-003: 프로젝트 문서 목록 조회
@router.get("/{project_id}", response_model=list[DocumentResponse])
async def get_documents(
    project_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)
    documents = await document_repo.find_by_project_id(project_id)

    return [
        DocumentResponse(
            id=str(d["_id"]),
            project_id=d["project_id"],
            user_email=d["user_email"],
            original_filename=d["original_filename"],
            stored_filename=d["stored_filename"],
            file_path=d["file_path"],
            file_size=d["file_size"],
            mime_type=d["mime_type"],
            source_type=d["source_type"],
            status=d["status"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            document_role=d.get("document_role", "target"),
            conversion_metadata=d.get("conversion_metadata"),
            unsupported_attachments=d.get("unsupported_attachments"),
        )
        for d in documents
    ]


# 가은/Claude(2026-07-21): 실측 요청 — /board에서 URL/파일로 올린 공고문·평가기준
# 문서를 잘못 올렸을 때 지울 수 있게. PRJ-004(프로젝트 전체 삭제, projects.py)와
# 같은 순서(Chroma 벡터 청크 -> MongoDB 문서 레코드)를 문서 1건 단위로 좁힌 것 — 벡터
# 삭제는 ChromaVectorStore.delete_document()를 직접 쓴다(RAGIndexingService는
# delete_project만 감싸고 있어 서비스 계층에 새 메서드를 추가하는 대신 documents.py
# 안에서 기존 _get_indexing_service() 싱글턴을 그대로 재사용).
@router.delete("/{project_id}/{document_id}")
async def delete_document(
    project_id: str,
    document_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    document = await document_repo.find_by_id(document_id)
    if not document or document.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")

    indexing_service = _get_indexing_service()
    await run_in_threadpool(indexing_service.vector_store.delete_document, project_id, document_id)
    await document_repo.delete_by_id(document_id)
    return {"message": "문서가 삭제되었습니다"}


# DOC-004: 문서 처리 상태 조회
@router.get("/{project_id}/{document_id}/status")
async def get_document_status(
    project_id: str,
    document_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    document = await document_repo.find_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")

    return {
        "document_id": str(document["_id"]),
        "project_id": document["project_id"],
        "original_filename": document["original_filename"],
        "status": document["status"],
        "updated_at": document["updated_at"],
        # 가은/Claude(2026-07-21): 파일 업로드 색인이 백그라운드로 바뀌면서(위 upload_document
        # 참고) HWP/HWPX 변환 실패 안내(user_message)를 업로드 응답에 실을 수 없게 됐다 —
        # 폴링하는 프론트가 여기서 읽는다. 순수 추가 필드(기존 폴링 클라이언트는 무시하면 그대로 동작).
        "conversion_metadata": document.get("conversion_metadata"),
    }


# DOC-006: 문서 미리보기
@router.get("/{project_id}/{document_id}/preview")
async def preview_document(
    project_id: str,
    document_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    document = await document_repo.find_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")

    return {
        "original_filename": document["original_filename"],
        "parsed_text": document.get("parsed_text"),
        "status": document["status"],
    }


# 재인/Claude(2026-07-21): "AI 피드백" 워크벤치가 기획서를 워드/한글 원본처럼(굵게·
# 기울임 서식 살려서) 보여주기 위해 추가 - 기존 /preview(parsed_text, 순수 텍스트)는
# 그대로 두고 완전히 새 엔드포인트만 추가한다. parsed_text는 업로드 시점에 DB에
# 저장해두지만 서식 정보(블록 metadata["runs"])는 저장하지 않으므로, 여기서는
# document["file_path"](업로드된 원본 파일, 업로드 후에도 uploads/에 그대로 남음)를
# 요청마다 다시 파싱해서 HTML로 변환한다 - DB 스키마나 업로드 흐름(_parse_chunk_and_index)은
# 하나도 안 건드림. docx는 ai/rag/parsers/docx_parser.py가 담은 runs 정보로 굵게/기울임이
# 살아나오고, 그 외 형식(HWP 등 PDF로 변환되는 경로)은 runs가 없어 일반 문단으로만
# 나온다(html_render.render_blocks_to_html의 폴백) - 크래시 없이 항상 뭔가는 반환됨.
def _render_document_html(file_path: str) -> str:
    source_path = Path(file_path)
    conversion_result = None
    try:
        conversion_result = convert_if_needed(source_path)
        processing_path = conversion_result.converted_path if conversion_result else source_path
        extraction = extract_document(processing_path)
        return render_blocks_to_html(extraction.blocks)
    finally:
        cleanup_converted_file(conversion_result)


@router.get("/{project_id}/{document_id}/preview-html")
async def preview_document_html(
    project_id: str,
    document_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    document = await document_repo.find_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")

    file_path = document.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="원본 파일을 찾을 수 없습니다 (다시 업로드해주세요)")

    try:
        html = await run_in_threadpool(_render_document_html, file_path)
    except Exception:
        logger.exception("[PREVIEW_HTML_ERROR] document_id=%s 서식 변환 실패", document_id)
        raise HTTPException(status_code=500, detail="원문을 서식과 함께 불러오지 못했습니다")

    return {
        "original_filename": document["original_filename"],
        "html": html,
    }


# 재인/Claude(2026-07-21): "AI 피드백" 워크벤치가 기획서를 워드/한글 원본과 완전히 같은
# 페이지 모습(줄바꿈·여백까지)으로 보여주기 위해 추가 - HTML 재구성(위 /preview-html)만으로는
# docx가 실제로 몇 페이지에서 어떻게 줄바꿈되는지는 재현할 수 없다(그 정보는 원본 파일에
# 없고 워드 같은 렌더러가 그릴 때 그때그때 계산하는 값이라서). 그래서 LibreOffice로 원본을
# PDF로 변환해 그대로 내려주고, 프론트가 pdf.js로 그 PDF를 그린다 - 페이지 레이아웃까지
# 원본 그대로 보장되는 유일한 방법. hwp/hwpx는 이미 이 방식(LibreOffice 변환)을 RAG
# 색인에서도 쓰고 있어 같은 인프라(서버에 LibreOffice 필요)를 재사용하는 셈이다.
@router.get("/{project_id}/{document_id}/preview-pdf")
async def preview_document_pdf(
    project_id: str,
    document_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    document = await document_repo.find_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")

    file_path = document.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="원본 파일을 찾을 수 없습니다 (다시 업로드해주세요)")

    if Path(file_path).suffix.lower() == ".pdf":
        return FileResponse(file_path, media_type="application/pdf")

    output_dir = HwpConversionConfig().resolve_temp_dir() / "preview_pdf"
    try:
        pdf_path = await run_in_threadpool(
            convert_to_preview_pdf, Path(file_path), output_dir=output_dir
        )
    except DocumentConversionError:
        logger.exception("[PREVIEW_PDF_ERROR] document_id=%s 미리보기 PDF 변환 실패", document_id)
        raise HTTPException(status_code=500, detail="원문을 페이지 형태로 불러오지 못했습니다")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        background=BackgroundTask(lambda: pdf_path.unlink(missing_ok=True)),
    )


# 가은/Claude(2026-07-21): "공모전 분석" 화면 실제 데이터 연결 — 실측 제보(사용자,
# 2026-07-21) "URL 넣고 분석 시작 눌렀는데 예시 카드만 나온다"에 대한 대응이자,
# 팀 UX 스펙(사실/전략분석/근거를 분리하고 근거 없는 내용은 지어내지 않는다)을 그대로
# 구현한다. 이미 수집된 criteria 문서(URL/파일, document_role="criteria")의 parsed_text를
# 그대로 근거로 쓴다 — 새 RAG 파이프라인이나 청크 검색은 쓰지 않고 원문 전체(길면 앞부분)를
# 한 번에 LLM에 넣는 단순한 1회성 호출이라 mentor-candidates(get_mentor_candidates)와
# 같은 패턴을 그대로 따른다.
_ANNOUNCEMENT_TRUNCATE_CHARS = 16000
# 가은/Claude(2026-07-23): 재검증 로직을 바꿀 때마다(v3: temperature=0 안정화, v4: "20점"
# 재발 프롬프트 패치, v5: 재검증을 인용-대조 방식으로 재설계) 기존 캐시엔 반영이 안 되므로
# 버전을 올려 강제로 재계산한다.
_ANNOUNCEMENT_ANALYSIS_CACHE_VERSION = 6

# 가은/Claude(2026-07-21): scripts/classify_contest_works.py가 contest_works 문서에
# 붙인 category와 같은 8개 taxonomy — 이 공고문을 같은 기준으로 분류해야 contest_works를
# category로 매칭 조회할 수 있다. 두 목록은 반드시 동일하게 유지할 것(한쪽만 바꾸면
# 매칭이 조용히 0건이 된다).
_CONTEST_CATEGORIES = [
    "AI/데이터", "공공서비스", "환경/기후", "교육/연구",
    "복지/사회", "안전/재난", "창업/경제", "기타",
]


async def _load_criteria_documents_text(project_id: str) -> tuple[str, list[str]]:
    documents = await document_repo.find_by_project_id(project_id)
    criteria_docs = [
        d
        for d in documents
        if d.get("document_role", "target") == "criteria" and d.get("parsed_text")
    ]
    # URL 본문과 첨부 PDF를 함께 넣을 때 출처 경계가 없으면 LLM이 짧은 URL 요약만
    # 대표 공고로 오인해 PDF 표/부록을 건너뛸 수 있다. 각 원문에 이름을 붙여 서로
    # 독립된 근거임을 명확히 하고, 현재 모델 문맥 안에서 둘 다 충분히 읽도록 한다.
    sections = [
        f"[출처 문서: {d.get('original_filename') or '이름 없음'}]\n{d['parsed_text']}"
        for d in criteria_docs
    ]
    combined = "\n\n---\n\n".join(sections)
    names = [d.get("original_filename") or "이름 없음" for d in criteria_docs]
    return combined, names


# 가은/Claude(2026-07-22, 요청: "공모전 공고·평가기준·신청서 양식을 같은 업로드 영역으로
# 합치자"): 신청양식 전용 document_role/업로드 카드를 따로 두지 않는다 — 사용자가 이
# 화면에서 올리는 문서는 공고문이든 평가기준이든 신청서 양식이든 전부 document_role
# ="criteria" 하나로 저장되고(EntryScreen의 단일 업로드 영역, 안내 문구만 "공모전 공고,
# 평가기준, 신청서 양식"으로 확장), announcement-analysis와 application-form-analysis가
# 같은 문서 풀(_load_criteria_documents_text)을 각자의 관점으로 다시 읽는다 — 신청서
# 양식을 안 올렸으면(또는 올린 문서 중 실제 신청서 양식이 없으면) 프롬프트 규칙에 따라
# items가 자연히 빈 배열이 된다(지어내지 않는다).
#
# 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입): 신청양식 원문에서 실제로 기입해야
# 하는 항목만 뽑는다. announcement-analysis와 원칙이 같다 — 양식에 없는 항목·분량 제한을
# 지어내지 않는다(char_limit은 명시가 없으면 null).
#
# 가은/Claude(2026-07-22, 요청: 상한 제거 — "신청서 항목은 제대로 들어가야지. 말하는
# 주제가 신청서 항목 방향으로만 치우쳐지면 안 된다는 거였어"): 이전에는 항목 수를 6개로
# 제한했는데, 그건 "회의 주제가 신청서 쪽으로 쏠리지 않게" 하려던 목적에 맞지 않는
# 장치였다 — 그 방지는 discussion 프롬프트의 [신청양식 참고 규칙](질문 주제·순서를 안
# 바꾸고, 표현만 다듬는 데만 쓰고, "신청양식"이라는 말 자체를 발언에 안 씀)이 실제로
# 담당하고, 여기서 항목 개수를 줄이는 건 그 목적과 무관하게 항목만 누락시켰다(실측:
# LLM이 상한 지시를 못 지키면 코드가 앞에서부터 자르면서 정작 중요한 내용 필드가
# 잘려나가고 연락처 필드만 남는 사고가 재현됨). 이제 원문에 실제로 있는 항목은 개수
# 제한 없이 전부 추출한다 — 유효성 검증(field_name 비어있지 않은지)만 남긴다.
_APPLICATION_FORM_TRUNCATE_CHARS = _ANNOUNCEMENT_TRUNCATE_CHARS


def _build_application_form_analysis_prompt(text: str) -> str:
    truncated = text[:_APPLICATION_FORM_TRUNCATE_CHARS]
    return f"""당신은 공모전·지원사업 관련 문서 묶음(공고문, 평가기준, 신청서 양식이 섞여 있을
수 있습니다)에서, 신청서 양식에 실제로 존재하는 기입란/작성란 항목만 뽑는 보조입니다.
공고문이나 평가기준 설명은 항목이 아닙니다 — 신청서 양식 문서에 실제로 있는 기입란만
고르세요. 문서 묶음에 신청서 양식 자체가 없으면 items를 빈 배열로 두세요(지어내지 마세요).

다음은 항목이 아닙니다 — 포함하지 마세요: 접수 절차 안내, 제출 방법, 문의처, 심사 일정,
서명·날인란, 개인정보 수집 동의 문구, 표지·제목만 있는 섹션, 평가기준·심사 항목 설명.

field_name은 양식에 실제로 쓰인 항목 이름을 그대로 씁니다(의역하지 않습니다).
description은 그 항목에 무엇을 써야 하는지 양식 원문 기준으로 1문장 이내로 요약합니다 —
원문에 설명이 없으면 빈 문자열로 둡니다(지어내지 마세요).
char_limit은 양식에 명시된 글자 수 제한이 있을 때만 정수로 채우고, 없으면 null입니다
(추측해서 채우지 마세요).

양식에 실제로 있는 기입란은 개수 제한 없이 전부 뽑으세요 — 일부만 골라내지 마세요.

[문서 원문]
{truncated}

다음 JSON 형식으로만 응답하세요:
{{
  "items": [
    {{"field_name": "...", "description": "...", "char_limit": null}}
  ]
}}"""


def _build_announcement_analysis_prompt(text: str) -> str:
    truncated = text[:_ANNOUNCEMENT_TRUNCATE_CHARS]
    return f"""당신은 공모전·지원사업 공고문을 분석하는 보조입니다. 아래 공고문 원문을 읽고
announcement_title은 이 공고의 정식 명칭(공모전/지원사업 이름)만 뽑으세요 — 페이지
제목이나 게시판 메뉴명("공지사항", "보도자료" 등)이 아니라 본문에서 실제로 언급되는
공식 명칭을 쓰세요. 명확한 명칭을 못 찾으면 빈 문자열로 두세요(지어내지 마세요).

지어내지 말고, 못 찾은 항목은 빈 배열이나 "미공개"로 남기세요. 표 안의 정보도 반드시
읽으세요. evaluation_criteria에는 평가 대상/부문(예: 기업 평가, 도시 평가), 기준명,
배점을 빠짐없이 보존해 기준 하나당 문자열 하나로 작성하세요(예: "기업 평가 · 혁신성: 20점").
원문에 평가 기준은 있지만 배점만 없을 때만 각 기준에 "배점 미공개"를 표시하고, 평가
기준 자체를 못 찾았을 때만 ["배점 미공개"]를 사용하세요. 원문에 배점 숫자가 전혀 없는데
"평가 항목이 5개이니 100점을 5등분하면 20점" 같은 방식으로 배점을 계산해서 항목 이름에
붙이지 마세요(예: "혁신성 (20)") — 항목 개수로 배점을 역산하는 것도 지어내는 것과
같습니다. 원문에 그 항목 옆에 숫자(점수)가 실제로 적혀 있을 때만 그 숫자를 그대로
붙이세요.

submission_requirements에는 제출 서류·제출 방법을, application_review_conditions에는
신청 부문 선택·심사위원의 분야 변경·참여도에 따른 시상 수 변경 같은 신청/심사 운영
조건을 담으세요. disqualification_rules에는 수상 취소·결격 조건을 하나씩 분리해 담으세요.
key_dates에는 신청 기간뿐 아니라 평가일, 결과 발표일, 시상식 일시 등 모든 주요 일정을
"항목: 날짜/시간" 형태로 담으세요. schedule_items에는 같은 일정을 행사명과 날짜로
구조화하세요. 날짜는 YYYY-MM-DD 형식으로 쓰고, 기간이면 start_date와 end_date를 모두
채우세요. 결과 발표 방법(예: 공식 홈페이지)이 원문에 있으면 method에 넣으세요. 원문에
없는 날짜나 방법은 만들지 마세요. selection_benefits에는 선정·수상 혜택을 빠짐없이 담으세요.
각 배열 항목에는 서로 독립된 사실 하나만 담고, 별개의 조건을 "또는/및"으로 합치지 마세요.
URL과 첨부 문서가 함께 있으면 두 출처를 모두 읽고 서로 보완하세요. URL의 짧은 요약에
없는 세부 표·일정·혜택이 첨부 PDF에 있으면 반드시 첨부 PDF 내용을 결과에 포함하세요.

strategic_analysis는 원문을 근거로 한 당신의 추론(전략적 분석)입니다 — 사실 단정이
아니라 판단임을 유지하고, 근거 없는 단정을 피하세요.

category는 아래 8개 중 이 공모전/지원사업과 가장 가까운 것 하나를 정확히 그대로
고르세요(목록에 없는 값은 쓰지 마세요, 애매하면 "기타"): {', '.join(_CONTEST_CATEGORIES)}

[공고문 원문]
{truncated}

다음 JSON 형식으로만 응답하세요:
{{
  "announcement_title": "...",
  "category": "...",
  "official_facts": {{
    "eligibility": ["..."],
    "deadline": "...",
    "submission_requirements": ["..."],
    "evaluation_criteria": ["..."],
    "disqualification_rules": ["..."],
    "application_review_conditions": ["..."],
    "key_dates": ["..."],
    "schedule_items": [
      {{"event_label": "신청 기간", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD 또는 빈 문자열", "method": "공식 홈페이지 또는 빈 문자열", "source_text": "원문 일정 표현"}}
    ],
    "selection_benefits": ["..."]
  }},
  "strategic_analysis": {{
    "core_intent": "...",
    "winning_points": ["..."],
    "recommended_direction": ["..."],
    "risk_flags": ["..."]
  }},
  "evidence": [
    {{"claim": "...", "source_type": "announcement 또는 inference", "location": "원문 내 위치 또는 null", "confidence": "high, medium, low 중 하나"}}
  ]
}}

evidence는 official_facts/strategic_analysis 중 실제로 중요한 판단 3~6개를 골라 근거를
표시하세요. source_type="announcement"인 항목은 location에 어느 부분에서 확인했는지
짧게 적고(예: "제출 요건 문단"), source_type="inference"인 항목은 location을 null로 두세요.

응답 전 마지막으로 모든 출처를 다시 훑어 다음을 점검하세요:
- 기업/도시처럼 평가 대상별 표가 여러 개면 각 표의 모든 기준과 배점을 추출했는가
- 진행 절차에 평가일·결과 발표일·시상식이 있으면 key_dates에 모두 들어갔는가
- 수상 제한, 신청 분야 변경 조건, 선정 혜택을 각각 해당 배열에 빠짐없이 담았는가"""


def _call_announcement_analysis_llm(prompt: str) -> str:
    model = settings.reviewer_model()
    client = trace_openai_client(OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=1))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        # 가은/Claude(2026-07-22, 요청: 신청양식 항목 누락 재현 — "이 내용이 없다?"): temperature를
        # 지정하지 않으면 OpenAI 기본값(1.0)이 적용돼, 같은 문서·같은 프롬프트로 반복 호출해도
        # 매번 다른 항목이 빠지거나(실측: 도시 부문 신청서의 핵심 내용 필드가 통째로 누락된
        # 사례) 배점을 지어내는(실측: "혁신성 (20)") 등 변동성이 컸다. 이 함수는 announcement-
        # analysis/application-form-analysis 둘 다에 쓰이는데, 둘 다 "문서에 실제로 있는
        # 내용을 있는 그대로 뽑는" 추출 작업이라 창의성이 필요 없다 — 0에 가깝게 낮춰 매번
        # 같은 입력에 최대한 같은(빠짐없는) 결과가 나오게 한다.
        temperature=0,
    )
    return resp.choices[0].message.content


# 가은/Claude(2026-07-21): contest_works는 이 앱이 만드는 데이터가 아니라 kyh님이 별도로
# 크롤링/분류한 아카이브라 스키마가 느슨하다(work_title 없는 문서는 contest_title로
# 대체, inst_nm/source_org 둘 다 없을 수 있음) — 그래서 값을 지어내지 않고 빈 문자열로
# 두는 방어적 변환을 거친다. 컬렉션이 비어있거나 매칭이 0건이면 그냥 빈 리스트를 반환
# (에러 아님) — has_similar_case_data=False로 이어져 프론트가 기존 "미확보" 문구를 보여준다.
async def _find_similar_works(category: str) -> list[SimilarWork]:
    docs = await contest_work_repo.find_by_category(category)
    return [
        SimilarWork(
            title=str(doc.get("work_title") or doc.get("contest_title") or "").strip(),
            source_org=str(doc.get("source_org") or doc.get("inst_nm") or "").strip(),
            award_grade=str(doc.get("award_grade") or "").strip(),
            selection_status=str(doc.get("selection_status") or ""),
            contest_title=str(doc.get("contest_title") or ""),
        )
        for doc in docs
        if doc.get("work_title") or doc.get("contest_title")
    ]


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, (str, int, float)) and str(v).strip()]


_KOREAN_WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")
_SCHEDULE_DATE_PATTERN = re.compile(
    r"(?:(20\d{2})\s*(?:년|[./-])\s*)?(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})\s*일?"
)


def _infer_schedule_year(*values: str) -> int | None:
    for value in values:
        # "2026년"에서 숫자와 한글은 모두 정규식의 word 문자라 \b가 생기지 않는다.
        match = re.search(r"(?<!\d)(20\d{2})(?!\d)", value or "")
        if match:
            return int(match.group(1))
    return None


def _normalize_schedule_date(value: object, default_year: int | None) -> str:
    text = str(value or "").strip()
    match = _SCHEDULE_DATE_PATTERN.search(text)
    if not match:
        return ""
    year = int(match.group(1)) if match.group(1) else default_year
    if year is None:
        return ""
    try:
        return date(year, int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return ""


def _schedule_weekday(value: str) -> str:
    try:
        return _KOREAN_WEEKDAYS[date.fromisoformat(value).weekday()]
    except (ValueError, TypeError):
        return ""


def _canonical_schedule_label(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    if any(token in compact for token in ("신청기간", "접수기간", "공모신청", "접수마감", "신청마감")):
        return "신청 기간"
    if any(token in compact for token in ("결과발표", "심사결과", "선정발표", "수상자발표")):
        return "결과 발표"
    if "시상식" in compact:
        return "시상식"
    if any(token in compact for token in ("서류평가", "서면평가", "서류심사", "평가", "심사")):
        return "서류 평가"
    return value.strip().rstrip(":：·") or "주요 일정"


def _schedule_item_from_mapping(item: dict, default_year: int | None) -> ScheduleItem | None:
    label = _canonical_schedule_label(str(item.get("event_label") or item.get("label") or ""))
    source_text = str(item.get("source_text") or "").strip()
    start_date = _normalize_schedule_date(item.get("start_date"), default_year)
    end_date = _normalize_schedule_date(item.get("end_date"), default_year)
    if not start_date and source_text:
        matches = list(_SCHEDULE_DATE_PATTERN.finditer(source_text))
        if matches:
            start_date = _normalize_schedule_date(matches[0].group(0), default_year)
            if len(matches) > 1:
                inherited_year = int(start_date[:4]) if start_date else default_year
                end_date = _normalize_schedule_date(matches[1].group(0), inherited_year)
    if not start_date:
        return None
    return ScheduleItem(
        event_label=label,
        start_date=start_date,
        end_date=end_date,
        start_weekday=_schedule_weekday(start_date),
        end_weekday=_schedule_weekday(end_date) if end_date else "",
        method=str(item.get("method") or "").strip(),
        source_text=source_text,
    )


def _schedule_item_from_text(value: str, default_year: int | None) -> ScheduleItem | None:
    matches = list(_SCHEDULE_DATE_PATTERN.finditer(value))
    if not matches:
        return None
    label_text = value[: matches[0].start()].strip(" \t:：·-")
    start_date = _normalize_schedule_date(matches[0].group(0), default_year)
    inherited_year = int(start_date[:4]) if start_date else default_year
    end_date = _normalize_schedule_date(matches[1].group(0), inherited_year) if len(matches) > 1 else ""
    if not start_date:
        return None
    method = "공식 홈페이지" if "홈페이지" in value else ""
    return ScheduleItem(
        event_label=_canonical_schedule_label(label_text),
        start_date=start_date,
        end_date=end_date,
        start_weekday=_schedule_weekday(start_date),
        end_weekday=_schedule_weekday(end_date) if end_date else "",
        method=method,
        source_text=value,
    )


def _build_schedule_items(facts: dict, source_text: str = "") -> list[ScheduleItem]:
    key_dates = _coerce_str_list(facts.get("key_dates"))
    explicit_items = facts.get("schedule_items")
    explicit_items = explicit_items if isinstance(explicit_items, list) else []
    default_year = _infer_schedule_year(
        *(str(item.get("start_date") or "") for item in explicit_items if isinstance(item, dict)),
        *key_dates,
        source_text,
    )

    items: list[ScheduleItem] = []
    for raw_item in explicit_items:
        if isinstance(raw_item, dict):
            normalized = _schedule_item_from_mapping(raw_item, default_year)
            if normalized:
                items.append(normalized)
    if not items:
        for raw_date in key_dates:
            normalized = _schedule_item_from_text(raw_date, default_year)
            if normalized:
                items.append(normalized)

    # 구형 LLM 응답이 신청 기간을 key_dates에서 빠뜨린 경우 deadline을 최소 호환값으로
    # 사용한다. 기간 전체가 명시된 schedule_items가 있으면 이 분기는 실행되지 않는다.
    if not any(item.event_label == "신청 기간" for item in items):
        deadline = str(facts.get("deadline") or "")
        normalized = _schedule_item_from_text(f"신청 기간: {deadline}", default_year)
        if normalized:
            items.insert(0, normalized)

    deduplicated: list[ScheduleItem] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.event_label, item.start_date, item.end_date)
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    return deduplicated


def _build_official_facts(payload: object, source_text: str = "") -> OfficialFacts:
    facts = payload if isinstance(payload, dict) else {}
    return OfficialFacts(
        eligibility=_coerce_str_list(facts.get("eligibility")),
        deadline=str(facts.get("deadline") or "미공개"),
        submission_requirements=_coerce_str_list(facts.get("submission_requirements")),
        evaluation_criteria=_coerce_str_list(facts.get("evaluation_criteria")) or ["배점 미공개"],
        disqualification_rules=_coerce_str_list(facts.get("disqualification_rules")),
        application_review_conditions=_coerce_str_list(facts.get("application_review_conditions")),
        key_dates=_coerce_str_list(facts.get("key_dates")),
        schedule_items=_build_schedule_items(facts, source_text),
        selection_benefits=_coerce_str_list(facts.get("selection_benefits")),
    )


def _missing_announcement_details(text: str, facts: OfficialFacts) -> list[str]:
    """원문에 명시적 표지가 있는데 구조화 결과에서 빠진 항목만 재검증 대상으로 잡는다."""
    compact = re.sub(r"\s+", " ", text)
    evaluation = " ".join(facts.evaluation_criteria)
    dates = " ".join(facts.key_dates)
    missing: list[str] = []

    if "혁신성" in text and "사회적 가치성" in text and (
        not facts.evaluation_criteria or evaluation == "배점 미공개"
    ):
        missing.append("평가 기준과 배점")
    if re.search(r"평가.{0,160}\b8[.]?\s*19", compact) and "평가" not in dates:
        missing.append("평가일")
    if "결과발표" in compact and re.search(r"결과발표.{0,80}\b8[.]?\s*24", compact) and "결과" not in dates:
        missing.append("결과 발표일")
    if "17:30~19:00" in text and not any("17:30" in date for date in facts.key_dates):
        missing.append("시상식 일시")
    if "심사위원 판단" in text and not facts.application_review_conditions:
        missing.append("신청·심사 조건")
    if "기업설명회" in text and not facts.selection_benefits:
        missing.append("선정 혜택")
    return missing


# 가은/Claude(2026-07-23, 요청: 재검증을 "찾아내라" 압박형 -> "인용해서 확인"형으로 교체):
# _missing_announcement_details가 내는 한국어 라벨 -> official_facts 실제 필드명 매핑.
# 검증(quote)이 통과된 라벨에 해당하는 필드만 재검증 응답 값으로 덮어쓴다.
_MISSING_DETAIL_FIELD_MAP = {
    "평가 기준과 배점": "evaluation_criteria",
    "평가일": "key_dates",
    "결과 발표일": "key_dates",
    "시상식 일시": "key_dates",
    "신청·심사 조건": "application_review_conditions",
    "선정 혜택": "selection_benefits",
}


def _build_announcement_audit_prompt(text: str, missing_details: list[str]) -> str:
    """1차 응답에서 놓쳤을 가능성이 있는 항목만 원문 인용으로 재확인한다. 예전엔 전체
    스키마를 다시 채우게 시켰는데("누락을 보완한 전체 JSON을 처음부터 다시 출력"), "찾아
    내라"는 압박이 배점처럼 원문에 정말 없는 값을 100점÷항목수로 지어내는 걸로 이어지는
    걸 실측했다(2026-07-23). 대신 "원문에서 그 문장을 그대로 복사해와라, 없으면 없다고
    답해라"만 시키고, quote가 실제로 원문에 있는지는 모델의 자기보고를 믿지 않고 서버
    (_verified_audit_fields)가 직접 대조한다. 제목·전략분석·근거는 이 재검증 대상이 아니라
    응답 스키마에서 아예 뺐다 — 1차 프롬프트 전체를 다시 붙이면 원래 스키마 지시와 이번
    스키마 지시가 겹쳐서 모델이 헷갈릴 여지도 없앤다."""
    truncated = text[:_ANNOUNCEMENT_TRUNCATE_CHARS]
    missing_list = ", ".join(missing_details)
    return f"""당신은 공모전·지원사업 공고문 분석 결과를 재확인하는 보조입니다. 1차 분석에서
다음 항목이 원문에 있는데 빠졌을 가능성이 제기됐습니다: {missing_list}.

[공고문 원문]
{truncated}

각 항목마다 위 원문을 다시 훑어보고, 그 항목을 뒷받침하는 문장을 원문 그대로(요약·의역
없이) 복사해 quote에 넣으세요. 원문에서 그런 문장을 정말 못 찾으면 quote를 빈 문자열("")로
두고 found를 false로 하세요. 배점처럼 숫자 자체가 원문에 없으면, 항목명(기준)은 이번에 더
찾았더라도 그 숫자에 대한 quote는 만들어내지 말고 found를 false로 두세요 — 다른 항목의
배점을 참고해 값을 추정하거나 100점을 항목 개수로 나누는 것도 금지입니다. quote는 서버가
원문과 그대로 대조해서만 인정하므로, 원문에 없는 문장을 지어내도 통과되지 않습니다.

다음 JSON 형식으로만 응답하세요:
{{
  "verification": [
    {{"field": "위 누락 항목 이름을 정확히 그대로", "quote": "원문에서 그대로 복사한 문장 또는 \\"\\"", "found": true 또는 false}}
  ],
  "official_facts": {{
    "eligibility": ["..."],
    "deadline": "...",
    "submission_requirements": ["..."],
    "evaluation_criteria": ["..."],
    "disqualification_rules": ["..."],
    "application_review_conditions": ["..."],
    "key_dates": ["..."],
    "selection_benefits": ["..."]
  }}
}}

official_facts는 found:true로 확인한 항목에 해당하는 필드만 이번 재검증 값으로 채우면
됩니다. 확인 못한 필드는 서버가 어차피 쓰지 않으니 빈 배열로 두세요."""


def _verified_audit_fields(verification: object, text: str) -> set[str]:
    """재검증 응답의 quote가 실제로 원문에 있는 문장인지 서버에서 직접 대조한다 — 모델이
    "찾았다"(found=true)고 자체 보고한 것만 믿지 않는다. 공백 차이만 정규화해서 비교하고,
    quote가 원문에 없으면 그 항목은 검증 실패로 보고 official_facts 병합에서 제외한다."""
    if not isinstance(verification, list):
        return set()
    compact_text = re.sub(r"\s+", " ", text)
    verified_labels: set[str] = set()
    for item in verification:
        if not isinstance(item, dict) or not item.get("found"):
            continue
        quote = str(item.get("quote") or "").strip()
        if not quote:
            continue
        compact_quote = re.sub(r"\s+", " ", quote)
        if compact_quote in compact_text:
            verified_labels.add(str(item.get("field") or ""))
    return {_MISSING_DETAIL_FIELD_MAP[label] for label in verified_labels if label in _MISSING_DETAIL_FIELD_MAP}


@router.post("/{project_id}/announcement-analysis", response_model=AnnouncementAnalysisResponse)
async def get_announcement_analysis(
    project_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    project = await verify_project_owner(project_id, user_email)

    # 가은/Claude(2026-07-21): dynamic_rubric_mapping과 동일한 "프로젝트당 1회 계산 후
    # 캐시" 패턴 — 재방문마다 다시 LLM을 부르지 않는다. 공고문을 나중에 추가/교체해도
    # 스키마/추출 지시가 바뀌면 analysis_version이 달라져 기존 캐시를 한 번 재계산한다.
    # 같은 버전 안에서는 재방문마다 LLM을 다시 호출하지 않는다.
    cached = project.get("announcement_analysis_cache")
    if isinstance(cached, dict) and cached.get("analysis_version") == _ANNOUNCEMENT_ANALYSIS_CACHE_VERSION:
        logger.info("[announcement-analysis] project_id=%s 캐시된 분석 결과 재사용", project_id)
        return AnnouncementAnalysisResponse(**cached)

    text, names = await _load_criteria_documents_text(project_id)
    if not text.strip():
        # 가은/Claude(2026-07-21): 공고문을 하나도 안 넣었으면 LLM을 호출하지 않는다 —
        # "정보 없음"을 지어내는 것보다 화면에서 그 상태 자체를 명시적으로 보여준다.
        # has_announcement: false는 "아직 없음"이라 캐시하지 않는다 — 공고문을 나중에
        # 추가하면 그때는 실제로 분석해야 하기 때문.
        return AnnouncementAnalysisResponse(has_announcement=False)

    prompt = _build_announcement_analysis_prompt(text)
    raw = await run_in_threadpool(_call_announcement_analysis_llm, prompt)

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    facts_raw = parsed.get("official_facts") if isinstance(parsed, dict) else None
    official_facts = _build_official_facts(facts_raw, text)
    missing_details = _missing_announcement_details(text, official_facts)
    if missing_details:
        logger.warning(
            "[announcement-analysis] project_id=%s 누락 감지=%s, 인용 재검증",
            project_id,
            missing_details,
        )
        audit_prompt = _build_announcement_audit_prompt(text, missing_details)
        audited_raw = await run_in_threadpool(_call_announcement_analysis_llm, audit_prompt)
        try:
            audited = json.loads(audited_raw)
        except (json.JSONDecodeError, TypeError):
            audited = None
        if isinstance(audited, dict):
            verified_fields = _verified_audit_fields(audited.get("verification"), text)
            logger.info(
                "[announcement-analysis] project_id=%s 인용 검증 통과 필드=%s",
                project_id,
                sorted(verified_fields) or "없음",
            )
            audited_facts_raw = audited.get("official_facts")
            # 검증(quote 대조)을 통과한 필드만 official_facts에 덮어쓴다 — parsed(제목/전략
            # 분석/근거)는 재검증 대상이 아니므로 그대로 1차 응답 값을 유지한다(예전엔
            # parsed 전체를 재검증 응답으로 바꿔치기해서, 검증하지도 않은 필드까지 두 번째
            # 생성 결과로 조용히 덮였다).
            if verified_fields and isinstance(audited_facts_raw, dict):
                merged_facts_raw = dict(facts_raw) if isinstance(facts_raw, dict) else {}
                for field in verified_fields:
                    if field in audited_facts_raw:
                        merged_facts_raw[field] = audited_facts_raw[field]
                facts_raw = merged_facts_raw
                official_facts = _build_official_facts(facts_raw, text)

    strategy_raw = parsed.get("strategic_analysis") if isinstance(parsed, dict) else None
    strategy_raw = strategy_raw if isinstance(strategy_raw, dict) else {}
    strategic_analysis = StrategicAnalysis(
        core_intent=str(strategy_raw.get("core_intent") or ""),
        winning_points=_coerce_str_list(strategy_raw.get("winning_points")),
        recommended_direction=_coerce_str_list(strategy_raw.get("recommended_direction")),
        risk_flags=_coerce_str_list(strategy_raw.get("risk_flags")),
    )

    evidence: list[AnnouncementEvidence] = []
    for item in (parsed.get("evidence") if isinstance(parsed, dict) else None) or []:
        if not isinstance(item, dict) or not item.get("claim"):
            continue
        source_type = item.get("source_type") if item.get("source_type") in ("announcement", "inference") else "inference"
        confidence = item.get("confidence") if item.get("confidence") in ("high", "medium", "low") else "medium"
        evidence.append(
            AnnouncementEvidence(
                claim=str(item["claim"]),
                source_type=source_type,
                location=str(item["location"]) if item.get("location") else None,
                confidence=confidence,
            )
        )

    announcement_title = str(parsed.get("announcement_title") or "").strip() if isinstance(parsed, dict) else ""

    category = parsed.get("category") if isinstance(parsed, dict) else None
    category = category if category in _CONTEST_CATEGORIES else "기타"
    similar_works = await _find_similar_works(category)

    result = AnnouncementAnalysisResponse(
        has_announcement=True,
        announcement_title=announcement_title,
        official_facts=official_facts,
        strategic_analysis=strategic_analysis,
        evidence=evidence,
        has_similar_case_data=len(similar_works) > 0,
        similar_works=similar_works,
        source_document_names=names,
    )
    cache_payload = result.model_dump()
    cache_payload["analysis_version"] = _ANNOUNCEMENT_ANALYSIS_CACHE_VERSION
    await project_repo.update_project(project_id, {"announcement_analysis_cache": cache_payload})
    return result


@router.post("/{project_id}/application-form-analysis", response_model=ApplicationFormAnalysisResponse)
async def get_application_form_analysis(
    project_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    """가은/Claude(2026-07-22, 요청: 업로드 영역 통합) — announcement-analysis와 같은 문서
    풀(document_role="criteria", 공고문·평가기준·신청서 양식이 한 영역에서 함께 업로드됨)을
    다시 읽어 그중 신청서 양식에 해당하는 기입 항목만 추출한다. announcement-analysis와
    정책이 같다(프로젝트당 1회 계산 후 캐시, 문서가 없으면 LLM을 부르지 않고
    has_application_form=False, 지어내지 않고 원문에 없으면 빈 값) — 캐시 필드 이름과
    프롬프트 관점(신청서 양식 항목 추출 vs 공고문 분석)만 다르다."""
    user_email = get_current_user(authorization)
    project = await verify_project_owner(project_id, user_email)

    cached = project.get("application_form_analysis_cache")
    if cached:
        logger.info("[application-form-analysis] project_id=%s 캐시된 분석 결과 재사용", project_id)
        return ApplicationFormAnalysisResponse(**cached)

    text, names = await _load_criteria_documents_text(project_id)
    if not text.strip():
        # 문서를 하나도 등록하지 않았으면 LLM을 호출하지 않는다 — 회의 화면은 이 값을
        # 그대로 "선택 사항"으로 취급해 항목 없이 정상 진행한다(약한 주입은 없을 뿐 회의
        # 자체를 막지 않는다).
        return ApplicationFormAnalysisResponse(has_application_form=False)

    prompt = _build_application_form_analysis_prompt(text)
    raw = await run_in_threadpool(_call_announcement_analysis_llm, prompt)

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    items_raw = parsed.get("items") if isinstance(parsed, dict) else None
    items: list[ApplicationFormItem] = []
    for item in (items_raw if isinstance(items_raw, list) else []):
        if not isinstance(item, dict) or not str(item.get("field_name") or "").strip():
            continue
        char_limit = item.get("char_limit")
        items.append(
            ApplicationFormItem(
                field_name=str(item["field_name"]).strip(),
                description=str(item.get("description") or "").strip(),
                char_limit=char_limit if isinstance(char_limit, int) else None,
            )
        )

    result = ApplicationFormAnalysisResponse(
        has_application_form=True,
        items=items,
        source_document_names=names,
    )
    await project_repo.update_project(project_id, {"application_form_analysis_cache": result.model_dump()})
    return result
