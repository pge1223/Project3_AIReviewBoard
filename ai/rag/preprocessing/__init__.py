"""
HTML Content Cleaning Module
=============================
ai.rag.loaders가 수집한 WebPageContent에서 RAG에 필요한 핵심 블록만 남기고
상단 메뉴/추천 콘텐츠/뉴스레터/footer 등 노이즈를 규칙 기반으로 제거한다.
LLM 호출 없이 동작하며, 원본 WebPageContent는 수정하지 않는다.

사용 예시:
    from ai.rag.loaders import load_from_url
    from ai.rag.preprocessing import clean_page_content

    result = load_from_url(url)
    if result.page_content:
        cleaned = clean_page_content(result.page_content)
        # cleaned.cleaned_blocks -> 이후 청킹 모듈 입력
"""

from ai.rag.preprocessing.boilerplate_detection import (
    extract_boundary_lines,
    find_repeated_boundary_texts,
    is_boilerplate_text,
)
from ai.rag.preprocessing.content_hashing import (
    compute_chunk_content_hash,
    compute_normalized_content_hash,
    compute_source_content_hash,
    deduplicate_chunks,
    normalize_for_content_hash,
)
from ai.rag.preprocessing.html_cleaner import clean_page_content
from ai.rag.preprocessing.schemas import (
    CleanedWebContent,
    RemovedBlock,
    RemovalReason,
    CleaningMethod,
)
from ai.rag.preprocessing.short_chunk_policy import (
    ShortChunkResult,
    merge_or_filter_short_chunks,
)

__all__ = [
    "clean_page_content",
    "CleanedWebContent",
    "RemovedBlock",
    "RemovalReason",
    "CleaningMethod",
    "extract_boundary_lines",
    "find_repeated_boundary_texts",
    "is_boilerplate_text",
    "compute_chunk_content_hash",
    "compute_normalized_content_hash",
    "compute_source_content_hash",
    "deduplicate_chunks",
    "normalize_for_content_hash",
    "ShortChunkResult",
    "merge_or_filter_short_chunks",
]
