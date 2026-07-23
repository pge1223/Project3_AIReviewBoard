"""
Chunking Configuration Defaults
================================
"""

import re
from typing import Optional

DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 120

# 한국어 문단/문장 경계를 우선 고려한 분할 기준 (문자 수 기준).
# 문단(빈 줄) -> 줄바꿈 -> 문장 종결부호(한글은 마침표 앞의 '다/요' 등에 의존하지 않고
# 보편적인 종결부호 자체로 판단) -> 전각 문장부호 -> 공백 -> 강제 문자 단위 분할("").
# RecursiveCharacterTextSplitter는 순서대로 첫 번째로 텍스트에 존재하는 구분자를 사용하므로
# 더 구체적인 구분자를 먼저 두되, 중복되는 하위집합(예: "다. "는 이미 ". "에 포함)은 넣지 않는다.
DEFAULT_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    ". ", "! ", "? ",
    "。", "！", "？",
    " ",
    "",
]

# v1 -> v2: PDF 줄바꿈 정규화(merge_wrapped_pdf_lines)/whole-line 제목 인식
# (extract_whole_line_heading_title)/글머리표 보존 도입으로 청크 내용·section_title·chunk_id가
# 실질적으로 달라진다. chunk_id 해시에 chunking_version이 포함되므로(_generate_chunk_id),
# 동일 문서라도 v1 청크와 v2 청크는 서로 다른 chunk_id를 가진다 — 기존 v1로 색인된 Chroma
# 데이터는 자동으로 갱신되거나 삭제되지 않으며, 개선 효과를 적용하려면 해당 문서를 재색인해야 한다.
# v2 -> v3: 평가표의 여러 평가 항목과 세부 질문이 한 800자 청크에 함께 들어가
# 검색/Planner가 현재 쟁점과 다른 질문을 고르는 문제를 해결한다. 평가표로 확인된 본문만
# "평가 항목 + 세부 질문 1개" 단위로 분리하며 일반 본문/표 청킹은 기존 규칙을 유지한다.
# 기존 v2 Chroma 레코드는 자동 변환되지 않으므로 효과를 적용하려면 문서를 재색인해야 한다.
CHUNKING_VERSION: str = "chunking_v3"

# 목차 판정: MVP는 강한 heading 키워드가 있을 때만 확정한다 (과탐 방지 우선)
TOC_HEADING_KEYWORDS: list[str] = ["목차", "차례", "contents"]

# 구조적 목차 의심(점선/페이지번호로 끝나는 줄 비율) 임계값.
# 이 신호만으로는 절대 TOC로 확정하지 않고 warning 용도로만 사용한다.
TOC_STRUCTURAL_LINE_MATCH_RATIO: float = 0.8

# 의사(pseudo) heading 인식: "□ 접수방법: ..." 형태의 단락을 section_title 후보로 인식.
# 특정 사이트에 하드코딩하지 않고 이 마커 문자 집합만으로 판단한다.
# '▢'(U+25A2)는 '□'(U+25A1)와 시각적으로 비슷해 HWPX→PDF 변환 산출물에서 흔히 쓰이는 변형이다.
PSEUDO_HEADING_MARKERS: str = "□■◇◆○●▢"
PSEUDO_HEADING_MAX_TITLE_LENGTH: int = 30

# 목록형 블록 인식: 아래 마커가 반복되면(최소 발생 횟수 이상) 항목 단위로 취급해
# RecursiveCharacterTextSplitter가 항목 중간을 자르지 않도록 한다.
# "-"(줄바꿈 뒤 하이픈), "□", "※", "①"~"⑩"(U+2460~U+2469)
LIST_ITEM_MARKER_PATTERN: str = r"[-□※]|[①-⑩]"
LIST_ITEM_MIN_MARKER_COUNT: int = 2

# 마지막 청크가 이 길이(문자 수) 미만이면 직전 청크와 병합을 시도한다.
# 병합 후 chunk_size를 초과하면 병합하지 않는다.
TAIL_CHUNK_MIN_CHARS: int = 80

# PDF(특히 HWPX→PDF 변환) 줄바꿈 정규화: 이 문자'만'으로 이루어진 블록은 장식용 기호로 보고
# 제거한다(ai.rag.chunking.adapters.merge_wrapped_pdf_lines). '‧'(U+2027 HYPHENATION POINT)만
# 포함한다 — 실제 K-Lawyer HWPX→PDF 변환 산출물에서 문장 사이에 독립된 장식 줄로 남는 것을
# 직접 확인한 문자다. '•'/'◦'/'∙'/'·'는 실제 글머리표(목록 마커)로 흔히 쓰이므로 여기서
# 무조건 삭제하지 않고 BULLET_MARKER_CHARS로 별도 취급한다(보수적 접근 — 확실한 것만 제거).
DECORATIVE_SYMBOL_CHARS: str = "‧"

# 목록 글머리표로 흔히 쓰이는 문자. PDF가 줄 단위로 블록을 쪼개면 이 마커가 텍스트 없이
# 독립 블록으로 남는 경우가 있는데(예: '•' 한 글자짜리 블록), 장식 기호처럼 제거하면
# 실제 목록 구조가 사라진다. merge_wrapped_pdf_lines()는 이 마커만 있는 블록을 만나면
# 삭제하지 않고 다음 본문 블록과 "마커 텍스트" 형태로 결합해 목록 항목으로 보존한다.
BULLET_MARKER_CHARS: str = "•◦∙·"

# 이 문자로 끝나는 블록은 "문장이 끝났다"고 보고 다음 블록과 공백으로 병합하지 않는다.
SENTENCE_TERMINATOR_CHARS: str = ".!?:)”’」』】》"

_WHOLE_LINE_HEADING_PREFIX_RE = re.compile(
    rf"^\s*(?:\d{{1,2}}[).]|[①-⑩]|[{re.escape(PSEUDO_HEADING_MARKERS)}])\s*"
)


def extract_whole_line_heading_title(text: str) -> Optional[str]:
    """
    "1) 개요" / "① 법제처 법령 학습" / "▢ 기대효과"처럼 블록 전체가 통째로 짧은 제목인 경우
    제목만 추출한다. ai.rag.chunking.chunker._extract_pseudo_heading_title()은 "마커 + 제목 +
    구분자 + 본문"이 한 블록에 섞여 있는 경우(예: "□ 접수방법: 이메일로 제출...")를 다루고,
    이 함수는 PDF처럼 제목 한 줄이 통째로 별도 블록인 경우를 다룬다.

    줄바꿈 정규화(ai.rag.chunking.adapters)와 section 추론(ai.rag.chunking.chunker) 양쪽에서
    heading 경계를 동일한 기준으로 판단하기 위해 이 모듈에 둔다(두 모듈 간 순환 import 회피 목적도 겸함).
    확신할 수 없으면(구분자 뒤 내용이 없거나 너무 길거나 줄바꿈을 포함하면) None을 반환한다 —
    임의로 제목을 만들어내지 않는다.
    """
    stripped = text.strip()
    match = _WHOLE_LINE_HEADING_PREFIX_RE.match(stripped)
    if match is None:
        return None

    remainder = stripped[match.end():].strip()
    if not remainder or "\n" in remainder or len(remainder) > PSEUDO_HEADING_MAX_TITLE_LENGTH:
        return None
    return remainder


def looks_like_whole_line_heading(text: str) -> bool:
    """extract_whole_line_heading_title(text)가 제목을 추출할 수 있는지 여부만 필요할 때 사용."""
    return extract_whole_line_heading_title(text) is not None
