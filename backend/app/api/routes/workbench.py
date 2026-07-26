# 작성자: 재인/Claude (2026-07-21)
# 목적: "AI 피드백" 워크벤치(frontend/src/pages/board/WorkbenchScreen.jsx)가 원문에
#   정확한 하이라이트를 그릴 수 있도록, 위원 피드백이 기획서 원문 어디를 가리키는지
#   찾아준다.
#
# 원래는 GPT-4.1-mini에게 "원문 그대로 인용해달라"고 재질문하는 방식으로 만들었으나,
# 실측 비교 결과 청크 ID 직접 조회 방식(아래)이 AI 호출 없이도 항상 100% 정확한
# 원문을 반환해 더 낫다고 판단, 청크 방식만 최종으로 남긴다(GPT 재질문 방식은 삭제).
#
# 기존 경이님 파이프라인(ai/meeting/graph/evidence.py, reviewer_prompt.txt)의
# evidence.quote 문자열 자체를 하이라이트로 쓰지 않는 이유: reviewer_prompt.txt는
# "원문 의미를 바꾸지 않고 짧게 인용"만 요구해 글자 단위 정확한 인용을 보장하지 않는다.
# 대신 위원이 인용한 evidence.chunk_id로 벡터DB(Chroma)에서 청크 원문을 ID로 직접
# 조회한다 - 유사도 검색이 아니라 정확한 ID 매칭이라 항상 원문 그대로다.
# ChromaVectorStore.get_by_chunk_id는 ai/rag/retrieval/chroma_store.py(용준님 파일)에
# 추가한 메서드. documents.py의 _get_indexing_service() 싱글턴을 그대로 재사용하고
# 새 PersistentClient는 만들지 않는다(중복 client 생성 시 Windows에서 SQLite 잠김으로
# 멈추는 문제가 2026-07-18에 실제로 있었음).
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from fastapi import APIRouter, Header
from openai import OpenAI

from app.core.llm import trace_openai_client
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ai.rag.converters.config import HwpConversionConfig
from ai.rag.converters.exceptions import DocumentConversionError
from ai.rag.converters.preview_pdf_converter import convert_to_preview_pdf
from ai.rag.parsers import extract_document
from ai.rag.parsers.schemas import BlockType
from app.api.routes.documents import get_current_user, verify_project_owner, _get_indexing_service
from app.config import settings
from app.repositories.document_repository import DocumentRepository

router = APIRouter(prefix="/workbench", tags=["workbench"])
document_repo = DocumentRepository()
logger = logging.getLogger(__name__)


class ChunkLookup(BaseModel):
    id: str
    chunk_id: str


class QuotesRequest(BaseModel):
    lookups: list[ChunkLookup]


class QuoteMatch(BaseModel):
    id: str
    quote: str
    found: bool


class QuotesResponse(BaseModel):
    matches: list[QuoteMatch]


@router.post("/{project_id}/quotes", response_model=QuotesResponse)
async def get_quotes(
    project_id: str,
    body: QuotesRequest,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    if not body.lookups:
        return QuotesResponse(matches=[])

    vector_store = _get_indexing_service().vector_store

    def _fetch_all() -> list[QuoteMatch]:
        out: list[QuoteMatch] = []
        for lookup in body.lookups:
            text = vector_store.get_by_chunk_id(project_id, lookup.chunk_id)
            out.append(QuoteMatch(id=lookup.id, quote=text or "", found=bool(text)))
        return out

    results = await run_in_threadpool(_fetch_all)
    return QuotesResponse(matches=results)


# 재인/Claude(2026-07-21): "맥락 이상 감지" — 사용자(2026-07-21)에게 확인받은 방향.
#
# 실측으로 시도해본 순서(전부 실제 테스트 문서 5개 - 정상 2개, 확 다른 주제가 섞인
# 것 2개, 같은 주제인데 타겟층만 살짝 어긋난 미묘한 것 1개 - 로 검증):
# 1) RAG 청크(chunk_size=800) 임베딩을 그대로 재활용해 평균에서 먼 것을 찾음 →
#    청크 하나가 여러 문단을 묶어버려서 신호가 묻힘, 아예 못 잡음.
# 2) 문단 단위로 다시 KURE-v1 임베딩 후 z-score로 후보만 추려 GPT에 재판단 시킴 →
#    "완전히 다른 주제"는 잡지만, "같은 주제인데 타겟층만 어긋난" 미묘한 경우는
#    임베딩 거리 자체가 크게 안 벌어져서 후보에도 못 들었다(실측: z=1.25, 임계값
#    미달). 임베딩 방식은 애초에 이런 미묘한 불일치를 잡을 수 있는 신호가 아니었다.
# 3) 그래서 통계 단계를 아예 버리고, GPT가 문서 전체를 한 번에 읽고 "앞뒤 안 맞는
#    문단" 후보를 찾게 한 뒤(1단계), 그 후보를 다시 깐깐한 2차 검토자 역할로
#    재검증시킨다(2단계) - 2026년 문서 수준 불일치 탐지 연구(arXiv:2601.02627)에서
#    "후보 추출 후 재검증하면 정확도가 오른다"는 결과와 같은 방향. temperature=0으로
#    고정해 실행마다 결과가 흔들리는 것도 줄였다. 5개 테스트 문서 전부(정상 2개는
#    깨끗하게, 확실한 이상 2개 + 미묘한 이상 1개는 전부 정확히) 통과 확인.
#
# LLM 호출은 문서당 최대 2번(후보 찾기 + 재검증, 후보가 없으면 1번만)이고, 결과는
# 문서에 캐싱해 재방문 시 다시 부르지 않는다.
_CONTEXT_CHECK_MIN_PARAGRAPHS = 3  # 문단이 이보다 적으면 "전체 흐름"이라 할 게 없어 스킵
_CONTEXT_CHECK_MAX_PARAGRAPHS_IN_PROMPT = 60  # 비정상적으로 큰 문서에 대한 방어적 상한
_CONTEXT_CHECK_MIN_PARAGRAPH_CHARS = 40  # 소제목처럼 문장이 아닌 짧은 조각은 판단 대상에서 제외


class ContextFinding(BaseModel):
    id: str
    quote: str
    message: str


class ContextCheckResponse(BaseModel):
    findings: list[ContextFinding]


async def _find_target_document(project_id: str) -> Optional[dict]:
    documents = await document_repo.find_by_project_id(project_id)
    targets = [d for d in documents if d.get("document_role", "target") == "target"]
    return targets[0] if targets else None


# 재인/Claude(2026-07-22): PDF에서 문단을 세로 위치로 다시 묶는다.
#
# 왜 필요한가: extract_document(=ai.rag.parsers, 용준님)이 PDF를 파싱할 때 만드는 TEXT
# 블록은 "문단"이 아니라 PyMuPDF가 주는 시각적 "한 줄"(때로는 그보다 더 잘린 span)
# 단위다. 한국어 기획서를 실측해보니 한 문단이 5~15개 줄 블록으로 쪼개지고, 각 줄이
# 단어 중간에서 끊긴다("...활용해 정" / "책을..."). 이 상태로 맥락 이상 감지·오탈자
# 검사에 넣으면 (1)잘린 한 글자("정","어")를 문단으로 오인하고 (2)오탈자 검사가 줄 끝의
# 잘린 조각을 오타로 착각한다.
#
# 여기서는 RAG 색인 경로(pdf_parser.py)를 건드리지 않고, 이 워크벤치 기능(맥락 이상·
# 오탈자)만을 위해 fitz로 PDF를 직접 다시 읽어 문단을 복원한다 - pdf_parser.py의 블록
# granularity를 바꾸면 RAG 청킹/하이라이트 chunk_id에까지 영향이 가므로(용준님·경이님
# 영역) 의도적으로 이 레이어에 가둔다.
#
# 한계(중요): PDF는 줄바꿈 지점의 띄어쓰기 정보를 잃는다. 한국어는 단어 중간에서도
# 줄바꿈되므로, 줄을 공백으로 이으면 단어가 벌어지고("통계청"→"통 계청") 공백 없이
# 붙이면 진짜 띄어쓰기가 사라진다("그중 20대"→"그중20대"). 복구가 원리적으로 불가능한
# 정보라, 여기서는 공백 없이 붙이고(단어 깨짐이 더 위험) 오탈자 검사 쪽에서 PDF일 때
# 띄어쓰기 판정을 끈다(_build_typo_check_prompt(skip_spacing=True)). 맥락 이상 감지는
# 의미 기반이라 이 아티팩트에 영향을 거의 안 받는다.
def _extract_pdf_paragraphs(file_path: str) -> list[str]:
    doc = fitz.open(file_path)
    try:
        paragraphs: list[str] = []
        for page in doc:
            lines: list[tuple[float, float, str]] = []  # (y0, y1, text)
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:  # 텍스트 블록만
                    continue
                for line in block.get("lines", []):
                    text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if not text:
                        continue
                    bbox = line["bbox"]
                    lines.append((bbox[1], bbox[3], text))
            lines.sort(key=lambda t: t[0])

            # 세로 간격으로 문단을 나눈다: 앞 줄 아래끝~현재 줄 위끝 간격이 줄 높이의
            # 0.6배를 넘으면 문단이 바뀐 것으로 본다(문단 내 줄 간격보다 문단 사이
            # 간격이 확연히 크다는 실측에 기반).
            current: list[str] = []
            prev_bottom: Optional[float] = None
            prev_height: Optional[float] = None
            for top, bottom, text in lines:
                if prev_bottom is not None and (top - prev_bottom) > 0.6 * prev_height:
                    paragraphs.append("".join(current))
                    current = []
                current.append(text)
                prev_bottom, prev_height = bottom, bottom - top
            if current:
                paragraphs.append("".join(current))
        return paragraphs
    finally:
        doc.close()


def _extract_body_paragraphs(file_path: str) -> list[str]:
    """서술형 문단만 후보로 쓴다 - 짧은 소제목·글머리 항목은 "앞뒤 문맥이 맞는지" 판단
    대상으로 적절하지 않아 최소 글자수 미만은 제외한다. PDF는 위 _extract_pdf_paragraphs로
    문단을 복원하고, 그 외(docx 등)는 extract_document이 이미 문단 단위를 주므로 그대로 쓴다."""
    if os.path.splitext(file_path)[1].lower() == ".pdf":
        candidates = _extract_pdf_paragraphs(file_path)
    else:
        candidates = [
            block.content.strip()
            for block in extract_document(file_path).blocks
            if block.block_type == BlockType.TEXT
        ]
    return [c for c in candidates if len(c) >= _CONTEXT_CHECK_MIN_PARAGRAPH_CHARS]


def _build_context_find_prompt(paragraphs: list[str]) -> str:
    numbered = "\n\n".join(
        f"{i + 1}. {text[:400]}" for i, text in enumerate(paragraphs[:_CONTEXT_CHECK_MAX_PARAGRAPHS_IN_PROMPT])
    )
    return f"""당신은 사업계획서를 처음부터 끝까지 꼼꼼히 읽는 편집자입니다. 아래 문단들을
전부 읽고, 문서 전체의 흐름과 앞뒤가 안 맞는 부분이 있는지 찾으세요. 특히 이 두
가지를 중점적으로 확인하세요:
1) 완전히 다른 업종/주제가 맥락 없이 끼어든 경우
2) 문서 전체에서 대상 독자·타겟층(예: 연령대, 이용자 집단)이 문단마다 다르게
   설정되어 있는 경우 - 이건 자연스러운 "확장"이 아니라 "불일치"이니 반드시
   지적하세요.
단순히 문체가 다르거나 수치/일정표처럼 원래 그런 문단이라면 지적하지 마세요.

[전체 문단]
{numbered}

다음 JSON 형식으로만 응답하세요:
{{
  "candidates": [
    {{"index": 문단 번호, "issue": "무엇이 왜 안 맞는지 한 문장"}}
  ]
}}
이상이 없으면 candidates를 빈 배열로 두세요."""


def _call_context_check_llm(prompt: str, temperature: float = 0) -> str:
    # 기본 temperature=0: 판정(맞다/아니다) 단계는 매번 흔들리면 안 되므로 고정한다.
    # 단, 맥락 이상 "찾기" 단계는 일부러 temperature를 올려 여러 번 샘플링한다 -
    # temp=0으로 여러 번 돌리면 매번 같은 답이라 union이 무의미하고, 미묘한 이상은
    # 실행마다 잡혔다 안 잡혔다 하기 때문(재인/Claude 2026-07-22 실측). 다양하게 탐색해
    # 후보를 모은 뒤 2단계 검증(temp=0)으로 걸러 recall과 precision을 모두 챙긴다.
    profile = (settings.LLM_PROFILE or "dev").lower()
    model = settings.QUALITY_LLM_REVIEWER_MODEL if profile == "quality" else settings.DEV_LLM_REVIEWER_MODEL
    client = trace_openai_client(OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=1))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return resp.choices[0].message.content


# 맥락 이상 감지: "찾기"를 여러 번 샘플링해 문단별 득표수로 판단한다(self-consistency 투표).
# 재인/Claude(2026-07-22) 실측: 찾기 단계는 temp 0.5~0.8에서 7/7로 진짜 이상 2개를 매번
# 다 찾고 헛것은 0표였다. 반면 별도 2차 검증(verify)은 진짜 이상(원예 취미 클래스)을
# 들쭉날쭉 기각해 false negative를 만들었다 - 그래서 verify를 버리고, 여러 샘플이 동의하면
# (득표수 >= 임계) 채택하는 투표로 바꿨다. 노이즈는 어쩌다 1표라 임계에서 걸러진다.
_CONTEXT_FIND_SAMPLES = 3
_CONTEXT_FIND_TEMPERATURE = 0.5
_CONTEXT_VOTE_THRESHOLD = 2  # 3표 중 2표 이상(다수결) 지목되면 이상으로 확정


# 재인/Claude(2026-07-22): "오탈자 검사" — 맥락 이상 감지 만들 때(2026-07-21) 사용자가
# 보류해뒀던 기능. 맥락 이상 감지와 달리 오탈자 판정은 비교적 명확해서(창작적 해석의
# 여지가 적음) 2단계(후보 찾기+재검증) 없이 LLM 1회 호출로 끝낸다(사용자 확인).
#
# 오탐 방지: 고유명사/전문용어/직접 인용문까지 오탈자로 잘못 지적하는 걸 막기 위해,
# 프롬프트에 "지적하지 말아야 할 것"을 명시적으로 나열하고 "확실하지 않으면 지적하지
# 말라"는 원칙을 넣었다(맥락 이상 감지의 2차 검증 단계에서 쓴 것과 같은 방향의 안전장치
# - 거기서도 "관련이 있을 수도 있다는 이유로 봐주지 마세요"처럼 명확한 기준을 줄수록
# LLM 판단이 안정적이었다).
class TypoFinding(BaseModel):
    id: str
    quote: str
    corrected: str
    message: str


class TypoCheckResponse(BaseModel):
    findings: list[TypoFinding]


# PDF 원문은 줄바꿈 지점의 띄어쓰기 정보를 잃는다(_extract_pdf_paragraphs 주석 참고).
# 그래서 PDF일 때는 skip_spacing=True로 띄어쓰기 관련 지적을 아예 금지하고, 글자 자체가
# 틀린 오탈자만 보게 한다 - 안 그러면 줄바꿈 아티팩트를 죄다 "띄어쓰기 오류"로 잡아
# 실측상 20건 넘는 오탐이 쏟아진다. docx 등은 원문 띄어쓰기가 그대로 보존되므로 그대로
# 띄어쓰기까지 검사한다.
def _build_typo_check_prompt(paragraphs: list[str], skip_spacing: bool = False) -> str:
    numbered = "\n\n".join(
        f"{i + 1}. {text[:400]}" for i, text in enumerate(paragraphs[:_CONTEXT_CHECK_MAX_PARAGRAPHS_IN_PROMPT])
    )
    scope = "글자 자체가 틀린 오탈자·맞춤법 오류만" if skip_spacing else "오탈자·맞춤법 오류·띄어쓰기 오류만"
    spacing_rule = (
        "- 띄어쓰기(간격) 문제는 절대 지적하지 마세요 - 이 문서는 원문 간격 정보가 정확하지 "
        "않아 띄어쓰기 판단이 불가능합니다. 오직 글자 자체가 틀린 경우만 보세요.\n"
        if skip_spacing else ""
    )
    return f"""당신은 사업계획서를 꼼꼼히 교정하는 편집자입니다. 아래 문단들에서
{scope} 하나도 빠짐없이 찾으세요. 오탈자를 놓치면 제출 문서에 그대로 남아 큰 흠이 되므로,
글자가 틀린 오탈자·맞춤법 오류는 조금이라도 의심되면 적극적으로 지적하세요.

단, 다음은 오탈자가 아니므로 지적하지 마세요 (띄어쓰기 교정도 절대 하지 마세요):
- 고유명사·기관명·서비스명·플랫폼명 (예: 공공데이터포털, 워크넷, HRD-Net)
- 전문용어·기술용어·업계 용어·합성어·외래어 표기 (예: 협업필터링, 언어모델, 임베딩)
- 따옴표로 직접 인용한 원문 그대로로 보이는 부분
- 문체상 의도적인 표현이나 반복
- 개조식(불릿·항목) 문장이 서술어 없이 짧게 끝나는 것 (원래 그렇게 쓰는 형식이므로 정상)
- 숫자와 단위·횟수는 붙여 쓰는 게 맞습니다 (예: '주 2회', '3명', '5개' — 절대 띄우라고 하지 마세요)
{spacing_rule}위 제외 대상만 아니라면, 글자 오탈자는 애매해도 일단 지적하는 편이 낫습니다 - 사용자가 판단하면 됩니다.

[전체 문단]
{numbered}

다음 JSON 형식으로만 응답하세요:
{{
  "typos": [
    {{"index": 문단 번호, "wrong": "틀린 부분 그대로(문단 원문에서 정확히 복사)", "corrected": "올바른 표기", "reason": "왜 틀렸는지 한 문장"}}
  ]
}}
오탈자가 없으면 typos를 빈 배열로 두세요."""


# PDF에서 추출한 텍스트에는 워드 자동 글머리 기호 등이 사설 영역(PUA, U+E000~U+F8FF)
# 유니코드로 렌더링돼 들어오는데(예: U+F0B7), 이건 사용자가 틀린 게 아니라 PDF 추출
# 아티팩트라 오탈자로 보여주면 안 된다 - wrong/corrected에 이 영역 글자가 섞이면 버린다.
_PUA_PATTERN = re.compile("[-]")


# 오탈자 검사 배치 크기: 한 번의 LLM 호출에 넣는 문단 수. 작을수록 recall이 높지만
# 호출 수가 늘어난다(3이면 문단이 많아도 배치들을 병렬 호출해 지연은 크지 않다).
_TYPO_BATCH_SIZE = 3


@router.get("/{project_id}/typo-check", response_model=TypoCheckResponse)
async def get_typo_check(
    project_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    document = await _find_target_document(project_id)
    if document is None:
        return TypoCheckResponse(findings=[])

    cached = document.get("typo_check_cache")
    if cached:
        logger.info("[typo-check] document_id=%s 캐시된 결과 재사용", document["_id"])
        return TypoCheckResponse(**cached)

    document_id = document["_id"]
    file_path = document.get("file_path")

    async def _cache_and_return(response: TypoCheckResponse) -> TypoCheckResponse:
        await document_repo.update_fields(document_id, {"typo_check_cache": response.model_dump()})
        return response

    if not file_path or not os.path.exists(file_path):
        return await _cache_and_return(TypoCheckResponse(findings=[]))

    paragraphs = await run_in_threadpool(_extract_body_paragraphs, file_path)
    if not paragraphs:
        return await _cache_and_return(TypoCheckResponse(findings=[]))

    all_paragraphs = paragraphs[:_CONTEXT_CHECK_MAX_PARAGRAPHS_IN_PROMPT]

    # PDF는 줄바꿈 지점의 띄어쓰기 정보를 잃으므로 띄어쓰기 판정을 끈다(위 주석 참고).
    skip_spacing = os.path.splitext(file_path)[1].lower() == ".pdf"

    logger.info(
        "[typo-check] document_id=%s 문서 파싱 완료 - 문단 %d개, %d개씩 묶어 배치 병렬 LLM 호출 시작",
        document_id, len(all_paragraphs), _TYPO_BATCH_SIZE,
    )
    _typo_start = time.monotonic()

    # 재인/Claude(2026-07-22): 문단을 한꺼번에 다 주면 LLM 주의가 분산돼 한 문단의 두 번째
    # 오탈자를 놓치는 걸 실측으로 확인했다(전체 25문단 → "구지자" 놓침, 그 문단만 단독으로
    # 주면 잡음). 그래서 작은 배치(_TYPO_BATCH_SIZE)로 나눠 병렬 호출한다 - recall이 크게
    # 오르고, 배치끼리 독립이라 asyncio.gather로 동시에 부를 수 있다. 결과는 캐싱되므로
    # 호출 수가 늘어도 문서당 한 번만 든다. index는 배치 내부 1-based라 offset을 더해
    # 전체 문단 번호로 복원한다.
    async def _run_batch(offset: int, batch: list[str]) -> list[tuple[int, dict]]:
        raw = await run_in_threadpool(
            _call_context_check_llm, _build_typo_check_prompt(batch, skip_spacing=skip_spacing)
        )
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        out: list[tuple[int, dict]] = []
        for item in (parsed.get("typos") if isinstance(parsed, dict) else None) or []:
            idx = item.get("index") if isinstance(item, dict) else None
            if isinstance(idx, int) and 1 <= idx <= len(batch):
                out.append((offset + idx, item))  # 전체 문단 기준 1-based
        return out

    batches = [
        (offset, all_paragraphs[offset:offset + _TYPO_BATCH_SIZE])
        for offset in range(0, len(all_paragraphs), _TYPO_BATCH_SIZE)
    ]
    batch_results = await asyncio.gather(*(_run_batch(off, b) for off, b in batches))
    raw_candidate_count = sum(len(batch) for batch in batch_results)
    logger.info(
        "[typo-check] document_id=%s LLM 응답 완료(%dms, %d개 배치 병렬 호출) - 오탈자 후보 %d건, "
        "중복/공백만 다름/PUA 문자 필터링 시작",
        document_id, int((time.monotonic() - _typo_start) * 1000), len(batches), raw_candidate_count,
    )

    findings: list[TypoFinding] = []
    seen: set[tuple[int, str]] = set()
    for global_index, item in [pair for batch in batch_results for pair in batch]:
        wrong = str(item.get("wrong") or "").strip()
        corrected = str(item.get("corrected") or "").strip()
        reason = str(item.get("reason") or "").strip()
        # wrong이 실제로 그 문단 원문에 있는 부분 문자열이어야 PDF에서 정확히 하이라이트
        # 가능하다(하이라이트 매칭은 quote를 원문에서 찾아 위치를 계산하는 방식이라,
        # LLM이 지어낸 문구면 매칭 자체가 실패한다) - 없으면 이 후보는 버린다.
        if not wrong or not corrected or wrong not in all_paragraphs[global_index - 1]:
            continue
        # 고친 게 원문과 같으면(진짜 수정이 없음) 보여줄 게 없다 - 버린다.
        if wrong == corrected:
            continue
        # 사설 영역(PUA) 글자가 섞인 건 PDF 추출 아티팩트(워드 글머리 기호 등)라 사용자
        # 오탈자가 아니다 - 보여주지 않는다.
        if _PUA_PATTERN.search(wrong) or _PUA_PATTERN.search(corrected):
            continue
        # 띄어쓰기만 다른 지적(글자는 같고 공백만 다름)은 포맷과 무관하게 버린다.
        # 프롬프트로 "전문용어·합성어는 띄어쓰기 건드리지 마라"를 예시까지 넣어 지시해도
        # LLM이 협업필터링·공공데이터포털 같은 걸 계속 띄우라고 하는 걸 실측으로 확인했다
        # (한국어 띄어쓰기 판정을 LLM이 신뢰성 있게 못 함). 반면 글자가 실제로 바뀌는
        # 오탈자(역활→역할, 구지자→구직자)는 이 필터에 안 걸리고 그대로 남는다. 즉 이
        # 검사는 "글자 오탈자"에 집중하고 띄어쓰기는 보지 않는다(skip_spacing 여부 무관).
        if re.sub(r"\s", "", wrong) == re.sub(r"\s", "", corrected):
            continue
        dedupe_key = (global_index, wrong)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        findings.append(TypoFinding(
            id=f"typo-{global_index}-{len(findings)}",
            quote=wrong,
            corrected=corrected,
            message=reason or f"'{wrong}' → '{corrected}'",
        ))

    logger.info(
        "[typo-check] document_id=%s 필터링 완료 - 최종 오탈자 %d건 확정 (후보 %d건 중)",
        document_id, len(findings), raw_candidate_count,
    )
    for f in findings:
        logger.info(
            "[typo-check]   -> '%s' → '%s' (%s)",
            f.quote, f.corrected, f.message,
        )
    return await _cache_and_return(TypoCheckResponse(findings=findings))


@router.get("/{project_id}/context-check", response_model=ContextCheckResponse)
async def get_context_check(
    project_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    document = await _find_target_document(project_id)
    if document is None:
        return ContextCheckResponse(findings=[])

    cached = document.get("context_check_cache")
    if cached:
        logger.info("[context-check] document_id=%s 캐시된 결과 재사용", document["_id"])
        return ContextCheckResponse(**cached)

    document_id = document["_id"]
    file_path = document.get("file_path")

    async def _cache_and_return(response: ContextCheckResponse) -> ContextCheckResponse:
        await document_repo.update_fields(document_id, {"context_check_cache": response.model_dump()})
        return response

    if not file_path or not os.path.exists(file_path):
        return await _cache_and_return(ContextCheckResponse(findings=[]))

    paragraphs = await run_in_threadpool(_extract_body_paragraphs, file_path)
    if len(paragraphs) < _CONTEXT_CHECK_MIN_PARAGRAPHS:
        return await _cache_and_return(ContextCheckResponse(findings=[]))

    numbered_paragraphs = paragraphs[:_CONTEXT_CHECK_MAX_PARAGRAPHS_IN_PROMPT]

    logger.info(
        "[context-check] document_id=%s 문단 %d개 추출 완료 - GPT가 문서 전체를 한 번에 읽고 "
        "temperature=%.1f로 %d회 독립 샘플링(다수결 투표, 임계값 %d표) 시작",
        document_id, len(numbered_paragraphs), _CONTEXT_FIND_TEMPERATURE,
        _CONTEXT_FIND_SAMPLES, _CONTEXT_VOTE_THRESHOLD,
    )
    _context_start = time.monotonic()

    # 찾기를 temperature를 올려 여러 번 샘플링하고, 문단별 득표수로 판단한다(투표).
    # 각 샘플에서 지목된 문단 번호를 세고, 대표 issue 문구는 처음 나온 것을 쓴다.
    find_prompt = _build_context_find_prompt(paragraphs)

    async def _find_once() -> list[dict]:
        raw = await run_in_threadpool(
            _call_context_check_llm, find_prompt, _CONTEXT_FIND_TEMPERATURE
        )
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        return [
            c for c in ((parsed.get("candidates") if isinstance(parsed, dict) else None) or [])
            if isinstance(c, dict) and isinstance(c.get("index"), int) and 1 <= c["index"] <= len(numbered_paragraphs)
        ]

    sampled = await asyncio.gather(*(_find_once() for _ in range(_CONTEXT_FIND_SAMPLES)))
    votes: dict[int, int] = {}
    issue_by_index: dict[int, str] = {}
    for cand_list in sampled:
        for index in {c["index"] for c in cand_list}:  # 한 샘플 내 중복은 1표로
            votes[index] = votes.get(index, 0) + 1
        for c in cand_list:
            issue_by_index.setdefault(c["index"], str(c.get("issue") or "").strip())

    logger.info(
        "[context-check] document_id=%s 샘플링 완료(%dms) - %d회 중 %d개 문단이 최소 1표 이상 득표 "
        "(문단별 득표: %s)",
        document_id, int((time.monotonic() - _context_start) * 1000), _CONTEXT_FIND_SAMPLES,
        len(votes), {k: v for k, v in sorted(votes.items())},
    )
    for idx, issue in issue_by_index.items():
        logger.info(
            "[context-check]   -> %d번 문단(%d표) 후보 사유: %s | 원문: %s",
            idx, votes.get(idx, 0), issue, numbered_paragraphs[idx - 1][:80],
        )

    findings: list[ContextFinding] = []
    for index in sorted(votes):
        if votes[index] < _CONTEXT_VOTE_THRESHOLD:
            continue  # 다수결 미달(어쩌다 한 번 지목된 노이즈)은 버린다
        message = issue_by_index.get(index) or "문서 전체 맥락과 어긋나는 문단입니다."
        paragraph_index = index - 1
        findings.append(ContextFinding(
            id=f"context-{paragraph_index}",
            quote=numbered_paragraphs[paragraph_index],
            message=message,
        ))

    logger.info(
        "[context-check] document_id=%s 다수결 판정 완료 - 임계값(%d표) 이상 득표한 맥락 이상 %d건 확정",
        document_id, _CONTEXT_VOTE_THRESHOLD, len(findings),
    )
    return await _cache_and_return(ContextCheckResponse(findings=findings))


# 재인/Claude(2026-07-22): "분량·밀도 체크" — 팀 요구사항.
# 심사에서 "공고문이 15페이지로 쓰라고 했는데 14페이지만 쓰면 성의 없어 보인다",
# "페이지 수만 맞추고 줄바꿈·여백을 많이 남겨서 채우면 싫어한다"는 실무 감각을
# 자동으로 체크한다. 두 가지를 본다:
#   (1) 분량: 공고문에서 요구 페이지 수를 뽑아 실제 문서 페이지 수와 비교.
#   (2) 밀도: 페이지마다 텍스트가 세로로 얼마나 차 있는지(여백 비율)를 재서, 절반 넘게
#       빈 페이지를 "여백 많음"으로 표시. 마지막 페이지는 원래 짧게 끝나는 게 정상이라 제외.
# 오탈자/맥락 이상과 같은 캐싱(target 문서에 format_check_cache) 방식이다.

# 공고문에서 요구 분량(페이지)을 뽑을 때 LLM에 넣는 원문 최대 길이. 분량 기준은 보통
# 제출/작성 안내 근처에 있어 앞부분으로 충분하지만, 넉넉히 준다.
_PAGE_REQ_MAX_CHARS = 12000
# 이 비율 미만이면 그 "개별 페이지"가 여백 많음으로 본다. 재인/Claude(2026-07-22)
# 실측(벡터 도형 포함 전, 구간 병합 지표 기준): 빽빽한 실제 문서(BioFitRAG·WageGuard)는
# 평균 52~55%, 보통 문서(리스킬·도서관리커넥트·톡스탠드)는 39~40%로 나뉘어, 45%는
# "보통도 걸리고 빽빽해야 통과"하는 까다로운 기준이다 - 사용자 확인(2026-07-22) "45로 해"로
# 확정. 이후 벡터 도형(색깔박스·표·다이어그램)을 커버리지에 포함하도록 지표를 고친 뒤에도
# 이 값은 그대로 유효했다(실제 대상 수상작 SchoolBridge 최저 페이지 41.5%가 45% 바로
# 아래라 딱 하나만 걸리고, 평범한 문서의 진짜 빈 페이지 10~30%대는 확실히 걸러짐).
_DENSITY_SPARSE_THRESHOLD = 0.45

# 이 비율 미만이면 "문서 전체"가 전반적으로 여백이 많다고 본다(개별 페이지가 아니라
# 평균 채움률 기준). 재인/Claude(2026-07-22) 실측(벡터 도형 포함 지표): 실제 대상
# 수상작(SchoolBridge) 평균 66.8% vs 평범한 실제 사업계획서(WageGuard) 평균 52.7% -
# 데이터 2건뿐이라 정확한 경계는 불확실하나, 사용자 확인(2026-07-22) "63%로, 수상작에
# 근접하게 까다롭게"로 확정. 평범한 문서는 확실히 걸리고 수상작 수준이어야 통과한다.
_DENSITY_OVERALL_THRESHOLD = 0.63

class PageRequirement(BaseModel):
    # 단일 기준이면 min==max, 범위("10~30매")면 min<max. 둘 다 없으면 요구 없음.
    required_min: Optional[int] = None
    required_max: Optional[int] = None
    source_text: Optional[str] = None


class PageDensity(BaseModel):
    page: int
    coverage: float  # 0~1, 페이지 세로 대비 텍스트가 차지한 비율


class FormatCheckResponse(BaseModel):
    required_min: Optional[int] = None
    required_max: Optional[int] = None
    required_source: Optional[str] = None
    actual_pages: Optional[int] = None
    page_verdict: Optional[str] = None  # "부족"/"충족"/"초과"/None(기준 없음)
    page_message: Optional[str] = None
    overall_coverage: Optional[float] = None  # 전체 평균 밀도(0~1)
    overall_verdict: Optional[str] = None  # "양호"/"부족"/None(판정 대상 없음)
    sparse_pages: list[int] = []
    density_message: Optional[str] = None


async def _find_criteria_document(project_id: str) -> Optional[dict]:
    documents = await document_repo.find_by_project_id(project_id)
    criteria = [d for d in documents if d.get("document_role") == "criteria" and d.get("parsed_text")]
    return criteria[0] if criteria else None


def _build_page_req_prompt(criteria_text: str) -> str:
    return f"""당신은 공고문(모집요강)에서 "제출할 사업계획서·기획서·제안서 본문의 요구 분량
(페이지 수)"만 정확히 찾아내는 도우미입니다.

규칙:
- 제출 본문(사업계획서/기획서/제안서/응모원고)의 페이지·쪽·매 기준만 찾으세요.
- 별첨·첨부·참고자료·다른 서류(동의서, 서약서, 매뉴얼 등)의 분량은 무시하세요.
- "A4 10~30매"처럼 범위면 min과 max를 모두, "15페이지 이내"처럼 단일이면 min·max를 같은
  값으로 채우세요.
- 분량 기준이 명시돼 있지 않으면 min·max를 모두 null로 두세요. 절대 추측하지 마세요.
- "3장 구성"의 '장'(챕터)이나 연도·개수·인원 같은 건 분량이 아닙니다.

[공고문]
{criteria_text[:_PAGE_REQ_MAX_CHARS]}

다음 JSON 형식으로만 응답하세요:
{{"min": 숫자 또는 null, "max": 숫자 또는 null, "source": "근거가 된 원문 구절을 그대로 인용(없으면 빈 문자열)"}}"""


def _extract_required_pages(criteria_text: str) -> PageRequirement:
    """공고문에서 제출 본문의 요구 분량을 LLM으로 뽑는다. 범위면 min<max, 단일이면
    min==max, 없거나 비정상이면 둘 다 None. 정규식은 "별첨 분량/조사/총 N페이지" 같은
    문맥을 구분 못 해 오탐·누락이 있어 LLM으로 바꿨다(재인/Claude 2026-07-22 실측)."""
    raw = _call_context_check_llm(_build_page_req_prompt(criteria_text))
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return PageRequirement()
    if not isinstance(parsed, dict):
        return PageRequirement()

    def _num(v) -> Optional[int]:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            n = int(v)
            return n if 1 <= n <= 300 else None
        return None

    lo, hi = _num(parsed.get("min")), _num(parsed.get("max"))
    if lo is None and hi is None:
        return PageRequirement()
    # 한쪽만 있으면 단일 기준으로 본다.
    lo = lo if lo is not None else hi
    hi = hi if hi is not None else lo
    if lo > hi:
        lo, hi = hi, lo
    source = str(parsed.get("source") or "").strip() or None
    return PageRequirement(required_min=lo, required_max=hi, source_text=source)


def _analyze_pdf_format(pdf_path: Path) -> tuple[int, list[PageDensity]]:
    """PDF의 페이지 수와 페이지별 세로 커버리지(밀도)를 계산한다.

    커버리지 = "내용(텍스트+이미지+색깔박스·표 등 벡터 도형)이 세로로 차지한 구간"의
    길이 / 페이지 높이. 블록 높이를 단순 합산하지 않고 겹치는 세로 구간을 병합해서
    재는 이유: (1) 표·다단 레이아웃은 블록이 세로로 겹쳐 단순 합산 시 100%를 넘어버린다,
    (2) 차트·다이어그램으로 꽉 찬 페이지는 텍스트만 세면 '여백 많음'으로 오판되므로 이미지
    블록도 내용으로 포함해야 한다(재인/Claude 2026-07-22 실측 대응).

    벡터 도형(색깔 박스·표 배경·테두리)도 포함하는 이유: 실측(2026-07-22)으로 확인한
    실제 대상 수상작(SchoolBridge)에서, 텍스트+이미지만 셌을 때 색깔 박스·표·다이어그램이
    많은 페이지(도형 40~66개)가 오히려 '여백 많음'으로 잘못 판정됐다 - 디자인 요소가 많은
    페이지일수록 불리해지는 정반대 결과였다. get_drawings()로 사각형/선 등 채워진 도형의
    세로 범위도 같은 방식으로 병합해 넣는다."""
    doc = fitz.open(pdf_path)
    try:
        densities: list[PageDensity] = []
        for i, page in enumerate(doc):
            page_height = page.rect.height or 1.0
            # 텍스트(type 0)·이미지(type 1)·벡터 도형(색깔박스·표 등) 모두 '내용'으로 보고
            # 세로 구간을 모은다.
            drawing_intervals = [
                (d["rect"].y0, d["rect"].y1)
                for d in page.get_drawings()
                if d.get("rect") and d["rect"].y1 > d["rect"].y0
            ]
            intervals = sorted(
                [
                    (b["bbox"][1], b["bbox"][3])
                    for b in page.get_text("dict").get("blocks", [])
                    if b.get("bbox") and b["bbox"][3] > b["bbox"][1]
                ]
                + drawing_intervals
            )
            filled = 0.0
            cur_start = cur_end = None
            for y0, y1 in intervals:
                if cur_end is None:
                    cur_start, cur_end = y0, y1
                elif y0 <= cur_end:  # 앞 구간과 겹치면 이어붙인다(이중계산 방지)
                    cur_end = max(cur_end, y1)
                else:
                    filled += cur_end - cur_start
                    cur_start, cur_end = y0, y1
            if cur_end is not None:
                filled += cur_end - cur_start
            coverage = min(1.0, filled / page_height)
            densities.append(PageDensity(page=i + 1, coverage=round(coverage, 3)))
        return len(doc), densities
    finally:
        doc.close()


@router.get("/{project_id}/format-check", response_model=FormatCheckResponse)
async def get_format_check(
    project_id: str,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    await verify_project_owner(project_id, user_email)

    target = await _find_target_document(project_id)
    if target is None:
        return FormatCheckResponse()

    cached = target.get("format_check_cache")
    if cached:
        logger.info("[format-check] document_id=%s 캐시된 결과 재사용", target["_id"])
        return FormatCheckResponse(**cached)

    document_id = target["_id"]

    async def _cache_and_return(response: FormatCheckResponse) -> FormatCheckResponse:
        await document_repo.update_fields(document_id, {"format_check_cache": response.model_dump()})
        return response

    # (1) 공고문에서 요구 페이지 수 (LLM 추출이라 스레드풀에서 호출)
    criteria = await _find_criteria_document(project_id)
    logger.info(
        "[format-check] document_id=%s 공고문(criteria) 문서=%s(%s) 참고해 요구 분량 LLM 추출 시작 "
        "(원문 앞 %d자만 프롬프트에 사용)",
        document_id, criteria.get("_id") if criteria else "없음",
        (criteria.get("original_filename") or criteria.get("source_url") or "") if criteria else "요구 분량 판정 생략",
        _PAGE_REQ_MAX_CHARS,
    )
    _page_req_start = time.monotonic()
    page_req = (
        await run_in_threadpool(_extract_required_pages, criteria["parsed_text"])
        if criteria else PageRequirement()
    )
    logger.info(
        "[format-check] document_id=%s 요구 분량 추출 완료(%dms): min=%s max=%s 근거=\"%s\"",
        document_id, int((time.monotonic() - _page_req_start) * 1000),
        page_req.required_min, page_req.required_max, (page_req.source_text or "")[:80],
    )

    # (2) 실제 문서를 PDF로 (docx 등이면 변환) 열어 페이지 수 + 밀도
    file_path = target.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return await _cache_and_return(FormatCheckResponse(
            required_min=page_req.required_min, required_max=page_req.required_max,
            required_source=page_req.source_text))

    def _measure() -> tuple[int, list[PageDensity]]:
        src = Path(file_path)
        if src.suffix.lower() == ".pdf":
            return _analyze_pdf_format(src)
        out_dir = HwpConversionConfig().resolve_temp_dir() / "format_check"
        pdf_path = convert_to_preview_pdf(src, output_dir=out_dir)
        try:
            return _analyze_pdf_format(pdf_path)
        finally:
            pdf_path.unlink(missing_ok=True)

    logger.info(
        "[format-check] document_id=%s 원본(%s)을 PDF로 열어 페이지 수·페이지별 밀도 측정 시작",
        document_id, Path(file_path).suffix,
    )
    try:
        actual_pages, densities = await run_in_threadpool(_measure)
    except DocumentConversionError:
        logger.exception("[format-check] document_id=%s 변환 실패", document_id)
        return await _cache_and_return(FormatCheckResponse(
            required_min=page_req.required_min, required_max=page_req.required_max,
            required_source=page_req.source_text))
    logger.info(
        "[format-check] document_id=%s 측정 완료 - 실제 %d페이지, 페이지별 커버리지=%s",
        document_id, actual_pages, [d.coverage for d in densities],
    )

    # 분량 판정 - 목표는 "상한(required_max)을 채우는 것". 범위(10~30)라도 상한에 못 미치면
    # 성의 부족으로 보므로, 상한을 기준으로 "부족/충족/초과"를 낸다(사용자 확인: 10~30이면
    # 30을 채워야 한다). 하한 미달은 더 강하게 경고한다.
    page_verdict = page_message = None
    lo, hi = page_req.required_min, page_req.required_max
    if hi:
        req_label = f"{lo}~{hi}페이지" if lo != hi else f"{hi}페이지"
        if lo and actual_pages < lo:
            page_verdict = "부족"
            page_message = (f"공고문 기준은 {req_label}인데 문서는 {actual_pages}페이지로 최소 분량({lo}p)에도 "
                            f"미치지 못합니다. 분량이 크게 적으면 성의가 부족해 보입니다.")
        elif actual_pages < hi:
            page_verdict = "부족"
            page_message = (f"공고문 기준은 {req_label}입니다. 현재 {actual_pages}페이지인데, 상한인 "
                            f"{hi}페이지에 가깝게 채우는 게 좋습니다 — 분량을 덜 채우면 성의가 부족해 보일 수 있습니다.")
        elif actual_pages > hi:
            page_verdict = "초과"
            page_message = f"공고문 기준은 {req_label}인데 문서는 {actual_pages}페이지로 {actual_pages - hi}페이지 초과했습니다."
        else:
            page_verdict = "충족"
            page_message = f"공고문 기준 {req_label}에 맞게 충분히 작성되었습니다."

    # 밀도 판정 - 마지막 페이지는 원래 짧게 끝나는 게 정상이라 판정에서 제외한다.
    # 두 기준을 같이 본다: (1) 개별 페이지가 너무 비었는지(sparse_pages, 45%), (2) 문서
    # 전체 평균이 실제 수상작 수준으로 빽빽한지(overall_verdict, 63% - 사용자 확인
    # 2026-07-22 "평균을 봐야 하지 않나" 요청으로 추가).
    judged = densities[:-1] if len(densities) > 1 else densities
    sparse_pages = [d.page for d in judged if d.coverage < _DENSITY_SPARSE_THRESHOLD]
    overall = round(sum(d.coverage for d in judged) / len(judged), 3) if judged else None
    overall_verdict = None
    density_message = None
    if overall is not None:
        overall_verdict = "양호" if overall >= _DENSITY_OVERALL_THRESHOLD else "부족"
        parts = [f"문서 전체 평균 채움률은 {round(overall * 100)}%입니다({overall_verdict})."]
        if overall_verdict == "부족":
            parts.append("실제 수상작 수준(약 65~70%)보다 여백이 많은 편이라, 내용을 더 채우는 게 좋습니다.")
        if sparse_pages:
            parts.append(f"그중 {', '.join(map(str, sparse_pages))}페이지는 특히 절반 넘게 비어 있습니다.")
        density_message = " ".join(parts)

    logger.info(
        "[format-check] document_id=%s 최종 판정 - 분량: %s, 밀도: 평균 %s%%(%s), 여백 많음 페이지=%s",
        document_id, page_verdict,
        round(overall * 100) if overall is not None else None, overall_verdict, sparse_pages,
    )
    return await _cache_and_return(FormatCheckResponse(
        required_min=page_req.required_min,
        required_max=page_req.required_max,
        required_source=page_req.source_text,
        actual_pages=actual_pages,
        page_verdict=page_verdict,
        page_message=page_message,
        overall_coverage=overall,
        overall_verdict=overall_verdict,
        sparse_pages=sparse_pages,
        density_message=density_message,
    ))
