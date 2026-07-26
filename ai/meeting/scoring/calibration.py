"""제출문서의 구체적 근거 신호로 위원 제안 점수의 상한을 계산한다.

LLM은 평가와 피드백을 담당하지만, 근거가 빈약한 문서에도 중간 점수를 주는 경향이 있다.
이 모듈은 문서 텍스트만으로 재현 가능한 신호를 판별해 항목별 점수 상한을 반환한다.
상한만 적용하므로 위원이 이미 더 낮게 준 점수를 올리지는 않는다.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

_QUANT = Decimal("0.01")

# 명세의 예시는 기본 사전일 뿐이다. 아래의 일반 기관명 패턴도 함께 사용하므로 다른
# 공모전의 데이터셋·기관명도 인식한다. 공공데이터포털/data.go.kr은 데이터의 입수
# 경로이지 실제 데이터셋·보유기관이 아니므로 구체 출처로 세지 않는다.
DATA_SOURCE_TERMS = (
    "hrd-net",
    "hrdnet",
    "워크넷",
    "고용보험",
    "국가기술자격",
    "내일배움카드",
    "한국고용정보원",
    "한국산업인력공단",
    "고용노동부",
    "통계청",
    "국가통계포털",
    "kosis",
    "keis",
)

METHOD_TERMS = (
    "임베딩",
    "embedding",
    "모델",
    "model",
    "알고리즘",
    "algorithm",
    "파이프라인",
    "pipeline",
    "api",
    "정규화",
    "전처리",
    "재정렬",
    "rerank",
    "학습",
    "추론",
    "벡터",
    "vector",
    "아키텍처",
    "architecture",
    "데이터베이스",
    "database",
    "생성형 ai",
    "llm",
)

_FUTURE_RE = re.compile(
    r"(?:예정|계획(?:이다|한다|하고|으로)?|할\s*것이다|될\s*것이다|"
    r"가능할\s*것이다|기대(?:된다|한다)|추진할|도입할|구축할|"
    r"앞으로|나중|추후|차차|향후|하겠다|나가겠다|정하겠다|만들겠다|"
    r"있을\s*것이다|할\s*수\s*있)"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?:[.!?。！？]+|\n+)")
_QUANTITATIVE_RE = re.compile(
    r"(?:\d+(?:[.,]\d+)?\s*(?:%|퍼센트|배|명|원|건|개|개월|년|월|일|회|단계|위|점))"
    r"|(?:\d+\s*(?::~|~|-|:|/)\s*\d+)"
    r"|(?:top\s*\d+)"
    r"|(?:수\s*(?:시간|분|개월|년|명|건|개))"
    r"|(?:수십|수백|수천)\s*(?:명|건|개|원|시간|분)?"
    r"|(?:(?:한|두|세|네|다섯|열)\s*(?:배|명|건|개|개월|단계))",
    re.IGNORECASE,
)
_NAMED_ORGANIZATION_RE = re.compile(
    r"(?:[가-힣A-Za-z0-9-]{2,}(?:부|청|공단|공사|연구원|진흥원|정보원|포털))"
)
_PLACEHOLDER_TEXTS = {"", ".", "..", "...", "…"}

_DATA_HINTS = ("data", "데이터", "자료 활용", "정보 활용")
_AI_HINTS = ("ai_", "ai ", "인공지능", "ai혁신", "ai 혁신")
_FEASIBILITY_HINTS = ("feasibility", "실현", "구현", "기술 가능", "실행 가능")
_CREATIVITY_HINTS = ("creativ", "창의", "독창", "차별", "혁신성")
_EFFECT_HINTS = ("effect", "impact", "효과", "성과", "파급", "기여")
_NEGATED_METHOD_RE = re.compile(
    r"(?:방법|기술|모델|알고리즘|학습|구현).{0,25}(?:정할|미정|추후|나중|차차|검토)"
)
_METHOD_ACTION_RE = re.compile(
    r"(?:사용|활용|적용|검색|추천|분석|수집|결합|구현|구축|정규화|재정렬|"
    r"추론|학습|적재|호출|생성|설명|처리|연결)"
)
_HEADING_RE = re.compile(
    r"^\s*(?:제?\s*\d+\s*(?:장|절|[.)])|\d+\s*[-.)]|[ivx]+\s*[.)])\s*",
    re.IGNORECASE,
)

_SECTION_ALIASES = {
    "data": ("데이터 활용", "자료 활용", "활용 데이터", "데이터의 활용"),
    "ai": ("ai 활용", "인공지능 활용", "ai 혁신", "인공지능 혁신"),
    "feasibility": (
        "실현 가능",
        "구현 가능",
        "실용성",
        "상세 설명",
        "추진 일정",
        "개발 계획",
        "사업화",
    ),
    "creativity": ("창의", "독창", "차별"),
    "effect": ("기대효과", "기대 효과", "성과", "파급효과", "정책 기여"),
}


def extract_submission_text(submission: dict[str, Any] | None) -> str:
    """파일명은 제외하고 실제 제출 본문만 꺼낸다."""
    if not submission:
        return ""
    for key in ("text", "content", "raw_text", "body"):
        value = submission.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None)
    return ""


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _is_data_criterion(criterion: dict[str, Any]) -> bool:
    label = _normalize(
        f'{criterion.get("criterion_id", "")} {criterion.get("criterion_name", "")}'
    )
    return any(hint in label for hint in _DATA_HINTS)


def _needs_method_evidence(criterion: dict[str, Any]) -> bool:
    label = _normalize(
        f'{criterion.get("criterion_id", "")} {criterion.get("criterion_name", "")}'
    )
    return any(hint in label for hint in (*_AI_HINTS, *_FEASIBILITY_HINTS))


def _criterion_kind(criterion: dict[str, Any]) -> str | None:
    label = _normalize(
        " ".join(
            str(criterion.get(key, ""))
            for key in ("criterion_id", "criterion_name", "description")
        )
    )
    if any(hint in label for hint in _DATA_HINTS):
        return "data"
    if any(hint in label for hint in _AI_HINTS):
        return "ai"
    if any(hint in label for hint in _FEASIBILITY_HINTS):
        return "feasibility"
    if any(hint in label for hint in _CREATIVITY_HINTS):
        return "creativity"
    if any(hint in label for hint in _EFFECT_HINTS):
        return "effect"
    return None


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    if _HEADING_RE.search(stripped):
        return True
    # "AI 혁신성", "기대 효과"처럼 평가축 자체만 적힌 짧은 제목도 허용한다.
    return len(stripped) <= 45 and any(
        alias in _normalize(stripped)
        for aliases in _SECTION_ALIASES.values()
        for alias in aliases
    )


def _split_sections(text: str) -> list[tuple[str, str]]:
    """DOCX parser가 보존한 줄 경계를 이용해 제목과 본문을 나눈다."""
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_heading(line):
            if current_heading or current_body:
                sections.append((current_heading, current_body))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)
    if current_heading or current_body:
        sections.append((current_heading, current_body))
    return [(heading, "\n".join(body)) for heading, body in sections]


def _criterion_evidence_text(text: str, criterion: dict[str, Any]) -> str:
    """criterion과 의미가 맞는 섹션만 반환하고, 못 찾을 때만 전체 본문으로 폴백한다."""
    kind = _criterion_kind(criterion)
    sections = _split_sections(text)
    if kind is not None:
        aliases = _SECTION_ALIASES[kind]
        matched = [
            body
            for heading, body in sections
            if "공모전" not in _normalize(heading)
            and "제안서" not in _normalize(heading)
            and any(alias in _normalize(heading) for alias in aliases)
        ]
        if matched:
            return "\n".join(matched)

    # 알려지지 않은 동적 평가축은 criterion 이름의 의미 토큰과 제목이 겹치는 섹션을 찾는다.
    label_tokens = {
        token
        for token in re.findall(
            r"[a-zA-Z]{3,}|[가-힣]{2,}",
            _normalize(str(criterion.get("criterion_name", ""))),
        )
        if token not in {"평가", "항목", "여부", "정도"}
    }
    matched = [
        body
        for heading, body in sections
        if label_tokens
        and any(token in _normalize(heading) for token in label_tokens)
    ]
    return "\n".join(matched) if matched else text


def _has_named_data_source(text: str) -> bool:
    if any(term in text for term in DATA_SOURCE_TERMS):
        return True
    text = text.replace("공공데이터포털", "").replace("data.go.kr", "")
    # 사전에 없는 기관도 "○○공단/○○부/○○정보원"처럼 이름이 명시되면 출처로 인정한다.
    return _NAMED_ORGANIZATION_RE.search(text) is not None


def _future_sentence_ratio(text: str) -> float:
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(text)
        if len(re.sub(r"\s+", "", sentence)) >= 6
    ]
    if not sentences:
        return 0.0
    future_count = sum(1 for sentence in sentences if _FUTURE_RE.search(sentence))
    return future_count / len(sentences)


def _has_quantitative_evidence(text: str) -> bool:
    # 공모전 제목의 "제5회"와 문서 목차 번호는 성과·규모·일정의 정량 근거가 아니다.
    without_ordinals = re.sub(r"제\s*\d+\s*회", "", text)
    return _QUANTITATIVE_RE.search(without_ordinals) is not None


def _needs_quantitative_evidence(criterion: dict[str, Any], kind: str | None) -> bool:
    """수치가 평가 근거인 항목에만 S2를 적용한다.

    AI 혁신성·창의성처럼 방법/차별 논리가 핵심인 항목까지 숫자 하나 없다는 이유로
    깎지 않는다. 알려지지 않은 동적 rubric은 항목명·설명의 정량 관련 표현을 따른다.
    """
    if kind in {"data", "feasibility", "effect"}:
        return True
    label = _normalize(
        " ".join(
            str(criterion.get(key, ""))
            for key in ("criterion_name", "description")
        )
    )
    return any(
        keyword in label
        for keyword in ("정량", "수치", "규모", "예산", "일정", "성과", "효과", "파급")
    )


def _has_concrete_method(text: str, kind: str | None) -> bool:
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(text)
        if sentence.strip()
    ]
    if kind == "feasibility":
        feasibility_terms = (
            "mvp",
            "프로토타입",
            "오픈api",
            "서버",
            "클라우드",
            "데이터베이스",
            "개발자",
            "아키텍처",
            "파이프라인",
            "개월차",
            "단계(",
        )
        terms = feasibility_terms
    else:
        terms = METHOD_TERMS

    for sentence in sentences:
        normalized_sentence = _normalize(sentence)
        if not any(term in normalized_sentence for term in terms):
            continue
        if kind == "ai" and _METHOD_ACTION_RE.search(normalized_sentence) is None:
            continue
        if _FUTURE_RE.search(normalized_sentence) or _NEGATED_METHOD_RE.search(normalized_sentence):
            continue
        return True
    return False


def _depth_cap_ratio(text: str, criterion_count: int) -> Decimal | None:
    """평가항목 수 대비 본문이 지나치게 짧으면 전 항목의 근거 깊이 상한을 둔다."""
    compact_length = len(re.sub(r"\s+", "", text))
    chars_per_criterion = compact_length / max(criterion_count, 1)
    if chars_per_criterion < 80:
        return Decimal("0.10")
    if chars_per_criterion < 300:
        return Decimal("0.20")
    if chars_per_criterion < 500:
        return Decimal("0.40")
    if chars_per_criterion < 650:
        return Decimal("0.64")
    return None


def _required_keywords_missing(
    criterion: dict[str, Any],
    rubric: dict[str, Any],
    text: str,
) -> bool:
    """동적 rubric이 필수 키워드를 제공한 경우에만 S5를 적용한다.

    required_keywords/mandatory_keywords는 모두 포함되어야 하는 단순 목록이고,
    required_keyword_groups는 각 그룹에서 하나 이상 포함되면 충족되는 OR 그룹이다.
    메타데이터가 없는 공모전에 필수요건을 임의로 만들지는 않는다.
    """
    top_level = (rubric.get("mandatory_keywords_by_criterion") or {}).get(
        criterion.get("criterion_id"), []
    )
    required = [
        *criterion.get("required_keywords", []),
        *criterion.get("mandatory_keywords", []),
        *top_level,
    ]
    if any(_normalize(str(keyword)) not in text for keyword in required):
        return True

    for group in criterion.get("required_keyword_groups", []):
        normalized_group = [_normalize(str(keyword)) for keyword in group]
        if normalized_group and not any(keyword in text for keyword in normalized_group):
            return True
    return False


def build_score_cap(
    rubric: dict[str, Any],
    criterion: dict[str, Any],
    submission: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """criterion의 결정론적 점수 상한과 발동 신호를 반환한다.

    테스트용 ``...`` 같은 placeholder는 실제 문서로 보지 않아 기존 호출의 하위호환을
    유지한다. 실제로 매우 짧은 문서는 최하단 앵커(S0)로 15% 상한을 적용한다.
    """
    raw_text = extract_submission_text(submission)
    normalized = _normalize(raw_text).strip()
    if normalized in _PLACEHOLDER_TEXTS:
        return None

    compact_length = len(re.sub(r"\s+", "", normalized))
    signals: list[dict[str, Any]] = []

    if compact_length < 120:
        signals.append(
            {
                "code": "S0",
                "cap_ratio": Decimal("0.15"),
                "reason": "평가 근거로 사용할 본문 분량이 지나치게 적음",
            }
        )

    depth_cap = _depth_cap_ratio(normalized, len(rubric.get("criteria", [])))
    if depth_cap is not None:
        signals.append(
            {
                "code": "S6",
                "cap_ratio": depth_cap,
                "reason": (
                    "평가항목 수에 비해 제출 본문의 실질 분량이 부족해 "
                    f"항목별 근거 깊이 상한 {int(depth_cap * 100)}%를 적용함"
                ),
            }
        )

    criterion_text = _criterion_evidence_text(normalized, criterion)
    kind = _criterion_kind(criterion)

    if _is_data_criterion(criterion) and not _has_named_data_source(criterion_text):
        signals.append(
            {
                "code": "S1",
                "cap_ratio": Decimal("0.25"),
                "reason": "구체적인 데이터셋 또는 기관 출처가 명시되지 않음",
            }
        )

    if _needs_quantitative_evidence(criterion, kind) and not _has_quantitative_evidence(criterion_text):
        signals.append(
            {
                "code": "S2",
                "cap_ratio": Decimal("0.64"),
                "reason": "숫자 또는 정량 표현이 없음",
            }
        )

    future_ratio = _future_sentence_ratio(criterion_text)
    if future_ratio >= 0.5:
        signals.append(
            {
                "code": "S3",
                "cap_ratio": Decimal("0.64"),
                "reason": f"예정·계획 중심 문장 비율이 {future_ratio:.0%}로 50%를 초과함",
            }
        )

    if _needs_method_evidence(criterion) and not _has_concrete_method(criterion_text, kind):
        signals.append(
            {
                "code": "S4",
                "cap_ratio": Decimal("0.40"),
                "reason": "구체적인 모델·알고리즘·절차·기술 방법이 명시되지 않음",
            }
        )

    if _required_keywords_missing(criterion, rubric, normalized):
        signals.append(
            {
                "code": "S5",
                "cap_ratio": Decimal("0.50"),
                "reason": "공고문에서 지정한 필수 요소가 제출문서에 없음",
            }
        )

    if not signals:
        return None

    # 두 가지 이상의 근거 결핍이 겹치면 25% 상한을 적용한다. 이는 저품질 앵커의
    # data 5/20, AI 6/25, feasibility 5/20을 재현하면서 좋은 문서는 건드리지 않는다.
    evidence_gap_codes = {signal["code"] for signal in signals}
    severe_combination = (
        bool(evidence_gap_codes & {"S1", "S4"})
        and len(evidence_gap_codes & {"S1", "S2", "S3", "S4"}) >= 2
    )
    effect_without_measurement = kind == "effect" and {"S2", "S3"} <= evidence_gap_codes
    if severe_combination or effect_without_measurement:
        signals.append(
            {
                "code": "MULTI",
                "cap_ratio": Decimal("0.25"),
                "reason": "구체적 출처·수치·방법·실행 근거 중 둘 이상이 동시에 부족함",
            }
        )

    cap_ratio = min(signal["cap_ratio"] for signal in signals)
    max_score = Decimal(str(criterion["max_score"]))
    cap_score = (max_score * cap_ratio).quantize(_QUANT, rounding=ROUND_HALF_UP)
    return {
        "cap_score": cap_score,
        "cap_ratio": cap_ratio,
        "signals": signals,
    }
