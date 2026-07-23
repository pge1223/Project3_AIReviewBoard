"""
Ideation Evidence Planner (Phase 1 — Shadow Deterministic Evidence Planner)
================================================================================
용준/Claude(2026-07-23). 아이디어 회의(ideation-conversation) 전문가 발언을 생성하기
"전에" 규칙 기반으로 이번 턴에 쓸 evidence를 확정하는 planner.

실제 병목(로그 확인): retrieval 자체는 성공해도(target_count>0) 답변 생성 모델이 검색된
근거를 인용하지 않고 expert_judgment만 만드는 턴이 대부분이다(6턴 중 5턴이
evidence_status="expert_judgment_only"). Phase 1은 이 문제를 고치지 않는다 — 대신
"결정적 규칙만으로 적절한 evidence를 실제로 골라낼 수 있는가"를 shadow(그림자) 모드로
검증한다: 이 모듈이 만든 계획(EvidencePlan)은 prompt/claims/grounding/routing 어디에도
쓰이지 않고, 호출부(ai/meeting/graph/ideation_conv_nodes.py::make_conv_discussion_node)가
trace 로그로만 기록한다.

ai/meeting은 이 모듈을 import하지 않는다 — ai/meeting/graph는 backend가 주입하는
Callable(plain dict 입출력)의 "모양"만 안다(evidence_lookup/ground_claims와 동일한 경계
원칙, ai/rag/tests/test_meeting_evidence_service.py::TestScopeBoundary가 강제).

관련성 판정에 대해: ai.rag.evidence_linking.relevance.is_relevant_candidate()는 사후
claim-grounding용(생성된 claim 문장과 인용된 청크의 관련성)이라 여기서는 하드 게이트로
쓰지 않는다 — 의미 있는 keyword가 없으면 자동 통과하고, role keyword 하나만 겹쳐도 통과하는
등 "이 턴에 쓸 evidence를 미리 고르는" 목적에는 느슨하다. 그 결과는 legacy_relevance_pass로
진단용으로만 남기고, 실제 채택 여부는 별도 issue_relevance_score(질의문 vs 청크 키워드
겹침, calculate_relevance_score 재사용) 임계값으로 판단한다.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from ai.rag.evidence_linking.config import EvidenceLinkingConfig
from ai.rag.evidence_linking.relevance import calculate_relevance_score, extract_keywords, is_relevant_candidate

POLICY_VERSION = "ideation-planner-v9"

# 이 값 미만이면 "이번 쟁점의 실제 질의문과 무관하다"고 보고 제외한다 — calculate_relevance_score는
# 0~1 근사치이고, claim_grounding의 EvidenceLinkingConfig.min_relevance_score(0.1, 사후 검증용
# 느슨한 값)와는 별개로 planner 전용 임계값을 둔다(요청: 기존 threshold를 임의로 낮추지 않고
# 새 임계값은 상수로 명시).
MIN_ISSUE_RELEVANCE_SCORE: float = 0.15
# 공모전 criteria는 대체로 "AI 활용 여부"처럼 범용 문장이라 target보다 구체적인 주장에
# 오용되기 쉽다. 현재 쟁점과의 직접 겹침이 더 강할 때만 주입한다.
MIN_CRITERIA_ISSUE_RELEVANCE_SCORE: float = 0.25

# 역할별로 selected_evidence에 담을 최대 개수(공통 정책 — 요청 9번: "역할이 다르더라도 동일
# target을 보는 것은 정상"이므로 role별로 독립적으로 계산한다).
_ROLE_MAX_SELECTION: dict[str, int] = {"target": 1, "criteria": 1}

# planning_expert가 criteria를 채택하려면 issue 제목이 이 키워드 중 하나와 직접 관련돼야
# 한다(요청 9번: "criteria가 단순히 검색됐다는 이유로 선택하지 않는다"). dev_expert도 동일한
# 원칙의 별도 키워드 집합을 쓴다. 이 목록은 결정적 정책이며 하나의 임의 가중합으로 숨기지
# 않는다 — role_policy_pass 탈락 사유로 그대로 로그에 남는다.
_PLANNING_CRITERIA_ISSUE_KEYWORDS = (
    "문제",
    "대상 사용자",
    "사용자 가치",
    "고객 가치",
    "핵심 가치",
    "차별",
    "공모전",
    "심사",
    "적합성",
    "사업성",
    "계획",
    "목표",
    "KPI",
    "데이터",
    "통합",
    "AI 역할",
    "운영",
    "사회적 가치",
    "지속 가능",
    "거버넌스",
)
_DEV_CRITERIA_ISSUE_KEYWORDS = (
    "문제 정의",
    "실현 가능",
    "기술",
    "데이터",
    "안전",
    "적용성",
    "mvp",
    "MVP",
    "성능",
    "보안",
)

_CLAIM_TYPE_BY_ROLE: dict[str, str] = {"target": "user_provided_fact", "criteria": "document_fact"}

_SHADOW_HISTORY_KEEP = 20

_SENTENCE_END_RE = re.compile(r"[.?!]")
_CLAUSE_BOUNDARY_RE = re.compile(r"[,;；]")
_BULLET_PREFIX_RE = re.compile(r"^\s*[-*•·]\s+")
_TARGET_FIELD_LABEL_RE = re.compile(
    r"^\s*(?:[-*•·]\s+)?(?P<label>"
    r"제목|문제|대상\s*사용자|해결\s*방식|주요\s*기능|차별점|기대\s*효과|필요\s*데이터|"
    r"기술\s*접근\s*방식|MVP\s*범위|현재까지\s*확인된\s*제약사항|질문|사용자\s*답변"
    r")\s*[:：]\s*",
    re.IGNORECASE,
)

_ISSUE_QUOTE_FOCUS_MARKERS: dict[str, tuple[str, ...]] = {
    "problem": ("문제", "불편", "위험", "피해", "원인", "영향", "비효율", "오염", "어려움", "부족"),
    "target_user": ("사용자", "이용자", "고객", "시민", "주민", "대상", "상황"),
    "core_value": ("가치", "효과", "개선", "절감", "편의", "안전", "혜택"),
    "contest_fit": ("공모", "평가", "심사", "기준", "주제", "적합"),
    "differentiation": ("차별", "기존", "대비", "독창", "혁신", "경쟁"),
    "mvp": ("MVP", "최소", "우선", "범위", "초기", "핵심 기능"),
    "data": ("데이터", "수집", "확보", "품질", "센서", "연동"),
    "ai_role": ("AI", "모델", "알고리즘", "예측", "분석", "자동화"),
    "roadmap": ("확장", "단계", "로드맵", "향후", "도입", "고도화"),
}

_ISSUE_QUOTE_FORBIDDEN_MARKERS: dict[str, tuple[str, ...]] = {
    # 문제 정의 단계에서 평가표의 확장성·적용성 문장이나 구현 계획을 근거로 고르는 것을 차단한다.
    "problem": (
        "확장성",
        "확장 가능",
        "적용 가능",
        "혁신성",
        "차별성",
        "MVP",
        "구현",
        "데이터 확보",
        "데이터 수집",
        "운영 비용",
        "보안",
        "KPI",
        "정량적 성과",
    ),
    "target_user": ("MVP", "구현", "확장성", "데이터 수집", "보안"),
    "core_value": ("MVP", "구현", "데이터 수집", "보안"),
    "differentiation": ("MVP", "데이터 수집", "보안", "로드맵"),
}

_ISSUE_ID_ALIASES = {
    "problem_definition": "problem",
    "user": "target_user",
    "customer": "target_user",
    "value": "core_value",
    "competition_fit": "contest_fit",
    "feasibility": "mvp",
    "data_integration": "data",
}

_TARGET_FIELD_ISSUES: dict[str, frozenset[str]] = {
    "문제": frozenset({"problem"}),
    "대상사용자": frozenset({"target_user"}),
    "해결방식": frozenset({"ai_role"}),
    "주요기능": frozenset({"ai_role", "mvp"}),
    "차별점": frozenset({"differentiation"}),
    "기대효과": frozenset({"core_value"}),
    "필요데이터": frozenset({"data"}),
    "기술접근방식": frozenset({"ai_role", "mvp"}),
    "MVP범위": frozenset({"mvp"}),
    "현재까지확인된제약사항": frozenset({"mvp", "data", "roadmap"}),
}

_META_INSTRUCTION_MARKERS = (
    "검토해주세요",
    "검토해 주세요",
    "논의해주세요",
    "논의해 주세요",
    "판단해주세요",
    "판단해 주세요",
    "구체화하고",
    "구체화해",
    "유지하면서",
    "한 차례 더",
    "문서 근거와 구분",
)
_SCORE_ONLY_HEADING_RE = re.compile(
    r"^\s*(?:평가\s*기준\s*)?[가-힣A-Za-z][가-힣A-Za-z\s·/&-]{1,30}\s*"
    r"\(\s*\d+(?:\.\d+)?\s*(?:점)?\s*\)\s*$"
)


def _is_low_information_quote(quote: str) -> bool:
    """배점이 붙은 평가항목 제목만으로는 주장 근거가 될 수 없으므로 quote 후보에서 제외한다."""
    normalized = (quote or "").strip()
    return bool(_SCORE_ONLY_HEADING_RE.fullmatch(normalized))


def _is_meta_instruction_quote(item: dict, quote: str) -> bool:
    """사용자 답변 저장 청크 중 회의 진행 지시만 있는 문장을 사실 근거에서 제외한다."""
    if item.get("ideation_source_type") != "user_session_answer":
        return False
    return any(marker in quote for marker in _META_INSTRUCTION_MARKERS)


def _role_allows_criteria_for_issue(persona_id: str, issue_title: str) -> bool:
    """공통 정책(요청 9번) — criteria는 현재 issue와 역할별로 직접 관련될 때만 채택 후보가
    된다. 매핑에 없는 persona_id(진행자 등)는 이 planner 자체를 호출하지 않으므로 여기서는
    다루지 않는다."""
    if persona_id == "planning_expert":
        keywords = _PLANNING_CRITERIA_ISSUE_KEYWORDS
    elif persona_id == "dev_expert":
        keywords = _DEV_CRITERIA_ISSUE_KEYWORDS
    else:
        return False
    return any(keyword in issue_title for keyword in keywords)


def resolve_retrieval_score(item: dict) -> tuple[Optional[float], Optional[str]]:
    """final_score -> semantic_score -> score 우선순위로 검색 점수를 조회한다. 셋 다 없으면
    (None, "missing_retrieval_score")를 반환해 호출부가 그 항목을 제외하게 한다."""
    for key in ("final_score", "semantic_score", "score"):
        value = item.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), None
    return None, "missing_retrieval_score"


def evaluate_evidence_eligibility(
    item: dict,
    *,
    persona_id: str,
    effective_issue: dict,
    runtime_scope: dict,
    config: Optional[EvidenceLinkingConfig] = None,
) -> dict:
    """retrieved_evidence 항목 1건이 이번 턴 evidence 후보로 적격한지 결정적으로 판정한다.
    각 신호를 개별 필드로 노출해(요청: 임의 가중합 하나로 숨기지 않는다) 탈락 사유를
    exclusion_reasons에 남긴다."""
    cfg = config or EvidenceLinkingConfig()
    exclusion_reasons: list[str] = []

    ref = item.get("ref")
    chunk_id = item.get("chunk_id")
    document_id = item.get("document_id")
    text = item.get("text") or item.get("quote") or ""
    structural_valid = bool(
        isinstance(ref, str)
        and ref
        and isinstance(chunk_id, str)
        and chunk_id
        and isinstance(document_id, str)
        and document_id
        and isinstance(text, str)
        and text.strip()
    )
    if not structural_valid:
        exclusion_reasons.append("structurally_invalid")

    document_role = item.get("document_role")
    role_policy_pass = True
    if document_role not in ("target", "criteria"):
        role_policy_pass = False
        exclusion_reasons.append("unsupported_document_role")
    elif document_role == "criteria" and not _role_allows_criteria_for_issue(
        persona_id, effective_issue.get("title") or ""
    ):
        role_policy_pass = False
        exclusion_reasons.append("criteria_not_relevant_to_issue")

    scope_valid = True
    ideation_source_type = item.get("ideation_source_type")
    if ideation_source_type == "ideation_candidate":
        selected_candidate_document_id = runtime_scope.get("selected_candidate_document_id")
        if not selected_candidate_document_id or item.get("document_id") != selected_candidate_document_id:
            scope_valid = False
            exclusion_reasons.append("candidate_scope_mismatch")
    elif ideation_source_type == "user_session_answer":
        session_id = runtime_scope.get("session_id")
        if not session_id or item.get("session_id") != session_id:
            scope_valid = False
            exclusion_reasons.append("session_scope_mismatch")

    retrieval_score, score_reason = resolve_retrieval_score(item)
    if score_reason:
        exclusion_reasons.append(score_reason)
        retrieval_score_pass = False
    else:
        retrieval_score_pass = retrieval_score >= cfg.min_evidence_score
        if not retrieval_score_pass:
            exclusion_reasons.append("below_retrieval_score")

    issue_query = effective_issue.get("query") or effective_issue.get("title") or ""
    issue_relevance_score = calculate_relevance_score(
        issue_query,
        text,
        section_title=item.get("section"),
        document_title=item.get("document_name"),
    )
    legacy_relevance_pass = is_relevant_candidate(
        issue_query,
        text,
        section_title=item.get("section"),
        document_title=item.get("document_name"),
        config=cfg,
    )
    issue_relevance_threshold = (
        MIN_CRITERIA_ISSUE_RELEVANCE_SCORE
        if document_role == "criteria"
        else MIN_ISSUE_RELEVANCE_SCORE
    )
    # v3 평가표 청크는 "항목 + 질문 1개"라 짧아서 어휘 비율 기반 점수가 0.25보다 낮을 수
    # 있다. 400자 이하의 criteria 세부 문항이 쟁점 marker/금지 marker 검사를 직접 통과하면
    # 이를 별도 신호로 인정한다. 대형 범용 청크에는 적용하지 않아 기존 과대 매칭을 막는다.
    direct_issue_focus_pass = False
    if document_role == "criteria" and len(text) <= 400:
        # 하나의 criteria 청크에 제목과 여러 세부 문항이 함께 있을 수 있다. 청크 전체에
        # 다른 쟁점의 금지어가 하나 있다는 이유로 현재 쟁점과 정확히 맞는 세부 문항까지
        # 버리지 않고, 실제 quote 후보 중 하나가 직접 통과하는지를 본다.
        direct_issue_focus_pass = any(
            not _is_low_information_quote(text[start:end])
            and _quote_issue_focus(
                text[start:end],
                effective_issue,
                field_label=field_label,
            )[0]
            for start, end, field_label in _candidate_spans(text)
        )
    issue_relevance_pass = (
        issue_relevance_score >= issue_relevance_threshold or direct_issue_focus_pass
    )
    if not issue_relevance_pass:
        exclusion_reasons.append("below_issue_relevance")

    eligible = (
        structural_valid
        and scope_valid
        and retrieval_score_pass
        and role_policy_pass
        and issue_relevance_pass
    )

    return {
        "ref": ref,
        "structural_valid": structural_valid,
        "scope_valid": scope_valid,
        "retrieval_score_pass": retrieval_score_pass,
        "retrieval_score": retrieval_score,
        "issue_relevance_score": issue_relevance_score,
        "issue_relevance_threshold": issue_relevance_threshold,
        "issue_relevance_pass": issue_relevance_pass,
        "direct_issue_focus_pass": direct_issue_focus_pass,
        "legacy_relevance_pass": legacy_relevance_pass,
        "role_policy_pass": role_policy_pass,
        "eligible": eligible,
        "exclusion_reasons": exclusion_reasons,
    }


def _iter_line_spans(content: str) -> list[tuple[int, int]]:
    """content를 줄바꿈 단위로 나눠 (원문 기준 start, end) span을 낸다. 빈 줄은 건너뛴다."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in content.splitlines(keepends=True):
        stripped_len = len(line.rstrip("\r\n"))
        line_body = line[:stripped_len]
        if line_body.strip():
            leading_ws = len(line_body) - len(line_body.lstrip())
            spans.append((offset + leading_ws, offset + stripped_len))
        offset += len(line)
    return spans


def _iter_sentence_spans(content: str, line_start: int, line_end: int) -> list[tuple[int, int]]:
    """한 줄(line_start:line_end) 안에서 문장 종결 기호(.?!) 단위로 span을 더 쪼갠다. 글머리
    기호(bullet)는 문장 내용에서 제외한다(quote에 "- "가 그대로 남지 않도록)."""
    text = content[line_start:line_end]
    bullet_match = _BULLET_PREFIX_RE.match(text)
    cursor = bullet_match.end() if bullet_match else 0
    spans: list[tuple[int, int]] = []
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end()
        if end <= cursor:
            continue
        if text[cursor:end].strip():
            spans.append((line_start + cursor, line_start + end))
        cursor = end
    if text[cursor:].strip():
        spans.append((line_start + cursor, line_start + len(text)))
    return spans


def _iter_clause_spans(content: str, start: int, end: int) -> list[tuple[int, int]]:
    """긴 문장 안에서 쉼표/세미콜론으로 분리되는 짧은 원문 절을 quote 후보로 추가한다."""
    text = content[start:end]
    boundaries = [0, *(match.end() for match in _CLAUSE_BOUNDARY_RE.finditer(text)), len(text)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        segment = text[left:right]
        leading = len(segment) - len(segment.lstrip(" \t,;；"))
        trailing = len(segment.rstrip(" \t,;；"))
        clause_start = start + left + leading
        clause_end = start + left + trailing
        if clause_end > clause_start:
            spans.append((clause_start, clause_end))
    return spans


def _field_value_span(content: str, start: int, end: int) -> tuple[int, int, str | None] | None:
    """target 합성 문서의 ``문제:`` 같은 필드 라벨을 quote 내용에서 제거한다.

    라벨과 값이 같은 줄이면 값 부분의 원문 span만 반환하고, 라벨만 있는 줄이면 None을
    반환한다. 값이 다음 줄에 있으면 그 다음 줄은 _iter_line_spans()가 별도 후보로 처리한다.
    """
    text = content[start:end]
    match = _TARGET_FIELD_LABEL_RE.match(text)
    if not match:
        return start, end, None
    value = text[match.end():]
    leading = len(value) - len(value.lstrip())
    trailing = len(value.rstrip())
    value_start = start + match.end() + leading
    value_end = start + match.end() + trailing
    if value_end <= value_start:
        return None
    return value_start, value_end, match.group("label")


def _candidate_spans(content: str) -> list[tuple[int, int, str | None]]:
    """quote 후보 구간을 줄바꿈/글머리 단위로 먼저 나누고, 각 줄 안에서 문장 단위로 다시
    나눈다(요청: "마침표/물음표/느낌표뿐 아니라 줄바꿈·bullet 단위도 후보로 분리")."""
    spans: list[tuple[int, int, str | None]] = []
    current_field_label: str | None = None
    for line_start, line_end in _iter_line_spans(content):
        value_span = _field_value_span(content, line_start, line_end)
        if value_span is None:
            label_match = _TARGET_FIELD_LABEL_RE.match(content[line_start:line_end])
            if label_match:
                current_field_label = label_match.group("label")
            continue
        value_start, value_end, inline_label = value_span
        if inline_label:
            current_field_label = inline_label
        sentence_spans = _iter_sentence_spans(content, value_start, value_end)
        spans.extend((start, end, current_field_label) for start, end in sentence_spans)
        for sentence_start, sentence_end in sentence_spans:
            spans.extend(
                (start, end, current_field_label)
                for start, end in _iter_clause_spans(content, sentence_start, sentence_end)
            )
    return spans


def extract_planner_quote(
    content: str,
    query: str,
    *,
    issue: dict | None = None,
) -> Optional[tuple[str, int, int]]:
    """content에서 query와 가장 관련 있는 원문 구간을 그대로 잘라 반환한다(quote, start, end).
    quote_extractor.extract_quote()와 달리 관련 구간을 못 찾으면 청크 앞부분으로 폴백하지
    않는다 — 관련성 없는 fallback quote를 계획에 담지 않기 위해서다(요청 사항 그대로).
    반환값이 None이면 이 청크는 quote 추출 실패로 제외해야 한다."""
    if not content or not content.strip():
        return None
    query_tokens = extract_keywords(query)
    if not query_tokens:
        return None

    best_span: Optional[tuple[int, int]] = None
    best_score: tuple[int, int, float] = (0, 0, 0.0)
    for start, end, field_label in _candidate_spans(content):
        segment_text = content[start:end]
        if _is_low_information_quote(segment_text):
            continue
        if issue is not None and not _quote_issue_focus(
            segment_text,
            issue,
            field_label=field_label,
        )[0]:
            continue
        segment_tokens = extract_keywords(segment_text)
        # 한국어 조사/어미가 붙은 "문제의/설정이"도 query의 "문제/설정"과 같은 개념으로
        # 취급한다. 완전 부분 문자열 비교는 2자 이상 토큰에만 적용해 한 글자 과매칭을 피한다.
        overlap = sum(
            1
            for query_token in query_tokens
            if any(
                query_token == segment_token
                or (
                    min(len(query_token), len(segment_token)) >= 2
                    and (query_token in segment_token or segment_token in query_token)
                )
                for segment_token in segment_tokens
            )
        )
        label_tokens = extract_keywords(field_label or "")
        label_overlap = sum(
            1
            for query_token in query_tokens
            if any(
                query_token == label_token
                or (
                    min(len(query_token), len(label_token)) >= 2
                    and (query_token in label_token or label_token in query_token)
                )
                for label_token in label_tokens
            )
        )
        precision = overlap / len(segment_tokens) if segment_tokens else 0.0
        # target 합성 문서는 필드 구조가 명시적이므로, 현재 query와 맞는 필드의 실제 값을
        # 다른 섹션의 우연한 단어 일치보다 우선한다. quote 자체에는 라벨을 포함하지 않는다.
        score = (label_overlap, overlap, precision)
        if score > best_score:
            best_score = score
            best_span = (start, end)
    if best_span is None:
        return None

    raw_start, raw_end = best_span
    segment = content[raw_start:raw_end]
    lstripped = segment.lstrip()
    leading_ws = len(segment) - len(lstripped)
    quote = lstripped.rstrip()
    if not quote:
        return None
    start = raw_start + leading_ws
    end = start + len(quote)
    assert content[start:end] == quote  # exact substring invariant
    return quote, start, end


def _normalized_issue_id(issue: dict) -> str:
    issue_id = str(issue.get("issue_id") or "").strip()
    if issue_id.startswith("topic_"):
        issue_id = issue_id[len("topic_"):]
    return _ISSUE_ID_ALIASES.get(issue_id, issue_id)


def _planner_quote_query(issue: dict) -> str:
    """아이디어 전체·역할 설명이 아닌 현재 쟁점 어휘만으로 quote를 고른다."""
    issue_id = _normalized_issue_id(issue)
    markers = _ISSUE_QUOTE_FOCUS_MARKERS.get(issue_id, ())
    focused = " ".join((str(issue.get("title") or ""), *markers)).strip()
    return focused or str(issue.get("query") or "")


def _normalized_target_field_label(field_label: str | None) -> str:
    return re.sub(r"\s+", "", str(field_label or "")).upper()


def _field_label_matches_issue(field_label: str | None, issue: dict) -> bool:
    normalized_label = _normalized_target_field_label(field_label)
    return _normalized_issue_id(issue) in _TARGET_FIELD_ISSUES.get(normalized_label, frozenset())


def _quote_issue_focus(
    quote: str,
    issue: dict,
    *,
    field_label: str | None = None,
) -> tuple[bool, float, Optional[str]]:
    """선택 quote 자체가 현재 쟁점에 직접 맞는지 판정한다.

    whole chunk 관련성만 보면 큰 평가표 chunk 안의 다른 문장이 점수를 올리는 문제가 있으므로,
    최종 주입 문장에 대해 쟁점 marker와 명시적 이탈 marker를 다시 검사한다.
    """
    issue_id = _normalized_issue_id(issue)
    markers = _ISSUE_QUOTE_FOCUS_MARKERS.get(issue_id)
    field_focus_pass = _field_label_matches_issue(field_label, issue)
    if not markers:
        score = calculate_relevance_score(issue.get("title") or issue.get("query") or "", quote)
        return (score >= MIN_ISSUE_RELEVANCE_SCORE, score, None if score >= MIN_ISSUE_RELEVANCE_SCORE else "quote_below_issue_relevance")

    normalized = quote.lower()
    matched = {marker for marker in markers if marker.lower() in normalized}
    if not matched and not field_focus_pass:
        return False, 0.0, "quote_missing_issue_focus"
    forbidden = _ISSUE_QUOTE_FORBIDDEN_MARKERS.get(issue_id, ())
    if any(marker.lower() in normalized for marker in forbidden):
        return False, len(matched) / len(markers), "quote_conflicts_with_issue"
    score = max(len(matched) / len(markers), 1.0 if field_focus_pass else 0.0)
    return True, score, None


def _field_label_for_span(content: str, start: int, end: int) -> str | None:
    """선택된 원문 span이 속한 target 합성 필드 라벨을 되찾는다."""
    for candidate_start, candidate_end, field_label in _candidate_spans(content):
        if candidate_start <= start and end <= candidate_end:
            return field_label
    return None


def _selection_reason_code(persona_id: str, document_role: str, reused: bool) -> str:
    base = "target_fact_for_current_issue" if document_role == "target" else "criteria_fact_for_current_issue"
    return f"{base}_reused" if reused else base


def _empty_plan(plan_id: str, persona_id: str, issue: dict, reason: str) -> dict:
    return {
        "plan_id": plan_id,
        "policy_version": POLICY_VERSION,
        "persona_id": persona_id,
        "issue": issue,
        "eligible_evidence_count": 0,
        "grounded_claim_required": False,
        "expert_judgment_required": True,
        "selected_evidence": [],
        "empty_plan_reason": reason,
        "validation": {"valid": True, "errors": []},
    }


def validate_evidence_plan(
    plan: dict,
    *,
    retrieved_evidence: list[dict],
    runtime_scope: dict,
) -> dict:
    """이미 만들어진 plan을 retrieved_evidence/runtime_scope와 대조해 결정적으로 재검증한다
    (요청: "생성된 plan을 로그에 남기기 전에 결정적으로 검증"). ref 존재 여부부터 quote
    invariant까지 하드 룰만 검사하고, 실패한 plan은 valid=false로 표시한다 — Phase 1에서는
    valid 여부와 무관하게 prompt에는 절대 쓰이지 않고 로그로만 남는다."""
    errors: list[str] = []
    by_ref = {item.get("ref"): item for item in retrieved_evidence if isinstance(item, dict) and item.get("ref")}
    seen_refs: set[str] = set()
    seen_chunk_ids: set[str] = set()
    role_counts: dict[str, int] = {}

    for evidence in plan.get("selected_evidence") or []:
        ref = evidence.get("ref")
        source = by_ref.get(ref)
        if source is None:
            errors.append(f"unknown_ref:{ref}")
            continue

        if source.get("chunk_id") != evidence.get("chunk_id") or source.get("document_id") != evidence.get(
            "document_id"
        ):
            errors.append(f"chunk_document_mismatch:{ref}")

        ideation_source_type = source.get("ideation_source_type")
        if ideation_source_type == "ideation_candidate":
            selected_candidate_document_id = runtime_scope.get("selected_candidate_document_id")
            if not selected_candidate_document_id or source.get("document_id") != selected_candidate_document_id:
                errors.append(f"scope_violation:{ref}")
        elif ideation_source_type == "user_session_answer":
            session_id = runtime_scope.get("session_id")
            if not session_id or source.get("session_id") != session_id:
                errors.append(f"scope_violation:{ref}")

        role = evidence.get("document_role")
        if role not in _CLAIM_TYPE_BY_ROLE:
            errors.append(f"unsupported_document_role:{ref}")
        elif evidence.get("claim_type") != _CLAIM_TYPE_BY_ROLE[role]:
            errors.append(f"claim_type_mismatch:{ref}")

        if role is not None:
            role_counts[role] = role_counts.get(role, 0) + 1
            if role_counts[role] > _ROLE_MAX_SELECTION.get(role, 1):
                errors.append(f"role_limit_exceeded:{role}")

        if ref in seen_refs:
            errors.append(f"duplicate_ref:{ref}")
        seen_refs.add(ref)
        chunk_id = evidence.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            errors.append(f"duplicate_chunk_id:{chunk_id}")
        seen_chunk_ids.add(chunk_id)

        quote = evidence.get("quote")
        start = evidence.get("quote_start")
        end = evidence.get("quote_end")
        content = source.get("text") or source.get("quote") or ""
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"empty_quote:{ref}")
        elif (
            not isinstance(start, int)
            or not isinstance(end, int)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or content[start:end] != quote
        ):
            errors.append(f"quote_offset_invariant_failed:{ref}")
        elif plan.get("issue"):
            focus_pass, _, focus_reason = _quote_issue_focus(
                quote,
                plan["issue"],
                field_label=evidence.get("field_label"),
            )
            if not focus_pass:
                errors.append(f"{focus_reason or 'quote_issue_mismatch'}:{ref}")

        if not evidence.get("selection_reason_code"):
            errors.append(f"missing_selection_reason:{ref}")

    return {"valid": not errors, "errors": errors}


def build_evidence_plan(
    *,
    persona_id: str,
    effective_issue: dict,
    retrieved_evidence: list[dict],
    runtime_scope: dict,
    shadow_history: Optional[list[dict]] = None,
    config: Optional[EvidenceLinkingConfig] = None,
) -> dict:
    """이번 턴에 쓸 evidence를 규칙 기반으로 확정한다(EvidencePlan, 항상 plain dict).
    shadow_history는 같은 speaker/issue에서 이전에 선택된 chunk_id 목록(dict: chunk_id 키
    포함) — 반복 사용 여부만 표시할 뿐 후보에서 제외하지는 않는다(요청 8번: 유일한 적격
    근거라면 제거하지 말고 reused로 표시)."""
    cfg = config or EvidenceLinkingConfig()
    plan_id = f"EP-{uuid.uuid4().hex[:10]}"
    issue = {
        "issue_id": effective_issue.get("issue_id", ""),
        "title": effective_issue.get("title", ""),
        "query": effective_issue.get("query", ""),
    }

    if not retrieved_evidence:
        return _empty_plan(plan_id, persona_id, issue, "no_retrieved_evidence")

    evaluations = [
        (item, evaluate_evidence_eligibility(item, persona_id=persona_id, effective_issue=issue, runtime_scope=runtime_scope, config=cfg))
        for item in retrieved_evidence
        if isinstance(item, dict)
    ]
    if not evaluations:
        return _empty_plan(plan_id, persona_id, issue, "no_retrieved_evidence")

    if not any(e["structural_valid"] for _, e in evaluations):
        return _empty_plan(plan_id, persona_id, issue, "no_structurally_valid_evidence")

    if not any(e["structural_valid"] and e["scope_valid"] for _, e in evaluations):
        return _empty_plan(plan_id, persona_id, issue, "no_scope_valid_evidence")

    if not any(e["structural_valid"] and e["scope_valid"] and e["retrieval_score_pass"] for _, e in evaluations):
        return _empty_plan(plan_id, persona_id, issue, "below_retrieval_score")

    eligible = [(item, e) for item, e in evaluations if e["eligible"]]
    if not eligible:
        score_passing = [
            e for _, e in evaluations if e["structural_valid"] and e["scope_valid"] and e["retrieval_score_pass"]
        ]
        if any(not e["role_policy_pass"] for e in score_passing):
            reason = "role_policy_excluded_all"
        else:
            reason = "no_issue_relevant_evidence"
        return _empty_plan(plan_id, persona_id, issue, reason)

    history_chunk_ids = {h.get("chunk_id") for h in (shadow_history or []) if h.get("chunk_id")}

    quote_candidates: list[tuple[dict, dict, tuple[str, int, int], float]] = []
    for item, evaluation in eligible:
        content = item.get("text") or item.get("quote") or ""
        extraction = extract_planner_quote(
            content,
            _planner_quote_query(issue),
            issue=issue,
        )
        if extraction is None:
            continue
        quote, _, _ = extraction
        if _is_meta_instruction_quote(item, quote):
            continue
        quote_start, quote_end = extraction[1], extraction[2]
        field_label = _field_label_for_span(content, quote_start, quote_end)
        focus_pass, focus_score, _ = _quote_issue_focus(
            quote,
            issue,
            field_label=field_label,
        )
        if focus_pass:
            quote_candidates.append((item, evaluation, extraction, focus_score, field_label))

    if not quote_candidates:
        return _empty_plan(plan_id, persona_id, issue, "no_issue_focused_quote")

    def sort_key(pair: tuple[dict, dict, tuple[str, int, int], float, str | None]):
        item, evaluation, _, quote_focus_score, _field_label = pair
        reused = item.get("chunk_id") in history_chunk_ids
        return (
            0 if item.get("document_role") == "target" else 1,
            1 if reused else 0,
            -quote_focus_score,
            -evaluation["issue_relevance_score"],
            -(evaluation["retrieval_score"] or 0.0),
            item.get("chunk_id") or "",
        )

    ordered = sorted(quote_candidates, key=sort_key)

    role_counts: dict[str, int] = {}
    selected: list[dict] = []
    for item, evaluation, extraction, quote_focus_score, field_label in ordered:
        role = item.get("document_role")
        if role_counts.get(role, 0) >= _ROLE_MAX_SELECTION.get(role, 0):
            continue
        quote, quote_start, quote_end = extraction
        reused = item.get("chunk_id") in history_chunk_ids
        selected.append(
            {
                "ref": item.get("ref"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "document_role": role,
                "claim_type": _CLAIM_TYPE_BY_ROLE[role],
                "quote": quote,
                "quote_start": quote_start,
                "quote_end": quote_end,
                "retrieval_score": evaluation["retrieval_score"],
                "issue_relevance_score": evaluation["issue_relevance_score"],
                "quote_issue_relevance_score": quote_focus_score,
                "field_label": field_label,
                "selection_reason_code": _selection_reason_code(persona_id, role, reused),
                "reused_in_same_issue": reused,
            }
        )
        role_counts[role] = role_counts.get(role, 0) + 1

    if not selected:
        return _empty_plan(plan_id, persona_id, issue, "quote_extraction_failed")

    plan = {
        "plan_id": plan_id,
        "policy_version": POLICY_VERSION,
        "persona_id": persona_id,
        "issue": issue,
        "eligible_evidence_count": len(eligible),
        "grounded_claim_required": True,
        "expert_judgment_required": False,
        "selected_evidence": selected,
        "empty_plan_reason": None,
        "validation": {"valid": True, "errors": []},
    }
    validation = validate_evidence_plan(plan, retrieved_evidence=retrieved_evidence, runtime_scope=runtime_scope)
    plan["validation"] = validation
    if not validation["valid"]:
        plan["empty_plan_reason"] = "plan_validation_failed"
    return plan


__all__ = [
    "POLICY_VERSION",
    "MIN_ISSUE_RELEVANCE_SCORE",
    "MIN_CRITERIA_ISSUE_RELEVANCE_SCORE",
    "resolve_retrieval_score",
    "evaluate_evidence_eligibility",
    "extract_planner_quote",
    "validate_evidence_plan",
    "build_evidence_plan",
]
