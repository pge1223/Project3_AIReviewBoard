# 작성자: 용준/Claude(2026-07-30, 요청: "최신 직접 사용자 지시와 대화 문맥 분리" — 출처
# 기반 설계)
# 목적: 이번 턴 HTTP 요청에서 직접 들어온 사용자 원문(state["current_user_input"])만을
#       입력으로 받아, 결정론적 정규식/키워드 규칙으로 "사용자가 이번 턴에 무엇을 다루라고/
#       다루지 말라고 했는가"를 구조화한다. classify_query_type()(같은 파일)과 동일한
#       설계 원칙을 따른다 — LLM을 쓰지 않고(라우팅 판단에는 결정론적 신호만 쓴다는 기존
#       원칙 유지), 확신 있게 매칭되는 신호가 없으면 전부 빈 값/False로 돌려줘 기존 동작을
#       그대로 보존한다("오탐보다 누락이 안전"하다는 classify_query_type의 폴백 철학과 동일).
#
#       보안적으로 중요한 점은 규칙의 "내용"이 아니라 "입력의 출처"다 — 이 함수는 항상
#       state["current_user_input"](ideation_conv_run.py가 /start·/reply의 최신 사용자
#       메시지에서만, 단일 지점에서 채우는 필드)만 받는다. 업로드 문서·RAG 청크·과거 위원
#       발언·conversation_context 어디에도 이 함수를 호출하지 않는다 — 그래서 반환값의
#       instruction_provenance는 항상 "direct_user_message"로 고정해도 된다(호출부 계약이
#       그 사실을 보장하지, 텍스트 내용 분석으로 보장하는 게 아니다).
from __future__ import annotations

import re
from typing import Literal, TypedDict

InstructionProvenance = Literal["direct_user_message"]


class CurrentUserInstruction(TypedDict):
    topic_override: str | None
    excluded_topics: list[str]
    requested_outputs: list[str]
    scope_constraints: list[str]
    interrupt_active_issue: bool
    instruction_provenance: InstructionProvenance


def _empty_instruction() -> CurrentUserInstruction:
    return CurrentUserInstruction(
        topic_override=None,
        excluded_topics=[],
        requested_outputs=[],
        scope_constraints=[],
        interrupt_active_issue=False,
        instruction_provenance="direct_user_message",
    )


# "다음 논의 주제를 X로 잡지 말고"류 문장의 상투적 도입부는 미리 제거해 캡처 그룹이
# 앞부분(문장 서두)까지 통째로 삼키지 않게 한다 — 문두에 있을 때만 제거한다(임의 위치의
# "주제" 단어까지 지우면 다른 의미가 훼손될 수 있어 ^ 앵커로 제한).
_TOPIC_INTRO_PREFIX_RE = re.compile(
    r"^(?:다음|이번|앞으로는|이제부터는)?\s*(?:논의\s*)?주제(?:는|를|로)?\s*"
)

# "X로/을/를 잡지 말고", "X는 논의하지 말고", "X는 빼고", "X 말고"처럼 주제를 배제하는
# 표현만 좁게 잡는다 — 일반 부정문("좋지 않습니다")까지 걸리지 않도록 "잡지/논의하지/
# 다루지/얘기하지 + 말고"나 "빼고"/"제외하고" 조합만 인정한다. 캡처 그룹을 공백 기준
# 최대 4어절로 제한해 트리거 앞의 무관한 문장 전체(도입부 등)를 함께 삼키는 것을 막는다 — 완벽한
# 한국어 구문 분석이 아니라 결정론적 휴리스틱이라는 한계를 인정하고, 애매하면 조금
# 넓게 잡히는 쪽(과소 필터링)을 택한다(classify_query_type과 동일한 보수적 원칙).
_EXCLUDED_TOPIC_RE = re.compile(
    r"([가-힣A-Za-z0-9]+(?:\s[가-힣A-Za-z0-9]+){0,3}?)\s*(?:으로|로|을|를)?\s*"
    r"(?:잡지\s*말고|논의하지\s*말고|다루지\s*말고|얘기하지\s*말고|말고|빼고|제외하고)"
)

# _QUERY_TYPE_KEYWORDS["expert_analysis_query"](ideation_conv_nodes.py)에 이미 등록된
# "검토 범위를 구체적인 복합 명사구로 직접 지정하는 표현"과 동일한 문구를 재사용한다 —
# 새 어휘 체계를 따로 만들지 않는다(요청: 기존 신호와 이중 판단 방지).
_REQUESTED_OUTPUT_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("주요 기능", "main_features"),
    ("핵심 기능", "main_features"),
    ("필요한 데이터", "required_data"),
    ("데이터 확보", "required_data"),
    ("기술 구현 방향", "technical_approach"),
    ("기술 구현", "technical_approach"),
    ("구현 방향", "technical_approach"),
    ("MVP", "mvp_scope"),
    ("mvp", "mvp_scope"),
)

# 여러 글자짜리 조사("으로"/"에서")를 한 글자짜리보다 먼저 시도해야 한다 — 그렇지 않으면
# "방식으로"에서 "로"만 떨어져 나가 "방식으"라는 어색한 조각이 남는다.
_TOPIC_STRIP_RE = re.compile(r"^(?:으로|에서|은|는|이|가|을|를|로)+|(?:으로|에서|은|는|이|가|을|를|로)+$")


def _normalize_topic(raw: str) -> str:
    text = raw.strip()
    text = _TOPIC_STRIP_RE.sub("", text).strip()
    return text


def parse_current_user_instruction(current_user_input: str | None) -> CurrentUserInstruction:
    """current_user_input(이번 턴 사용자 원문) 하나만 분석해 topic_override/excluded_topics/
    requested_outputs/interrupt_active_issue를 결정한다. 아무 신호도 찾지 못하면 완전히
    빈 값을 반환하고, 이때 interrupt_active_issue=False라 호출부(resolve_effective_issue 등)의
    기존 분기는 전혀 영향받지 않는다."""
    text = (current_user_input or "").strip()
    if not text:
        return _empty_instruction()

    search_text = _TOPIC_INTRO_PREFIX_RE.sub("", text)

    excluded_topics: list[str] = []
    for match in _EXCLUDED_TOPIC_RE.finditer(search_text):
        topic = _normalize_topic(match.group(1))
        if topic and topic not in excluded_topics:
            excluded_topics.append(topic)

    requested_outputs: list[str] = []
    for keyword, code in _REQUESTED_OUTPUT_KEYWORDS:
        if keyword in text and code not in requested_outputs:
            requested_outputs.append(code)

    remainder = _EXCLUDED_TOPIC_RE.sub("", search_text).strip(" ,.")
    if requested_outputs:
        topic_override = remainder or text
    elif excluded_topics:
        topic_override = remainder or None
    else:
        topic_override = None

    interrupt_active_issue = bool(excluded_topics or topic_override)

    return CurrentUserInstruction(
        topic_override=topic_override,
        excluded_topics=excluded_topics,
        requested_outputs=requested_outputs,
        scope_constraints=[],
        interrupt_active_issue=interrupt_active_issue,
        instruction_provenance="direct_user_message",
    )


__all__ = ["CurrentUserInstruction", "InstructionProvenance", "parse_current_user_instruction"]
