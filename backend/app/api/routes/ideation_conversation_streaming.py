# 작성자: 용준/Claude(2026-07-21)
# 목적: 대화형 아이디어 발전 회의(ideation-conversation)의 실시간 스트리밍 llm_call을
#       만든다. ai/meeting/graph 쪽 코드(그래프/노드)는 이 파일의 존재 자체를 모른다 —
#       기존 LLMCall 시그니처(Callable[[str], str])를 그대로 만족하는 함수 하나를
#       반환할 뿐이고, 그 함수 "내부에서만" OpenAI 스트리밍 응답을 받아 사용자에게 보여줄
#       메시지 텍스트만 실시간으로 sink에 흘려보낸다.
#
#       프롬프트 종류 판별은 ai/meeting/tests의 기존 스크립트 스텁이 이미 쓰고 있는
#       마커 문자열([질문 규칙]/[의견 규칙]/[판정 규칙]/[제안 규칙]/[해석 규칙]/
#       [후보 생성 규칙]/[검토 규칙]/"idea_name")을 그대로 재사용한다 — 이 코드베이스가
#       이미 "프롬프트 안의 고정 마커로 어떤 노드가 호출했는지 판별"하는 것을 테스트
#       전반에서 공식적인 관례로 쓰고 있으므로, 새 규약을 만들지 않고 그 위에 얹는다.
#
#       사용자에게 실제로 보이는 메시지를 만드는 호출([질문 규칙]/[의견 규칙]/
#       [제안 규칙])만 OpenAI 델타를 message_delta로 흘려보낸다 — 그 외(분류/후보 생성/
#       실현가능성 검토/후보 선택 해석/최종 종합)는 phase 진행 상태 문구만 보내고, 완성된
#       응답은 기존과 동일하게 한 번에 받는다(사용자에게 내부 분류 결과나 JSON 필드명을
#       보여주지 않기 위함).
# import: 표준 라이브러리 hashlib/logging/uuid, 같은 패키지 상위의 graph/prompts(둘 다
#         ai/meeting 하위 패키지, main.py가 이미 sys.path에 올려둔다).

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from typing import Callable, Iterable

from graph import (
    DELEGATION_FACILITATOR_STREAM_FIELDS,
    DELEGATION_REVIEW_STREAM_FIELDS,
    DISCUSSION_STREAM_FIELDS,
    EXPERT_DELEGATION_STREAM_FIELDS,
    EXPERT_DELEGATION_TRAILER,
    FACILITATOR_SUMMARY_STREAM_FIELDS,
    IdeationCancelled,
    JSONFieldStreamer,
    QUESTION_STREAM_FIELDS,
    sanitize_preview,
    stream_delta_trace_enabled,
    trace_event,
)
from prompts import get_persona_card

logger = logging.getLogger(__name__)

# sink는 FastAPI/OpenAI를 전혀 몰라도 되는 순수 콜백이다 — 호출부(reply_stream 엔드포인트)가
# 이벤트를 큐에 넣어 NDJSON으로 흘려보낸다.
StreamSink = Callable[[dict], None]
# 프롬프트 하나에 대해 "원문 텍스트 조각"을 순서대로 만들어내는 이터러블 — 실제 구현은
# OpenAI streaming 응답의 delta.content를 그대로 넘긴다. 테스트에서는 이 부분만 가짜로
# 대체해 실제 OpenAI 호출 없이 검증한다.
ChatCompletionStreamer = Callable[[str], Iterable[str]]
# 스트리밍 대상이 아닌 호출(분류 등)에 쓰는 기존과 동일한 블로킹 호출.
ChatCompletionCaller = Callable[[str], str]

_QUESTION_MARKER = "[질문 규칙]"
_DISCUSSION_MARKER = "[의견 규칙]"
_DELEGATION_MARKER = "[제안 규칙]"
# 용준/Claude(2026-07-21, 요청: 위원 간 실제 회의로 개편) — 진행자의 라운드 정리 메시지도
# 사용자에게 실제로 보이는 텍스트이므로 스트리밍 대상이다. 기획/개발과 달리 persona_id가
# 항상 "ideation_facilitator" 고정이라 _persona_from_prompt의 역할 마커 판별을 타지 않는다.
_FACILITATOR_SUMMARY_MARKER = "[진행자 정리 규칙]"
# 용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장) — 담당 위원의
# 위임 제안(_DELEGATION_MARKER, 최초/수정 공용)에 이어 반대 위원의 검토, 진행자의 최종
# 권고안도 사용자에게 실제로 보이는 텍스트이므로 각각 스트리밍 대상이다.
_DELEGATION_REVIEW_MARKER = "[위임 검토 규칙]"
_DELEGATION_FACILITATOR_MARKER = "[위임 정리 규칙]"
_PLANNING_ROLE_MARKER = "당신은 AI Review Board의 기획 전문가입니다"
_DEV_ROLE_MARKER = "당신은 AI Review Board의 개발 전문가입니다"

# 사용자에게 보이는 메시지를 만들지 않는 호출들 — 델타를 흘리지 않고 진행 상태 문구만
# 보낸다. 순서대로 검사하며 먼저 매칭되는 것을 쓴다.
_PHASE_ONLY_LABELS: tuple[tuple[str, str], ...] = (
    ("[판정 규칙]", "답변의 의도를 확인하고 있습니다"),
    ("[해석 규칙]", "선택하신 내용을 해석하고 있습니다"),
    ("[후보 생성 규칙]", "아이디어 후보를 만들고 있습니다"),
    ("[검토 규칙]", "후보의 실현 가능성을 검토하고 있습니다"),
    ("[캔버스 갱신 규칙]", "기획 캔버스를 정리하고 있습니다"),
    ('"idea_name"', "최종 결과를 정리하고 있습니다"),
)


def _persona_from_prompt(prompt: str) -> str | None:
    if _PLANNING_ROLE_MARKER in prompt:
        return "planning_expert"
    if _DEV_ROLE_MARKER in prompt:
        return "dev_expert"
    return None


def _stream_plan_for(prompt: str) -> tuple[str, tuple, Callable[[str], str | None]] | None:
    """이 프롬프트가 사용자에게 보이는 메시지를 만드는 호출이면
    (node_kind, field_plan, header_resolver)를 반환하고, 그 외(분류/후보 생성/종합 등)는
    None을 반환한다.

    용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환): 스트리밍
    대상 필드는 이제 어느 node_kind든 "spoken_text" 하나뿐이라(예전에는 기획/개발 헤더가
    달라 discussion_headers_for로 동적 해석이 필요했고, 위임 제안의 "proposal"도 페르소나
    표시 이름을 붙였다) header_resolver는 모두 항상 None을 반환한다 — 화면에는 순수
    spoken_text 텍스트만 흘러간다."""
    if _QUESTION_MARKER in prompt:
        return "question", QUESTION_STREAM_FIELDS, lambda _field: None
    if _DISCUSSION_MARKER in prompt:
        return "discussion", DISCUSSION_STREAM_FIELDS, lambda _field: None
    if _FACILITATOR_SUMMARY_MARKER in prompt:
        return "facilitator_summary", FACILITATOR_SUMMARY_STREAM_FIELDS, lambda _field: None
    if _DELEGATION_REVIEW_MARKER in prompt:
        return "delegation_review", DELEGATION_REVIEW_STREAM_FIELDS, lambda _field: None
    if _DELEGATION_FACILITATOR_MARKER in prompt:
        return "delegation_facilitator", DELEGATION_FACILITATOR_STREAM_FIELDS, lambda _field: None
    if _DELEGATION_MARKER in prompt:
        return "expert_delegation", EXPERT_DELEGATION_STREAM_FIELDS, lambda _field: None
    return None


def _phase_label_for(prompt: str) -> str | None:
    for marker, label in _PHASE_ONLY_LABELS:
        if marker in prompt:
            return label
    return None


class _CompositionAssembler:
    """JSONFieldStreamer가 만든 (field_name, delta) 이벤트를, canonical 메시지 조립 함수
    (ideation_conv_nodes.py::_compose_question_content 등)가 최종적으로 만드는 문자열과
    글자 단위로 동일해지도록 헤더를 붙여 message_delta 텍스트로 바꾼다. 필드가 여러 개면
    "먼저 시작된 필드 뒤에는 \\n\\n + 다음 헤더"를 붙이고, 맨 처음 시작되는 필드는 헤더만
    바로 붙인다 — compose 함수들이 항상 이 규칙으로 섹션을 이어붙이기 때문이다."""

    def __init__(self, field_plan: tuple[tuple[str, str | None], ...], header_resolver: Callable[[str], str | None]):
        self._static_headers = dict(field_plan)
        self._header_resolver = header_resolver
        self._started: set[str] = set()

    def to_delta(self, field_name: str, text: str) -> str:
        prefix = ""
        if field_name not in self._started:
            header = self._static_headers.get(field_name) or self._header_resolver(field_name)
            prefix = ("\n\n" if self._started else "") + (f"{header}\n" if header else "")
            self._started.add(field_name)
        return prefix + text


def _prompt_key(prompt: str) -> str:
    """구조화 응답 검증 실패로 같은 prompt가 다시 llm_call에 들어오면(재시도 1회) 그것을
    "새 메시지"가 아니라 "이전 스트리밍 메시지를 지우고 다시 시작"으로 처리해야 한다 —
    반대로 같은 노드 마커([질문 규칙] 등)가 한 요청 안에서 서로 다른 프롬프트로 두 번
    불리는 정상 케이스(예: 기획/개발 두 전문가가 순서대로 의견을 말하는 경우, 또는
    continue_round로 다음 라운드 질문까지 같은 요청 안에서 만들어지는 경우)는 "재시도"가
    아니라 "새 메시지"로 취급해야 한다. 이 둘을 구분하는 유일하게 정확한 기준은
    "정확히 같은 prompt 문자열이 다시 들어왔는가"이다(_safe_call_structured_json의
    재시도는 매번 동일한 prompt 인자로 llm_call을 다시 부른다) — 그래서 node_kind나
    persona_id가 아니라 prompt 원문의 해시를 재시도 감지 키로 쓴다."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def make_streaming_llm_call(
    session_id: str,
    sink: StreamSink,
    *,
    stream_chat_completion: ChatCompletionStreamer,
    call_chat_completion: ChatCompletionCaller,
    max_calls: int,
    cancel_event: threading.Event | None = None,
    request_id: str | None = None,
) -> Callable[[str], str]:
    """기존 LLMCall 시그니처(Callable[[str], str])를 그대로 만족하는 스트리밍 llm_call을
    만든다. 반환값(전체 텍스트)은 스트리밍이 아닌 기존 _build_llm_call()과 정확히 같은
    형태(OpenAI 응답 원문 그대로)이므로, ai/meeting/graph의 _safe_call_json/
    _safe_call_structured_json이 이 반환값을 그대로 json.loads + 스키마 검증한다 — 검증
    로직·재시도 정책은 단 한 줄도 바뀌지 않는다. 그래프가 이 llm_call을 두 번(재시도)
    부르면 이 함수가 그것을 감지해 message_reset을 먼저 보낸다.

    용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소): cancel_event가 주어지고 set()되면,
    다음 llm_call 진입 시점 또는 OpenAI 스트림의 다음 청크를 받는 시점 중 더 빠른 쪽에서
    IdeationCancelled를 던진다 — 이 예외는 _safe_call_structured_json/_safe_call_json이
    (좁은 예외만 잡으므로) 재시도하지 않고 그대로 그래프 실행까지 전파한다. 구조화 JSON이
    아직 불완전한 상태에서 취소되면 그 텍스트를 파싱하거나 반환하지 않는다(요청: "불완전한
    JSON을 파싱하거나 재시도하지 않는다") — message_end는 여전히 보내 프런트가 스트리밍
    말풍선을 정리할 수 있게 하되, canonical 텍스트로 이어지지 않는다."""
    call_count = 0
    reset_tokens: dict[str, str] = {}  # _prompt_key(prompt) -> 그 프롬프트로 시작했던 stream message_id
    # 용준/Claude(2026-07-23, 요청: 스트리밍 UX 버그 수정) — 방금 discard된(하지만 아직 그
    # 자리를 이어받을 새 message_start가 나오지 않은) stream message_id. 다음 message_start가
    # 이 값을 supersedes_message_id로 실어 보내면, 프런트는 "새 말풍선을 새로 추가"하는 대신
    # "검토 중이던 자리에 그대로 이어붙이는" 교체로 처리할 수 있다. discard_streamed_prompt와
    # llm_call은 항상 같은 그래프 노드 호출 흐름 안에서(중간에 다른 논리적 발언이 끼어들 수
    # 없게) 짝지어 호출되므로, 이 값을 소비하자마자 즉시 비워도 다른 위원의 발언을 잘못
    # 이어받을 위험이 없다.
    pending_supersede_id: str | None = None

    def _check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise IdeationCancelled(session_id, request_id)

    def llm_call(prompt: str) -> str:
        nonlocal call_count, pending_supersede_id
        _check_cancelled()
        call_count += 1
        if call_count > max_calls:
            raise RuntimeError(
                f"[{session_id}] 프리뷰 LLM 호출 상한({max_calls}회) 초과 — "
                "루프 또는 재시도 폭주 의심, 중단합니다."
            )

        plan = _stream_plan_for(prompt)
        if plan is None:
            label = _phase_label_for(prompt)
            if label:
                sink({"type": "phase", "label": label})
            return call_chat_completion(prompt)

        node_kind, field_plan, header_resolver = plan
        persona_id = _persona_from_prompt(prompt) or "ideation_facilitator"
        display_name = get_persona_card(persona_id).get("display_name", persona_id)

        key = _prompt_key(prompt)
        previous_stream_id = reset_tokens.get(key)
        if previous_stream_id:
            # 정확히 같은 prompt가 다시 왔다 — 이전 시도의 구조화 응답이 검증에 실패해
            # 재시도하는 중이다. 보통은 discard_streamed_prompt가 먼저 불려 reset_tokens에서
            # 이미 pop된 뒤라 이 분기를 타지 않지만(중복 reset 방지), 방어적으로 남겨둔다.
            sink(
                {
                    "type": "message_reset",
                    "message_id": previous_stream_id,
                    "reason": "retry_same_prompt",
                    "will_retry": True,
                    "request_id": request_id,
                }
            )
            pending_supersede_id = previous_stream_id

        stream_id = f"STREAM-{uuid.uuid4().hex[:10]}"
        reset_tokens[key] = stream_id
        sink({"type": "phase", "label": f"{display_name}이(가) 응답을 작성하고 있습니다"})
        message_start_event = {
            "type": "message_start",
            "message_id": stream_id,
            "speaker_id": persona_id,
            "speaker_name": display_name,
            "request_id": request_id,
        }
        if pending_supersede_id:
            # 직전에 discard된(검토 중 상태로 남아있는) 말풍선의 자리를 이 새 스트림이
            # 이어받는다는 것을 프런트에 명시적으로 알린다 — speaker_id 일치만으로 추정하지
            # 않기 위함이다.
            message_start_event["supersedes_message_id"] = pending_supersede_id
            pending_supersede_id = None
        sink(message_start_event)
        trace_event(
            "IDEATION_STREAM_MESSAGE_STARTED",
            speaker=persona_id,
            message_id=stream_id,
            node_kind=node_kind,
        )

        streamer = JSONFieldStreamer([name for name, _ in field_plan])
        assembler = _CompositionAssembler(field_plan, header_resolver)
        full_raw_parts: list[str] = []
        cancelled = False
        started_at = time.perf_counter()
        delta_count = 0
        streamed_chars = 0
        stream_iterator = iter(stream_chat_completion(prompt))
        try:
            for chunk in stream_iterator:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    close_stream = getattr(stream_iterator, "close", None)
                    trace_event(
                        "IDEATION_STREAM_CLOSE_ATTEMPTED",
                        speaker=persona_id,
                        message_id=stream_id,
                        close_supported=callable(close_stream),
                    )
                    if callable(close_stream):
                        close_stream()
                    break
                if not chunk:
                    continue
                full_raw_parts.append(chunk)
                for field_name, delta in streamer.feed(chunk):
                    if delta is None:
                        if node_kind == "expert_delegation" and field_name == "spoken_text":
                            # make_expert_delegation_message가 항상 고정으로 덧붙이는
                            # 문구 — LLM 출력이 아니라 우리 쪽 리터럴이므로, 이 필드가
                            # 닫힌 직후 곧바로 흘려보낸다(canonical 메시지와 100% 동일).
                            sink({"type": "message_delta", "message_id": stream_id, "delta": EXPERT_DELEGATION_TRAILER})
                        continue
                    text = assembler.to_delta(field_name, delta)
                    if text:
                        delta_count += 1
                        streamed_chars += len(text)
                        if stream_delta_trace_enabled():
                            trace_event(
                                "IDEATION_STREAM_DELTA",
                                level=logging.DEBUG,
                                speaker=persona_id,
                                message_id=stream_id,
                                seq=delta_count,
                                chars=len(text),
                                delta=sanitize_preview(text),
                            )
                        sink({"type": "message_delta", "message_id": stream_id, "delta": text})
        finally:
            sink({"type": "message_end", "message_id": stream_id})
            trace_event(
                "IDEATION_STREAM_MESSAGE_ENDED",
                speaker=persona_id,
                message_id=stream_id,
                cancelled=cancelled,
                delta_count=delta_count,
                char_count=streamed_chars,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 1),
            )

        if cancelled:
            raise IdeationCancelled(session_id, request_id)
        return "".join(full_raw_parts)

    def discard_streamed_prompt(prompt: str, reason: str, will_retry: bool = True) -> None:
        """검증에 실패한(또는 grounding 재시도로 대체될) 임시 초안을 프런트에서 제거하지
        않고 "검토 중" 상태로 전환시킨다.

        will_retry=True면 바로 이어서 같은 논리적 발언의 재시도 스트림이 시작될 예정이라는
        뜻이고(프런트는 검토 중 표시를 유지하다가 다음 message_start로 교체), False면 더 이상
        재시도가 없고(구조화 검증 2회 실패) 서버가 만든 안전한 fallback 발언이 담긴 canonical
        state가 도착할 때까지 검토 중 표시를 유지해야 한다는 뜻이다. 어느 쪽이든 canonical
        state에는 원래 이 임시 초안이 저장되지 않는다 — reset은 화면 표시만 바꾼다.
        """
        nonlocal pending_supersede_id
        # pop해서 다음 재시도 시작 시 같은 메시지에 reset을 중복 전송하지 않는다.
        stream_id = reset_tokens.pop(_prompt_key(prompt), None)
        if not stream_id:
            return
        sink(
            {
                "type": "message_reset",
                "message_id": stream_id,
                "reason": reason,
                "will_retry": will_retry,
                "request_id": request_id,
            }
        )
        if will_retry:
            pending_supersede_id = stream_id
        trace_event(
            "IDEATION_STREAM_MESSAGE_DISCARDED",
            message_id=stream_id,
            reason=reason,
            will_retry=will_retry,
        )

    setattr(llm_call, "discard_streamed_prompt", discard_streamed_prompt)
    return llm_call
