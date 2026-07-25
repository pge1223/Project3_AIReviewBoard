# 작성자: 경이
# 목적: LangGraph reviewer 노드(M4, MTG-001 / RAG-003·004·005 연동). 위원 1명을 독립적으로
#       실행해 rubric·검색 근거로 검토하고, raw 출력을 v2 reviewerResult로 변환해 State에
#       반영한다. State에 evidence_context(용준 RAG 어댑터)가 있으면 ①(persona,criterion)별
#       사전 prompt_guard를 프롬프트에 넣고, ②의견 생성 후 criterion마다 evidence_callback을
#       불러 RAG-004 근거 연결 + RAG-005 최종 판정을 받아, ③A안(근거는 RAG-004로 교체) +
#       최종 sufficiency 게이팅을 적용한다. evidence_context가 없으면 기존 flat
#       retrieved_evidence 경로로 그대로 동작한다(하위호환). 콜백은 backend가 주입하는
#       Callable이라 이 노드는 ai/rag를 직접 import하지 않는다(회의 ↔ RAG 분리 유지).
# import: prompts.build_reviewer_prompt(형제 패키지), 같은 패키지의 evidence/llm/state/transform.

from __future__ import annotations

import logging
from typing import Callable

from prompts import build_reviewer_prompt

from ..evidence import EvidencePool
from ..llm import LLMCall, parse_json_response
from ..state import MeetingState
from ..transform import raw_reviewer_to_v2

logger = logging.getLogger(__name__)

# backend가 주입하는 근거 콜백:
# (persona_id, criterion_id, review_item) -> {"linked_evidence_refs": [...],
#                                             "sufficiency": {"allow_numeric_score": bool, ...}}
EvidenceCallback = Callable[[str, str, dict], dict]


def _dedupe_by_chunk(items: list[dict]) -> list[dict]:
    """chunk_id 기준 중복 제거(먼저 등장한 것 유지). 여러 criterion의 retrieved_evidence를
    한 위원 프롬프트에 합칠 때 같은 청크가 중복 노출되지 않게 한다."""
    seen: set = set()
    out: list[dict] = []
    for it in items:
        cid = it.get("chunk_id")
        if cid in seen:
            continue
        if cid is not None:
            seen.add(cid)
        out.append(it)
    return out


# 재인/Claude(2026-07-21, 사용자 확인 하에 진행 — 경이님 확인 필요): 위원이 맡은
# criterion_id를 review_items에서 통째로 빼먹는 경우(GPT가 시킨 걸 안 지킨 것 - 근거
# 부족 판단도, RAG 게이트 차단도 아니고 그냥 언급 자체가 없는 경우)가 실측으로 확인됐다
# (transform.py의 unscored_criteria는 이 경우를 포함하지 않는다 - 위원이 시도조차 안 한
# 항목이라 raw.review_items에 애초에 없음). 프롬프트 문구만 강화해서는 확률만 낮출 뿐
# 보장이 안 되므로, 빠진 항목이 있으면 그 자리에서 바로 그 항목만 다시 짧게 물어봐서
# 채운다 - 재질문 결과도 raw_reviewer_to_v2()의 같은 판정(insufficient_evidence/RAG
# 게이트)을 그대로 거치므로, 최종적으로는 "점수가 매겨지거나 명확히 설명되는" 상태로
# 수렴한다("아무 설명 없이 사라짐"이 없어진다).
#
# "빠짐"의 기준은 rubric 전체가 아니라 expected_criterion_ids(이 위원이 실제로 맡은
# 범위)다 - rubric에는 다른 위원 담당 기준도 다 들어있어서(공통 입력), 전체 기준으로
# 비교하면 "원래 내 담당이 아니라서 안 쓴" 정상적인 경우까지 죄다 재질문하게 되고,
# 그러면 다른 전문가가 자기 분야도 아닌 기준에 억지 답을 내게 된다(설계상 원치 않는
# 동작). RAG-003/004 연동 경로(evidence_context)에서는 (persona_id, criterion_id) 배정이
# 이미 확정되어 my_ctx로 들어오므로 그 목록을 그대로 쓰고, 그 배정이 없는 레거시 경로
# (evidence_context 없음)는 "이 위원의 담당 범위"를 알 방법이 없어 재질문을 시도하지
# 않는다(잘못 강제하는 것보다 안 하는 게 안전).
def _fill_missing_criteria(
    persona_id: str,
    raw: dict,
    expected_criterion_ids: list[dict],
    submission: dict,
    retrieved: list[dict],
    guards: list[tuple] | None,
    llm_call: LLMCall,
) -> dict:
    if not expected_criterion_ids:
        return raw

    covered = {item.get("criterion_id") for item in raw.get("review_items", [])}
    missing = [c for c in expected_criterion_ids if c.get("criterion_id") not in covered]
    if not missing:
        return raw

    missing_ids = {c["criterion_id"] for c in missing}
    logger.warning(
        "[REVIEWER_MISSING_CRITERIA] persona_id=%s criteria=%s - 재질문 시도",
        persona_id, sorted(missing_ids),
    )
    missing_rubric = {
        "criteria": missing,
        "total_max_score": sum(c.get("max_score", 0) for c in missing),
    }
    retry_prompt = build_reviewer_prompt(
        persona_id, missing_rubric, submission, retrieved, evidence_guards=guards,
    )
    try:
        retry_raw = parse_json_response(llm_call(retry_prompt))
        retry_items = retry_raw.get("review_items", [])
    except Exception:
        logger.warning(
            "[REVIEWER_MISSING_CRITERIA_RETRY_FAILED] persona_id=%s criteria=%s - 재질문도 실패, 원본 그대로 반환",
            persona_id, sorted(missing_ids), exc_info=True,
        )
        return raw

    still_missing = missing_ids - {item.get("criterion_id") for item in retry_items}
    if still_missing:
        logger.warning(
            "[REVIEWER_MISSING_CRITERIA_AFTER_RETRY] persona_id=%s criteria=%s - 재질문 후에도 빠짐",
            persona_id, sorted(still_missing),
        )

    return {**raw, "review_items": [*raw.get("review_items", []), *retry_items]}


def make_reviewer_node(
    persona_id: str,
    llm_call: LLMCall,
    evidence_callback: EvidenceCallback | None = None,
) -> Callable[[MeetingState], dict]:
    """persona_id 위원 전용 노드 함수를 만든다. 1회차 독립 평가(MTG-001)만 다룬다."""

    def reviewer_node(state: MeetingState) -> dict:
        my_ctx = [
            e for e in (state.get("evidence_context") or []) if e.get("persona_id") == persona_id
        ]

        if my_ctx:
            # RAG 연동 경로: (persona,criterion)별 근거를 합치고 사전 guard를 모은다.
            retrieved = _dedupe_by_chunk(
                [ev for e in my_ctx for ev in (e.get("retrieved_evidence") or [])]
            )
            guards = [
                (e["criterion_id"], (e.get("sufficiency") or {}).get("prompt_guard", ""))
                for e in my_ctx
            ]
            # 재인/Claude(2026-07-21): my_ctx의 criterion_id가 곧 "이 위원이 실제로 맡은
            # 범위"다(RAG-003/004가 이미 persona-criterion을 배정해서 evidence_context에
            # 실어 보냄) - _fill_missing_criteria가 재질문 대상을 정할 때, rubric 전체가
            # 아니라 이 범위만 봐야 다른 위원 담당 기준까지 억지로 재질문하지 않는다.
            my_criterion_ids = {e["criterion_id"] for e in my_ctx}
            expected_criteria = [
                c for c in state["rubric"].get("criteria", []) if c.get("criterion_id") in my_criterion_ids
            ]
        else:
            # 레거시 경로: flat retrieved_evidence만 사용. persona-criterion 배정 정보가
            # 없어(evidence_context 자체가 없음) "이 위원 담당 범위"를 알 방법이 없으므로
            # 재질문 대상도 비워둔다(_fill_missing_criteria가 빈 목록이면 그냥 넘어감).
            retrieved = state["retrieved_evidence"]
            guards = None
            expected_criteria = []
            my_criterion_ids = None  # transform.py에 "범위를 모른다"로 전달 - 기존 동작 유지

        prompt = build_reviewer_prompt(
            persona_id,
            state["rubric"],
            state["submission"],
            retrieved,
            evidence_guards=guards,
        )
        raw = parse_json_response(llm_call(prompt))
        raw = _fill_missing_criteria(persona_id, raw, expected_criteria, state["submission"], retrieved, guards, llm_call)

        pool = EvidencePool(persona_id, retrieved)

        criterion_evidence: dict[str, dict] | None = None
        if evidence_callback is not None and my_ctx:
            # 의견 생성 후, criterion마다 RAG-004 링크 + RAG-005 최종 판정을 받는다.
            criterion_evidence = {}
            for item in raw.get("review_items", []):
                cid = item.get("criterion_id")
                if cid is None:
                    continue
                criterion_evidence[cid] = evidence_callback(persona_id, cid, item)

        v2_result = raw_reviewer_to_v2(
            raw, pool, criterion_evidence=criterion_evidence, expected_criterion_ids=my_criterion_ids,
            submission_text=(state.get("submission") or {}).get("text"),
        )
        return {
            "reviewer_results": {persona_id: v2_result},
            "evidence": pool.as_list(),
        }

    return reviewer_node
