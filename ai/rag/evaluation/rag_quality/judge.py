# 작성자: 용준/Claude(2026-07-22)
# 목적: Faithfulness/Persona Evidence Fit LLM-as-judge 호출. ai.meeting.graph.llm의
#       parse_json_response(순수 JSON 파싱, 코드블록/전후 설명문 제거)를 그대로
#       재사용한다 — 새로 만들지 않는다. 재시도 정책(_safe_call_json_retry)만 이 파일에서
#       새로 만든다 — ai.meeting 내부 _safe_call_json은 ai/meeting 전용 로그·정책과 얽혀
#       있어 ai/rag가 직접 import하기엔 계층 방향이 어색하다(요청: import 가능한 것은
#       재사용, 그 외는 최소한만 새로 작성).
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from ai.rag.evaluation.rag_quality.cache import JudgeCache
from ai.rag.evaluation.rag_quality.schemas import ClaimVerdict, ExpertJudgmentResult, PersonaFitResult

from ._meeting_path import ensure_meeting_on_path

ensure_meeting_on_path()
from graph.llm import parse_json_response  # noqa: E402

logger = logging.getLogger(__name__)

JudgeLLMCall = Callable[[str], str]

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
FAITHFULNESS_PROMPT_VERSION = "faithfulness_judge_v1"
PERSONA_FIT_PROMPT_VERSION = "persona_fit_judge_v1"
EXPERT_JUDGMENT_PROMPT_VERSION = "expert_judgment_judge_v1"

_VALID_VERDICTS = {"supported", "partially_supported", "unsupported", "contradicted", "non_factual"}


def _read_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _as_text(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def _render(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def _safe_call_json_retry(
    llm_call: JudgeLLMCall,
    prompt: str,
    validate: Callable[[dict], Optional[str]],
    *,
    max_attempts: int = 2,
) -> tuple[Optional[dict], bool, int]:
    """구조화 응답 검증 + 재시도(최대 max_attempts회). ai/meeting의
    _safe_call_structured_json과 같은 정책(최초 1회 + 재시도, 실패 사유만 로그에 남기고
    프롬프트 원문·LLM 응답 원문은 남기지 않음)."""
    last_reason = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            raw = parse_json_response(llm_call(prompt))
        except (ValueError, KeyError, TypeError):
            last_reason = "json_parse_failed"
            continue
        problem = validate(raw)
        if problem is None:
            return raw, True, attempt
        last_reason = problem
    logger.warning("[rag_quality.judge] 구조화 응답 검증 실패 reason=%s", last_reason)
    return None, False, max_attempts


def _validate_faithfulness_response(raw: dict) -> Optional[str]:
    claims = raw.get("claims")
    if not isinstance(claims, list):
        return "claims_missing_or_not_list"
    for claim in claims:
        if not isinstance(claim, dict):
            return "claim_not_object"
        if not (claim.get("claim") or "").strip():
            return "missing_or_empty_field:claim"
        if claim.get("verdict") not in _VALID_VERDICTS:
            return "invalid_verdict"
    return None


def _validate_persona_fit_response(raw: dict) -> Optional[str]:
    score = raw.get("score")
    if not isinstance(score, int) or not (0 <= score <= 4):
        return "invalid_score"
    return None


def _validate_expert_judgment_response(raw: dict) -> Optional[str]:
    for field in ("role_fit", "logic", "actionability"):
        value = raw.get(field)
        if not isinstance(value, int) or not (0 <= value <= 2):
            return f"invalid_{field}"
    if not isinstance(raw.get("conflicts_with_grounded_fact"), bool):
        return "invalid_conflicts_with_grounded_fact"
    return None


def judge_faithfulness(
    judge_llm_call: JudgeLLMCall,
    *,
    model: str,
    persona_id: str,
    statement_content: str,
    retrieved_context: list[dict],
    cache: Optional[JudgeCache] = None,
) -> tuple[list[ClaimVerdict], Optional[str]]:
    """반환값 (claims, error) — error가 있으면 claims는 빈 리스트다(재시도 후에도 구조화
    응답 검증 실패)."""
    template = _read_prompt("faithfulness_judge.txt")
    prompt = _render(
        template,
        {
            "<<PERSONA_ID>>": persona_id,
            "<<STATEMENT_CONTENT>>": statement_content,
            "<<RETRIEVED_CONTEXT_JSON>>": _as_text(retrieved_context),
        },
    )

    if cache is not None:
        cached = cache.get(prompt, model, FAITHFULNESS_PROMPT_VERSION)
        if cached is not None:
            raw = cached
        else:
            raw, ok, _ = _safe_call_json_retry(judge_llm_call, prompt, _validate_faithfulness_response)
            if not ok or raw is None:
                return [], "faithfulness_judge_failed_validation"
            cache.set(prompt, model, FAITHFULNESS_PROMPT_VERSION, raw)
    else:
        raw, ok, _ = _safe_call_json_retry(judge_llm_call, prompt, _validate_faithfulness_response)
        if not ok or raw is None:
            return [], "faithfulness_judge_failed_validation"

    claims = [
        ClaimVerdict(
            claim=c.get("claim", ""),
            verdict=c.get("verdict"),
            supporting_document_ids=[d for d in (c.get("supporting_document_ids") or []) if isinstance(d, str)],
            supporting_evidence_excerpt=c.get("supporting_evidence_excerpt"),
            reason=c.get("reason", ""),
            confidence=float(c.get("confidence") or 0.0),
        )
        for c in raw.get("claims", [])
    ]
    return claims, None


def judge_persona_fit(
    judge_llm_call: JudgeLLMCall,
    *,
    model: str,
    persona_id: str,
    message_id: str,
    statement_content: str,
    retrieved_context: list[dict],
    cache: Optional[JudgeCache] = None,
) -> tuple[Optional[PersonaFitResult], Optional[str]]:
    template = _read_prompt("persona_fit_judge.txt")
    prompt = _render(
        template,
        {
            "<<PERSONA_ID>>": persona_id,
            "<<STATEMENT_CONTENT>>": statement_content,
            "<<RETRIEVED_CONTEXT_JSON>>": _as_text(retrieved_context),
        },
    )

    raw = cache.get(prompt, model, PERSONA_FIT_PROMPT_VERSION) if cache is not None else None
    if raw is None:
        raw, ok, _ = _safe_call_json_retry(judge_llm_call, prompt, _validate_persona_fit_response)
        if not ok or raw is None:
            return None, "persona_fit_judge_failed_validation"
        if cache is not None:
            cache.set(prompt, model, PERSONA_FIT_PROMPT_VERSION, raw)

    score = int(raw["score"])
    return (
        PersonaFitResult(
            persona_id=persona_id,  # type: ignore[arg-type]
            message_id=message_id,
            score=score,
            normalized_score=score / 4.0,
            evidence_document_ids=[d for d in (raw.get("evidence_document_ids") or []) if isinstance(d, str)],
            role_aligned_points=[p for p in (raw.get("role_aligned_points") or []) if isinstance(p, str)],
            role_mismatch_points=[p for p in (raw.get("role_mismatch_points") or []) if isinstance(p, str)],
            rationale=raw.get("rationale", ""),
        ),
        None,
    )


def judge_expert_judgment(
    judge_llm_call: JudgeLLMCall,
    *,
    model: str,
    persona_id: str,
    message_id: str,
    claim_id: Optional[str],
    statement_content: str,
    grounded_claims: list[dict],
    cache: Optional[JudgeCache] = None,
) -> tuple[Optional[ExpertJudgmentResult], Optional[str]]:
    """용준/Claude(2026-07-29, 요청: 생성 품질 개선 — expert_judgment 별도 평가). 반환값
    (result, error) — judge_persona_fit과 동일한 재시도/캐시 정책을 재사용한다."""
    template = _read_prompt("expert_judgment_judge.txt")
    prompt = _render(
        template,
        {
            "<<PERSONA_ID>>": persona_id,
            "<<STATEMENT_CONTENT>>": statement_content,
            "<<GROUNDED_CLAIMS_JSON>>": _as_text(grounded_claims),
        },
    )

    raw = cache.get(prompt, model, EXPERT_JUDGMENT_PROMPT_VERSION) if cache is not None else None
    if raw is None:
        raw, ok, _ = _safe_call_json_retry(judge_llm_call, prompt, _validate_expert_judgment_response)
        if not ok or raw is None:
            return None, "expert_judgment_judge_failed_validation"
        if cache is not None:
            cache.set(prompt, model, EXPERT_JUDGMENT_PROMPT_VERSION, raw)

    role_fit = int(raw["role_fit"])
    logic = int(raw["logic"])
    actionability = int(raw["actionability"])
    return (
        ExpertJudgmentResult(
            persona_id=persona_id,  # type: ignore[arg-type]
            message_id=message_id,
            claim_id=claim_id,
            role_fit=role_fit,
            logic=logic,
            actionability=actionability,
            normalized_score=(role_fit + logic + actionability) / 6.0,
            conflicts_with_grounded_fact=bool(raw.get("conflicts_with_grounded_fact", False)),
            conflict_detail=raw.get("conflict_detail", ""),
            rationale=raw.get("rationale", ""),
        ),
        None,
    )
