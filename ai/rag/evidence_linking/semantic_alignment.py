# 작성자: 용준/Claude(2026-07-30, 요청: "claim-evidence 의미 정합성 2차 검사 — shadow mode")
# 목적: claim_grounding.py::ground_claims()의 키워드 stem 겹침 검사(1차, 검색 후보 필터)를
#       대체하지 않고, 그 검사를 통과해 실제로 LLM이 인용한 (claim, evidence) 쌍에만
#       LLM judge 한 번(배치)으로 의미 정합성을 재검증한다.
#
#       이 모듈은 항상 shadow 모드로만 동작한다 — ground_claims()의 반환값을 절대 바꾸지
#       않고, 판정 결과를 trace_sink로 로그만 남긴다. 실제로 topic_only/contradicted 쌍을
#       근거에서 제거하거나 claim을 강등/차단하는 로직(적용 순서 5단계)은 이번에 구현하지
#       않는다 — shadow 로그를 사람이 검토한 뒤 별도로 활성화한다.
#
#       LLM 호출 패턴은 ai/rag/evaluation/rag_quality/judge.py를 그대로 재사용한다(새 클라이언트/
#       재시도 로직을 재발명하지 않는다) — 차이는 judge.py가 "메시지 1건당 판정 1개"인 반면
#       이 모듈은 "이번 턴의 모든 (claim, evidence) 쌍을 프롬프트 하나에 담아 LLM 호출 1회"로
#       배치 처리한다는 점뿐이다.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal, Optional, TypedDict

from ai.rag.evaluation.rag_quality._meeting_path import ensure_meeting_on_path
from ai.rag.evidence_linking.claim_grounding import ground_claims
from ai.rag.evidence_linking.config import EvidenceLinkingConfig

ensure_meeting_on_path()
from graph.llm import parse_json_response  # noqa: E402

logger = logging.getLogger(__name__)

JudgeLLMCall = Callable[[str], str]

AlignmentLabel = Literal["entailed", "partially_entailed", "topic_only", "contradicted"]
_VALID_LABELS: frozenset[str] = frozenset({"entailed", "partially_entailed", "topic_only", "contradicted"})

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
CLAIM_EVIDENCE_ALIGNMENT_PROMPT_VERSION = "claim_evidence_alignment_judge_v1"


class AlignmentPair(TypedDict):
    claim_id: str
    claim_text: str
    evidence_ref: str
    chunk_id: str
    evidence_text: str
    evidence_meta: dict


class AlignmentJudgment(TypedDict):
    claim_id: str
    chunk_id: str
    label: AlignmentLabel
    reason: str


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


def _pair_key(claim_id: str, chunk_id: str) -> str:
    return f"{claim_id}::{chunk_id}"


def build_pairs(
    claim_evidence_links: list[dict],
    claims: list[dict],
    retrieved_evidence: list[dict],
) -> list[AlignmentPair]:
    """ground_claims()의 결과(claim_evidence_links)와 원본 claims/retrieved_evidence를
    대조해, 실제로 LLM이 인용하고 관련성 검증까지 통과한 (claim, evidence) 쌍만 만든다.
    검색됐지만 인용되지 않은 청크는 대상에 포함하지 않는다(요청: "검색된 전체 청크를
    모두 검사하지 않음")."""
    claim_text_by_id = {
        str(c.get("claim_id")): str(c.get("text") or "") for c in claims if isinstance(c, dict) and c.get("claim_id")
    }
    evidence_by_chunk_id = {
        str(item.get("chunk_id")): item
        for item in retrieved_evidence
        if isinstance(item, dict) and item.get("chunk_id")
    }
    pairs: list[AlignmentPair] = []
    seen: set[str] = set()
    for link in claim_evidence_links:
        if not isinstance(link, dict):
            continue
        claim_id = str(link.get("claim_id") or "")
        claim_text = claim_text_by_id.get(claim_id)
        if not claim_id or not claim_text:
            continue
        refs = link.get("evidence_refs") or []
        chunk_ids = link.get("chunk_ids") or []
        for ref, chunk_id in zip(refs, chunk_ids):
            chunk_id = str(chunk_id)
            key = _pair_key(claim_id, chunk_id)
            if key in seen:
                continue
            evidence_item = evidence_by_chunk_id.get(chunk_id)
            if evidence_item is None:
                continue
            evidence_text = str(evidence_item.get("text") or evidence_item.get("quote") or "")
            if not evidence_text.strip():
                continue
            seen.add(key)
            pairs.append(
                AlignmentPair(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    evidence_ref=str(ref),
                    chunk_id=chunk_id,
                    evidence_text=evidence_text,
                    evidence_meta={
                        "document_name": evidence_item.get("document_name"),
                        "publisher": evidence_item.get("publisher"),
                        "reference_date": evidence_item.get("reference_date"),
                        "published_at": evidence_item.get("published_at"),
                        "section": evidence_item.get("section"),
                        "source_type": evidence_item.get("source_type"),
                    },
                )
            )
    return pairs


def _fallback_judgments(pairs: list[AlignmentPair], reason: str) -> list[AlignmentJudgment]:
    """판정 실패/타임아웃 시 모든 쌍을 topic_only로 보수적으로 강등한다(요청: "근거를
    낙관적으로 승인하지 말고 expert_judgment로 강등") — entailed로 낙관 처리하지 않는다."""
    logger.warning(
        "[semantic_alignment] 판정 실패로 모든 쌍을 topic_only로 강등 pair_count=%d reason=%s",
        len(pairs),
        reason,
    )
    return [
        AlignmentJudgment(claim_id=p["claim_id"], chunk_id=p["chunk_id"], label="topic_only", reason=reason)
        for p in pairs
    ]


def _validate_alignment_response(raw: dict, expected_pairs: list[AlignmentPair]) -> Optional[str]:
    judgments = raw.get("judgments")
    if not isinstance(judgments, list):
        return "judgments_missing_or_not_list"
    expected_keys = {_pair_key(p["claim_id"], p["chunk_id"]) for p in expected_pairs}
    seen_keys: set[str] = set()
    for item in judgments:
        if not isinstance(item, dict):
            return "judgment_not_object"
        claim_id = item.get("claim_id")
        chunk_id = item.get("chunk_id")
        if not isinstance(claim_id, str) or not isinstance(chunk_id, str):
            return "missing_claim_or_chunk_id"
        if item.get("label") not in _VALID_LABELS:
            return "invalid_label"
        seen_keys.add(_pair_key(claim_id, chunk_id))
    if not expected_keys.issubset(seen_keys):
        return "missing_judgment_for_pair"
    return None


def judge_claim_evidence_alignment(
    pairs: list[AlignmentPair],
    llm_call: JudgeLLMCall,
    *,
    model: str,
    max_attempts: int = 2,
) -> list[AlignmentJudgment]:
    """한 턴의 모든 (claim, evidence) 쌍을 프롬프트 하나에 담아 LLM을 한 번만 호출한다.
    파싱 실패·검증 실패·예외 발생 시 모든 쌍을 topic_only로 보수적으로 강등해 반환한다
    (예외를 던지지 않는다 — 호출부가 shadow 로깅 실패로 회의를 막지 않아야 한다)."""
    if not pairs:
        return []

    template = _read_prompt("claim_evidence_alignment.txt")
    prompt = _render(
        template,
        {
            "<<PAIRS_JSON>>": _as_text(
                [
                    {
                        "claim_id": p["claim_id"],
                        "chunk_id": p["chunk_id"],
                        "claim_text": p["claim_text"],
                        "evidence_text": p["evidence_text"],
                        "evidence_meta": p["evidence_meta"],
                    }
                    for p in pairs
                ]
            ),
        },
    )

    last_reason = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            raw = parse_json_response(llm_call(prompt))
        except (ValueError, KeyError, TypeError):
            last_reason = "json_parse_failed"
            continue
        except Exception:  # noqa: BLE001 — LLM 호출 자체의 예기치 못한 실패도 shadow에서는 무해해야 한다
            logger.exception("[semantic_alignment] LLM 호출 중 예기치 못한 오류")
            return _fallback_judgments(pairs, "llm_call_unexpected_error")
        problem = _validate_alignment_response(raw, pairs)
        if problem is None:
            judgments_by_key = {
                _pair_key(str(item["claim_id"]), str(item["chunk_id"])): AlignmentJudgment(
                    claim_id=str(item["claim_id"]),
                    chunk_id=str(item["chunk_id"]),
                    label=item["label"],
                    reason=str(item.get("reason") or ""),
                )
                for item in raw["judgments"]
            }
            return [
                judgments_by_key[_pair_key(p["claim_id"], p["chunk_id"])]
                for p in pairs
                if _pair_key(p["claim_id"], p["chunk_id"]) in judgments_by_key
            ]
        last_reason = problem

    return _fallback_judgments(pairs, last_reason)


def ground_claims_with_alignment_shadow(
    raw_claims,
    retrieved_evidence: list[dict],
    *,
    role_keywords: list[str] | None = None,
    config: EvidenceLinkingConfig | None = None,
    llm_call: Optional[JudgeLLMCall] = None,
    model: str = "",
    trace_sink: Optional[Callable[[dict], None]] = None,
) -> dict:
    """ground_claims()를 그대로 호출한 뒤(1차 lexical 검사, 결과 변경 없음), 그 결과의
    claim_evidence_links에 대해서만 배치 LLM judge를 한 번 더 돌려 shadow 로그를 남긴다.

    이 함수는 항상 shadow다 — llm_call이 없거나(플래그 꺼짐) 판정 대상 쌍이 없으면
    ground_claims()의 원래 반환값을 그대로 돌려주고, 있어도 반환값 자체는 절대 바꾸지
    않는다(적용 순서 5단계에서 실제 강등/차단 로직을 별도로 추가할 예정 — 요청: "shadow
    mode 검증 후 활성화"). trace_sink에는 매 턴 최대 1번(판정 대상 쌍이 있을 때만)
    IDEATION_CLAIM_EVIDENCE_ALIGNMENT_SHADOW 이벤트를 넘긴다 — 호출부가 trace_event나
    로거로 그대로 연결하면 된다."""
    result = ground_claims(raw_claims, retrieved_evidence, role_keywords=role_keywords, config=config)
    if llm_call is None:
        return result

    pairs = build_pairs(result["claim_evidence_links"], result["claims"], retrieved_evidence)
    if not pairs:
        return result

    try:
        judgments = judge_claim_evidence_alignment(pairs, llm_call, model=model)
    except Exception:  # noqa: BLE001 — shadow 로깅 실패가 회의 흐름을 막으면 안 된다
        logger.exception("[semantic_alignment] shadow 판정 중 예기치 못한 오류 — 근거 표시에는 영향 없음")
        judgments = _fallback_judgments(pairs, "shadow_wrapper_unexpected_error")

    if trace_sink is not None:
        label_counts: dict[str, int] = {}
        for j in judgments:
            label_counts[j["label"]] = label_counts.get(j["label"], 0) + 1
        trace_sink(
            {
                "event": "IDEATION_CLAIM_EVIDENCE_ALIGNMENT_SHADOW",
                "pair_count": len(pairs),
                "label_counts": label_counts,
                "judgments": [dict(j) for j in judgments],
            }
        )

    return result


__all__ = [
    "AlignmentPair",
    "AlignmentJudgment",
    "AlignmentLabel",
    "JudgeLLMCall",
    "CLAIM_EVIDENCE_ALIGNMENT_PROMPT_VERSION",
    "build_pairs",
    "judge_claim_evidence_alignment",
    "ground_claims_with_alignment_shadow",
]
