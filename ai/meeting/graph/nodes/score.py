# 작성자: 경이
# 목적: LangGraph score 노드(M4, MTG-003). 모든 위원 결과가 모인 뒤(fan-in) 기존
#       ai/meeting/scoring 계산 엔진(M2, 결정론적 Python 계산 — LLM이 아니라 규칙으로
#       계산해 동일 입력=동일 출력을 보장)을 호출해 score_result를 채운다.
# import: scoring.calculate_score(형제 패키지 — 호출 시점에 ai/meeting이 sys.path에
#         있어야 한다), 같은 패키지의 state.

from __future__ import annotations

from scoring import calculate_score

from ..state import MeetingState


def score_node(state: MeetingState) -> dict:
    reviewers = [
        {"review_id": result.get("review_id", persona_id), "rubric_scores": result["rubric_scores"]}
        for persona_id, result in state["reviewer_results"].items()
    ]
    score_result = calculate_score(state["rubric"], reviewers, submission=state.get("submission"))
    return {"score_result": score_result, "stage": "평가"}
