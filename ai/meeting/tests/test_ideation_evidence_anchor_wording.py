# 작성자: 용준/Claude(2026-07-30, 요청: "문서 근거 0건이면 '문서 근거를 확인했습니다'
#       문구를 쓰지 마세요") — _evidence_anchor_response()가 target(선택 아이디어)만
#       앵커로 삼았을 때와 criteria/external(실제 문서)을 앵커로 삼았을 때 서로 다른
#       문구를 쓰는지 검증한다.

import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_nodes import _evidence_anchor_response  # noqa: E402

_BASE_RAW = {"active_issue_title": "핵심 쟁점"}


def test_target_only_anchor_does_not_claim_document_evidence():
    retrieved = [
        {
            "ref": "E1",
            "chunk_id": "T1",
            "document_role": "target",
            "text": "정기적인 설문조사 및 인터뷰를 통해 피드백을 수집하여 솔루션 개선에 활용합니다.",
        }
    ]
    result = _evidence_anchor_response(_BASE_RAW, retrieved, "planning_expert")
    assert result is not None
    assert "문서 근거를 확인해 정리했습니다" not in result["spoken_text"]
    assert "현재 선택된 아이디어를 검토한" in result["spoken_text"]


def test_criteria_anchor_keeps_document_evidence_wording():
    retrieved = [
        {
            "ref": "E1",
            "chunk_id": "C1",
            "document_role": "criteria",
            "text": "실현 가능성과 차별성을 중점적으로 평가한다.",
        }
    ]
    result = _evidence_anchor_response(_BASE_RAW, retrieved, "planning_expert")
    assert result is not None
    assert "문서 근거를 확인해 정리했습니다" in result["spoken_text"]
