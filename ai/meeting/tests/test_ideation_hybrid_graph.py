import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_build import _route_after_candidate_selection
from graph.ideation_conv_state import initial_conv_state


def _selected_state(conversation_flow: str) -> dict:
    return {
        "phase": "candidate_selection",
        "next_route": "to_refinement",
        "application_form_items": [{"field_name": "현황 및 문제점"}],
        "conversation_flow": conversation_flow,
    }


def test_form_coach_v2_hands_off_after_candidate_selection():
    assert _route_after_candidate_selection(_selected_state("form_coach_v2")) == "to_form_coach"


def test_hybrid_graph_asks_user_before_running_expert_roundtable():
    assert (
        _route_after_candidate_selection(_selected_state("hybrid_graph"))
        == "to_initial_user_question"
    )


def test_initial_state_preserves_hybrid_flow_mode():
    state = initial_conv_state(
        "HYBRID-TEST",
        {"competition_name": "테스트 공모전"},
        {},
        conversation_flow="hybrid_graph",
    )

    assert state["conversation_flow"] == "hybrid_graph"
