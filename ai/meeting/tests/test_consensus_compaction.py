import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_nodes import _compact_consensus_items


def test_consensus_is_short_deduplicated_and_limited_to_four_items():
    long_item = (
        "가장 중점적으로 둔 부분은 기능 중심이 아니라 업무 중심으로 접근하는 것입니다. "
        "AI가 무엇을 할 수 있는지보다 실제 업무 시간을 얼마나 줄이는지를 기준으로 설계합니다. "
        "이후 여러 사례와 배경 설명이 계속 이어집니다."
    )

    result = _compact_consensus_items(
        ["문제 정의와 대상 사용자를 구체화하기로 합의했습니다.", long_item],
        [
            "문제 정의와 대상 사용자를 구체화하기로 합의했습니다.",
            "성공 지표를 측정 가능한 형태로 정합니다.",
            "MVP 범위를 핵심 업무로 제한합니다.",
            "사용자 인터뷰로 가정을 검증합니다.",
        ],
    )

    assert len(result) == 4
    assert result[-1] == "사용자 인터뷰로 가정을 검증합니다."
    assert all(len(item) <= 91 for item in result)
    assert all("이후 여러 사례" not in item for item in result)
