# 작성자: 경이
# 목적: ai/meeting/scoring 패키지 공개 인터페이스. 점수 계산·점수 설명 카드·수정 전후
#       비교 진입점을 노출한다.
# import: 같은 패키지의 calculator, weights, deductions, explanation, comparison.

from .calculator import calculate_score
from .comparison import build_revision_comparison, build_version_history
from .deductions import compute_penalties
from .explanation import build_score_explanation
from .personalization import (
    TECHNICAL_PERSONA_IDS,
    attach_impl_guides,
    build_impl_guide,
    build_impl_guide_prompt,
    classify_impl_difficulty,
    is_technical_persona,
)
from .weights import resolve_weights

__all__ = [
    "TECHNICAL_PERSONA_IDS",
    "attach_impl_guides",
    "build_impl_guide",
    "build_impl_guide_prompt",
    "build_revision_comparison",
    "build_score_explanation",
    "build_version_history",
    "calculate_score",
    "classify_impl_difficulty",
    "compute_penalties",
    "is_technical_persona",
    "resolve_weights",
]
