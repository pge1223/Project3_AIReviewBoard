"""Application-form coaching orchestration.

The backend injects answer assessment, expert review, and RAG adapters so this
package can reuse existing capabilities without depending on API modules.
"""

from .answer_assessment import (
    AnswerAssessmentProvider,
    LLMAnswerAssessmentProvider,
    NullAnswerAssessmentProvider,
)
from .evidence_provider import EvidenceProvider, LookupEvidenceProvider, NullEvidenceProvider
from .expert_delegation import (
    ExpertDelegationProvider,
    LLMExpertDelegationProvider,
    NullExpertDelegationProvider,
)
from .expert_guidance import (
    ExpertGuidanceProvider,
    LLMExpertGuidanceProvider,
    NullExpertGuidanceProvider,
)
from .expert_review import ExpertReviewProvider, LLMExpertReviewProvider, NullExpertReviewProvider
from .service import finalize_session, reply_to_session, start_session, update_application_draft
from .synthesis import LLMSynthesisProvider, NullSynthesisProvider, SynthesisProvider

__all__ = [
    "AnswerAssessmentProvider",
    "EvidenceProvider",
    "ExpertReviewProvider",
    "ExpertDelegationProvider",
    "ExpertGuidanceProvider",
    "LLMAnswerAssessmentProvider",
    "LLMExpertReviewProvider",
    "LLMExpertDelegationProvider",
    "LLMExpertGuidanceProvider",
    "LLMSynthesisProvider",
    "LookupEvidenceProvider",
    "NullAnswerAssessmentProvider",
    "NullEvidenceProvider",
    "NullExpertReviewProvider",
    "NullExpertDelegationProvider",
    "NullExpertGuidanceProvider",
    "NullSynthesisProvider",
    "SynthesisProvider",
    "finalize_session",
    "reply_to_session",
    "start_session",
    "update_application_draft",
]
