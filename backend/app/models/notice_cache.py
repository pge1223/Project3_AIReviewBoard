from datetime import datetime
from typing import Optional
from bson import ObjectId


# 가은/Claude(2026-07-23, 요청: 공고문 분석 결과 프로젝트 간 재사용) — 기존
# project.announcement_analysis_cache는 프로젝트 1개 안에서만 재사용되는 캐시라, 다른
# 사용자가 완전히 같은 공고문(같은 URL/같은 파일)을 올려도 매번 새로 LLM을 호출했다.
# 이 컬렉션은 계정·프로젝트와 무관하게 cache_key(정규화 URL 또는 파일 내용 해시) 하나로
# 전역 공유된다 — 공고문은 공개 정보라 프로젝트 간 공유해도 개인정보 문제가 없다.
#
# 가은/Claude(2026-07-23, 요청: 신청서양식 분석도 같은 캐시로 확장) — analysis_kind
# ("announcement" | "application_form")를 추가했다. document_type이 아직 없는(미분류)
# 문서는 announcement/application_form 양쪽 그룹에 다 포함되므로(_load_criteria_documents_text
# 참고), 같은 cache_key가 두 분석 종류 모두에서 조회될 수 있다 — cache_key만으로는
# 어느 쪽 응답인지 구분이 안 되므로 반드시 analysis_kind와 함께 조회/저장해야 한다.
class NoticeAnalysisCacheModel:
    collection_name = "notice_analysis_cache"

    def __init__(
        self,
        cache_key: str,
        analysis_kind: str,
        analysis_version: int,
        response: dict,  # AnnouncementAnalysisResponse 또는 ApplicationFormAnalysisResponse의 model_dump()
        created_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None,
    ):
        self._id = _id
        self.cache_key = cache_key
        self.analysis_kind = analysis_kind
        self.analysis_version = analysis_version
        self.response = response
        self.created_at = created_at or datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "cache_key": self.cache_key,
            "analysis_kind": self.analysis_kind,
            "analysis_version": self.analysis_version,
            "response": self.response,
            "created_at": self.created_at,
        }
