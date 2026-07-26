from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator

from ai.rag.loaders.schemas import UrlExtractionResult


class DocumentResponse(BaseModel):
    id: str
    project_id: str
    user_email: str
    original_filename: str
    stored_filename: str
    file_path: str
    file_size: int
    mime_type: str
    source_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    document_role: str = "target"
    document_type: Optional[str] = None
    # 가은/Claude(2026-07-16): HWP/HWPX 변환 결과(성공/실패/불필요) — 프론트가 실패 시
    # conversion_metadata.conversion_error(user_message)를 그대로 보여준다.
    conversion_metadata: Optional[dict] = None
    # 가은/Claude(2026-07-18): URL 공고문 수집 시 발견됐지만 자동으로 못 읽은 첨부파일
    # (HWP/HWPX) — 프론트가 "직접 받아서 파일 업로드 탭으로 올려주세요" 안내 + 다운로드
    # 링크를 보여준다. [{"url", "file_name", "reason"}]
    unsupported_attachments: Optional[list[dict]] = None


class FetchUrlRequest(BaseModel):
    url: str
    # 가은/Claude (2026-07-15): project_id가 있으면 RAG 색인까지 하고 documents 컬렉션에
    # document_role="criteria"로 저장한다. 없으면(과거 호출 호환) 조회만 하고 저장하지 않는다.
    project_id: Optional[str] = None

    @field_validator("url")
    @classmethod
    def _url_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("url은 빈 문자열일 수 없습니다")
        return v


# 가은/Claude(2026-07-19, INF-007 — fetch-url 색인 백그라운드화): UrlExtractionResult는
# ai/rag/loaders/schemas.py(용준 담당) 소유라 그 파일은 건드리지 않고, 백엔드 쪽에서
# 상속으로 필드만 확장한다. 색인(Chroma 임베딩)이 더 이상 응답을 막지 않으므로 — 이
# document_id/document_status로 GET /{project_id}/{document_id}/status(기존 DOC-004
# 엔드포인트, 신규 아님)를 폴링해서 색인 완료 여부를 알 수 있다. project_id를 안 보낸
# 호출(과거 호환, 조회만)이나 page_content가 없는 경우(직접 파일 링크 등)는 색인 자체가
# 없으므로 둘 다 None으로 남는다.
class FetchUrlResponse(UrlExtractionResult):
    document_id: Optional[str] = None
    document_status: Optional[str] = None


# 가은/Claude(2026-07-21): "공모전 분석" 화면(사용자 UX 스펙, 2026-07-21) — 공고문에서
# 실제로 확인되는 사실(official_facts)과 AI가 추론한 전략적 분석(strategic_analysis)을
# 분리해서 반환한다. 공고문에 없는 정보를 지어내지 않는다 — 못 찾은 필드는 빈 배열/
# "미공개" 문자열로 명시한다(값을 비워두는 대신 근거 없음 자체를 표시).
class ScheduleItem(BaseModel):
    event_label: str
    start_date: str
    end_date: str = ""
    start_weekday: str = ""
    end_weekday: str = ""
    method: str = ""
    source_text: str = ""


class OfficialFacts(BaseModel):
    eligibility: list[str] = []
    deadline: str = "미공개"
    submission_requirements: list[str] = []
    evaluation_criteria: list[str] = []
    disqualification_rules: list[str] = []
    # 공고문에는 접수 마감 외에도 평가/발표/시상 일정이 따로 있으며, 제출 서류와
    # 심사 운영 조건도 성격이 다르다. 기존 필드는 유지하고 상세 사실을 별도 배열로
    # 내려 이전 클라이언트와의 호환성을 보존한다.
    application_review_conditions: list[str] = []
    key_dates: list[str] = []
    # key_dates는 기존 클라이언트 호환용으로 유지하고, 화면에서는 날짜의 의미를 잃지
    # 않도록 행사명·기간·발표 방법이 분리된 구조화 일정을 우선 사용한다.
    schedule_items: list[ScheduleItem] = []
    selection_benefits: list[str] = []


class StrategicAnalysis(BaseModel):
    core_intent: str = ""
    winning_points: list[str] = []
    recommended_direction: list[str] = []
    risk_flags: list[str] = []


class AnnouncementEvidence(BaseModel):
    claim: str
    # "announcement": 공고문 원문에 직접 근거가 있음 / "inference": AI가 추론한 내용(원문에 명시 없음)
    source_type: str
    location: Optional[str] = None
    confidence: str = "medium"


# 가은/Claude(2026-07-21): kyh님이 크롤링(contest_works, 소통혁신24)해서 category/
# source_org까지 채운(scripts/classify_contest_works.py) 수상작 아카이브가 생겨서, 여기
# 붙였던 "데이터 소스 자체가 없다"는 이전 제약은 더 이상 사실이 아니다. LLM이 원문에서
# 안 나오는 개별 수상작을 지어내는 건 여전히 막고, 대신 실제 DB에서 조회한 값만 담는다
# (없으면 has_similar_case_data=False로 그대로 "미확보" 상태를 보여준다).
class SimilarWork(BaseModel):
    title: str
    source_org: str = ""
    award_grade: str = ""
    selection_status: str = ""  # "winner" | "candidate"
    # 가은/Claude(2026-07-21): 실측 요청 — 카드에서 이 항목을 클릭하면 같은 공모전
    # (contest_title)의 다른 수상작/후보작을 옆 패널에서 더 보여준다. 프론트가 그 조회에
    # 쓸 키를 여기서 같이 내려준다.
    contest_title: str = ""


# 가은/Claude(2026-07-21): SimilarWork 클릭 시 상세 패널 — 같은 contest_title 안에서
# 어떤 아이디어가 수상하고 어떤 게 후보에 그쳤는지 전부 보여준다. ocr_text는 kyh님이
# 대표작 일부만 검증 후 확대 예정이라 아직 전부 비어있을 수 있다(그때는 빈 문자열).
class ContestWorkDetail(BaseModel):
    work_title: str
    award_grade: str = ""
    selection_status: str = ""  # "winner" | "candidate"
    images: list[str] = []
    ocr_text: str = ""
    # 가은/Claude(2026-07-21): kyh님이 크롤러에 추가한 필드 — 소통혁신24 원문 상세
    # 페이지 링크. 없는(예전에 크롤링된) 문서도 있을 수 있어 빈 문자열 기본값.
    source_url: str = ""


class ContestWorksByTitleResponse(BaseModel):
    contest_title: str
    works: list[ContestWorkDetail] = []


class AnnouncementAnalysisResponse(BaseModel):
    has_announcement: bool
    # 가은/Claude(2026-07-21): 실측 요청 — 화면 제목에 실제 공모전명이 나오게. 페이지
    # <title>은 "공지사항 - 부서명" 같은 사이트 boilerplate인 경우가 많아 신뢰 못 함
    # (실측: 외교부 공고 사례) — 본문에서 LLM이 직접 뽑는다. 못 찾으면 빈 문자열.
    announcement_title: str = ""
    official_facts: Optional[OfficialFacts] = None
    strategic_analysis: Optional[StrategicAnalysis] = None
    evidence: list[AnnouncementEvidence] = []
    has_similar_case_data: bool = False
    similar_works: list[SimilarWork] = []
    source_document_names: list[str] = []


# 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입 → 업로드 영역 통합): "공모전 공고 ·
# 평가기준 · 신청서 양식"을 한 업로드 영역(document_role="criteria")으로 합쳤으므로, 이
# 엔드포인트는 announcement-analysis와 같은 문서 풀을 신청서 양식 관점으로 다시 읽어
# 기입해야 하는 항목만 뽑는다 — announcement-analysis와 같은 "프로젝트당 1회 계산 후 캐시"
# 패턴. 이 항목들은 ① 아이디어 회의 화면에 그대로 보여주고, ② 회의 discussion 프롬프트에
# "참고 자료"로 약하게 주입한다(ai/meeting/prompts/ideation_conv_discussion.txt의
# [신청양식 참고 규칙] — 질문 주제·순서는 안 바꾸고 표현만 다듬는 용도). 양식에 없는 항목을
# 지어내지 않는다 — 못 찾은 필드는 담지 않는다(개수 자체가 그대로 신뢰도를 보여준다).
class ApplicationFormItem(BaseModel):
    field_name: str
    description: str = ""
    # 양식에 글자 수 제한이 명시돼 있지 않으면 None(모른다는 뜻 — 0이나 임의값을 지어내지 않는다).
    char_limit: Optional[int] = None


class ApplicationFormAnalysisResponse(BaseModel):
    # has_application_form: "criteria 문서를 하나라도 등록해 실제로 LLM 분석을 시도했는가"를
    # 뜻한다 — 업로드 영역이 통합돼 그 문서들이 실제 신청서 양식을 포함한다는 보장은 없으므로,
    # "신청서 양식이 실제로 발견됐는가"는 items가 비어 있는지로 판단해야 한다(True + items=[]는
    # "문서는 있었지만 신청서 양식 기입란은 못 찾았다"는 유효한 상태다).
    has_application_form: bool
    items: list[ApplicationFormItem] = []
    source_document_names: list[str] = []
