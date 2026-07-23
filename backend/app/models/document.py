from datetime import datetime
from typing import Optional
from bson import ObjectId

class DocumentModel:
    collection_name = "documents"

    def __init__(
        self,
        project_id: str,
        user_email: str,
        original_filename: str,
        stored_filename: str,
        file_path: str,
        file_size: int,
        mime_type: str,
        # status 값: "uploaded", "pending", "indexed", "indexed_empty", "indexing_failed",
        # "conversion_failed" | 윤한/Claude(2026-07-18, INF-007 fetch-url 백그라운드화):
        # "indexing"(색인 진행 중, fetch-url이 응답을 기다리지 않고 즉시 반환한 뒤 백그라운드
        # 태스크가 색인을 마치면 indexed/indexed_empty로 patch) / "indexing_timeout"
        # (백그라운드 색인이 asyncio.wait_for(timeout=120)에 걸려 중단된 경우) 추가.
        status: str = "uploaded",
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        source_type: str = "pdf",
        # 가은/Claude (2026-07-15, "다 이어버리자" 작업 — 윤한 합의 필요 항목):
        # document_role("target"=평가 대상 문서/기획서, "criteria"=공고문·평가기준)이 없으면
        # analyze_project()가 어떤 문서를 review 대상으로 삼을지, 어떤 문서를 RAG 근거로만
        # 쓸지 구분할 방법이 없었다. 프론트 DocumentUploadPage.jsx의 두 드롭존(왼쪽 "평가
        # 대상 문서" / 오른쪽 "기준 문서·공고문")과 1:1로 대응시켰다.
        document_role: str = "target",
        document_type: Optional[str] = None,
        # parsed_text: RAG-001(파싱) 결과 블록을 이어붙인 원문 텍스트. 색인(Chroma)은
        # 벡터/청크 단위라 "이 문서 전체 원문"을 그대로 돌려주는 용도로는 안 맞아서,
        # analyze_project()가 submission.text로 바로 쓸 수 있게 문서 레코드에 같이 저장한다.
        parsed_text: Optional[str] = None,
        # 가은/Claude(2026-07-16): HWP/HWPX -> PDF 변환 통합(용준, ai/rag/converters/
        # INTEGRATION.md). build_conversion_metadata()가 만드는 DocumentConversionMetadata를
        # dict로 그대로 저장 — 새 컬럼 여러 개 대신 dict 하나로 묶음(가이드 3번 권장 사항).
        conversion_metadata: Optional[dict] = None,
        # 가은/Claude(2026-07-18): URL 공고문 수집 시 발견됐지만 자동으로 못 읽은 첨부파일
        # (지금은 HWP/HWPX만 해당, ai/rag/loaders/url_loader.py의 UnsupportedAttachment를
        # {"url", "file_name", "reason"} dict로 그대로 저장) — 실측(sotong.go.kr): 평가
        # 기준이 본문이 아니라 HWP 요강 파일에만 있는 공고가 실제로 있어서, 사용자가
        # 그 파일을 직접 받아 "파일 업로드" 탭(HWP 직접 업로드는 LibreOffice 변환으로
        # 이미 지원됨)으로 올릴 수 있게 링크를 남겨둔다.
        unsupported_attachments: Optional[list[dict]] = None,
        _id: Optional[ObjectId] = None,

    ):
        self._id = _id
        self.project_id = project_id
        self.user_email = user_email
        self.original_filename = original_filename
        self.stored_filename = stored_filename
        self.file_path = file_path
        self.file_size = file_size
        self.mime_type = mime_type
        self.status = status
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.source_type = source_type
        self.document_type = document_type
        self.document_role = document_role
        self.parsed_text = parsed_text
        self.conversion_metadata = conversion_metadata
        self.unsupported_attachments = unsupported_attachments

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "user_email": self.user_email,
            "original_filename": self.original_filename,
            "stored_filename": self.stored_filename,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "document_type": self.document_type,
            "source_type": self.source_type,
            "document_role": self.document_role,
            "parsed_text": self.parsed_text,
            "conversion_metadata": self.conversion_metadata,
            "unsupported_attachments": self.unsupported_attachments,
        }