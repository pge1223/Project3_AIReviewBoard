from typing import Optional
from fastapi import APIRouter, HTTPException, Header
from datetime import datetime
from jose import jwt, JWTError
from app.schemas.project import ProjectCreateRequest, ProjectUpdateRequest, ProjectResponse
from app.repositories.project_repository import ProjectRepository
from app.models.project import ProjectModel
from app.config import settings
from starlette.concurrency import run_in_threadpool
from app.repositories.document_repository import DocumentRepository
from app.repositories.meeting_repository import MeetingRepository
from app.repositories.ideation_conversation_session_repository import (
    IdeationConversationSessionRepository,
)
from app.api.routes.documents import _get_indexing_service

router = APIRouter(prefix="/projects", tags=["projects"])
project_repo = ProjectRepository()

# 가은/Claude (2026-07-15): 비회원 로그인은 별도 인증 호출 없이 그냥 Authorization
# 헤더를 안 보낸다 — 헤더가 없으면 401 대신 고정 게스트 사용자로 통과시킨다.
GUEST_USER_EMAIL = "guest@local"


# JWT에서 현재 유저 이메일 추출 (없으면 게스트)
def get_current_user(authorization: Optional[str]) -> str:
    if not authorization:
        return GUEST_USER_EMAIL
    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")

# PRJ-001 프로젝트 생성
@router.post("/", response_model=ProjectResponse)
async def create_project(request: ProjectCreateRequest, authorization: Optional[str] = Header(None, alias="authorization")):
    user_email = get_current_user(authorization)

    project = ProjectModel(
        user_email=user_email,
        title=request.title,
        doc_type=request.doc_type,
        description=request.description,
        flow_mode=request.flow_mode,
    )
    result = await project_repo.create_project(project.to_dict())

    return ProjectResponse(
        id=str(result["_id"]),
        user_email=result["user_email"],
        title=result["title"],
        doc_type=result["doc_type"],
        description=result.get("description"),
        status=result["status"],
        created_at=result["created_at"],
        updated_at=result["updated_at"],
        flow_mode=result.get("flow_mode"),
        has_announcement_analysis=bool(result.get("announcement_analysis_cache")),
    )

# PRJ-002 프로젝트 목록 조회
@router.get("/", response_model=list[ProjectResponse])
async def get_projects(authorization: Optional[str] = Header(None, alias="authorization")):
    user_email = get_current_user(authorization)
    projects = await project_repo.find_by_user(user_email)

    return [
        ProjectResponse(
            id=str(p["_id"]),
            user_email=p["user_email"],
            title=p["title"],
            doc_type=p["doc_type"],
            description=p.get("description"),
            status=p["status"],
            created_at=p["created_at"],
            updated_at=p["updated_at"],
            flow_mode=p.get("flow_mode"),
            has_announcement_analysis=bool(p.get("announcement_analysis_cache")),
        )
        for p in projects
    ]

# PRJ-003 프로젝트 상세 조회
@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, authorization: Optional[str] = Header(None, alias="authorization")):
    user_email = get_current_user(authorization)
    project = await project_repo.find_by_id_and_user(project_id, user_email)

    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    return ProjectResponse(
        id=str(project["_id"]),
        user_email=project["user_email"],
        title=project["title"],
        doc_type=project["doc_type"],
        description=project.get("description"),
        status=project["status"],
        created_at=project["created_at"],
        updated_at=project["updated_at"],
        flow_mode=project.get("flow_mode"),
        has_announcement_analysis=bool(project.get("announcement_analysis_cache")),
    )

# PRJ-004 프로젝트 삭제
@router.delete("/{project_id}")
async def delete_project(project_id: str, authorization: Optional[str] = Header(None, alias="authorization")):
    user_email = get_current_user(authorization)
    project = await project_repo.find_by_id_and_user(project_id, user_email)

    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    # 1. Chroma 벡터 청크 삭제
    indexing_service = _get_indexing_service()
    await run_in_threadpool(indexing_service.delete_project, project_id)

    # 2. MongoDB 문서 삭제
    document_repo = DocumentRepository()
    documents = await document_repo.find_by_project_id(project_id)
    for doc in documents:
        await document_repo.delete_by_id(doc["_id"])

    # 3. MongoDB 회의 삭제
    meeting_repo = MeetingRepository()
    meetings = await meeting_repo.find_by_project_id(project_id)
    for meeting in meetings:
        await meeting_repo.delete_by_id(meeting["_id"])

    # 4. MongoDB 대화형 아이디어 회의 세션 삭제
    await IdeationConversationSessionRepository().delete_by_project_id(project_id)

    # 5. MongoDB 프로젝트 삭제
    await project_repo.delete_project(project_id)
    return {"message": "프로젝트가 삭제되었습니다"}


# 가은/Claude(2026-07-21): "작성 전" 흐름에서 주제를 확정할 때, EntryScreen에서 이미
# ensureProject()로 만들어둔 프로젝트(공고문이 붙어있을 수 있음)를 새로 하나 더 만들지
# 않고 제목·설명만 갱신하기 위한 범용 patch. 도메인 변경(DOM-002)과는 별개 목적이라
# 엔드포인트를 분리한다 — request.domain은 여기서 다루지 않는다.
@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    request: ProjectUpdateRequest,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    project = await project_repo.find_by_id_and_user(project_id, user_email)

    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    update_data = {}
    if request.title is not None:
        update_data["title"] = request.title
    if request.description is not None:
        update_data["description"] = request.description
    if request.flow_mode is not None:
        update_data["flow_mode"] = request.flow_mode
    if not update_data:
        raise HTTPException(status_code=400, detail="변경할 값이 없습니다")

    updated = await project_repo.update_project(project_id, update_data)

    return ProjectResponse(
        id=str(updated["_id"]),
        user_email=updated["user_email"],
        title=updated["title"],
        doc_type=updated["doc_type"],
        description=updated.get("description"),
        status=updated["status"],
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
        flow_mode=updated.get("flow_mode"),
    )


# DOM-002: 도메인 수동 변경
@router.patch("/{project_id}/domain")
async def update_project_domain(
    project_id: str,
    request: ProjectUpdateRequest,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    user_email = get_current_user(authorization)
    project = await project_repo.find_by_id_and_user(project_id, user_email)

    if not project:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    ALLOWED_DOMAINS = {"startup", "government", "competition"}
    if request.domain not in ALLOWED_DOMAINS:
        raise HTTPException(status_code=400, detail=f"허용되지 않는 도메인입니다. 허용: {ALLOWED_DOMAINS}")

    updated = await project_repo.update_project(project_id, {"domain": request.domain})

    return ProjectResponse(
        id=str(updated["_id"]),
        user_email=updated["user_email"],
        title=updated["title"],
        doc_type=updated["doc_type"],
        description=updated.get("description"),
        status=updated["status"],
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
        flow_mode=updated.get("flow_mode"),
    )
