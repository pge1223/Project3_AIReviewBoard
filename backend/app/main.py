import sys
from pathlib import Path

# backend/는 보통 이 디렉터리를 cwd로 uvicorn이 실행되어 레포 루트가 sys.path에 없다.
# ai.rag.documents 라우터가 ai.rag.* 를 import하므로 여기서 레포 루트를 추가해준다
# (ai/rag/tests/conftest.py의 동일 패턴 참고).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db.mongodb import connect_db, close_db
from app.api.routes.auth import router as auth_router
from app.api.routes.projects import router as project_router
from app.api.routes.documents import router as document_router
from app.api.routes.meetings import router as meeting_router
from app.api.routes.comparison import router as comparison_router  # 윤한/Claude(2026-07-21): RPT-004 버전 비교 (app/api/routes/comparison.py)
from app.api.routes.profile import router as profile_router  # 윤한/Claude(2026-07-21): 마이페이지 프로필 CRUD (app/api/routes/profile.py)
from app.api.routes.media import router as media_router  # 재인/Claude (2026-07-16): 위원 발언 영상 스트리밍 중계 (app/api/routes/media.py)
from app.api.routes.workbench import router as workbench_router  # 재인/Claude (2026-07-21): "AI 피드백" 워크벤치 인용 매칭 (app/api/routes/workbench.py)
from app.repositories.ideation_conversation_session_repository import (
    IdeationConversationSessionRepository,
)
from app.core.logger import logger
from ai.rag.converters.diagnostics import HwpDiagnosticsResult, log_hwp_diagnostics, run_hwp_diagnostics

app = FastAPI(
    title="AI Review Board API",
    description="RAG 기반 AI 심사위원회 시스템",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(project_router)
app.include_router(document_router)
app.include_router(meeting_router)
app.include_router(comparison_router)  # 윤한/Claude(2026-07-21): GET /projects/{project_id}/comparison
app.include_router(media_router)  # 재인/Claude (2026-07-16): /media/available-speakers, /media/stream
app.include_router(workbench_router)  # 재인/Claude (2026-07-21): /workbench/{project_id}/quotes
app.include_router(profile_router)  # 윤한/Claude(2026-07-21): GET·PUT /users/me/profile

# 용준/Claude(2026-07-20): 개발용 아이디어 발전 회의 프리뷰(POST /ideation-preview).
# ENABLE_IDEATION_PREVIEW가 기본값(False)이면 라우터 자체를 등록하지 않는다 — 비활성화
# 상태에서는 엔드포인트가 앱에 존재하지도 않게 해, 운영 환경에서 실수로 노출되지 않는다.
if settings.ENABLE_IDEATION_PREVIEW:
    from app.api.routes.ideation_preview import router as ideation_preview_router

    app.include_router(ideation_preview_router)
    logger.info("[ideation-preview] 개발용 프리뷰 라우터 등록됨 (ENABLE_IDEATION_PREVIEW=true)")

    # 용준/Claude(2026-07-20): 개발용 대화형 아이디어 발전 회의 프리뷰
    # (POST /ideation-conversation/start 등). 같은 플래그를 공유한다 — 둘 다 같은
    # "개발용 ideation 프리뷰" 묶음이라 별도 플래그를 새로 만들지 않았다.
    from app.api.routes.ideation_conversation_preview import router as ideation_conversation_router

    app.include_router(ideation_conversation_router)
    logger.info("[ideation-conversation] 개발용 대화형 프리뷰 라우터 등록됨 (ENABLE_IDEATION_PREVIEW=true)")

# HWP 업로드를 실제로 시도하기 전에 서버 시작 시 한 번만 변환 가능 상태를 점검해
# app.state에 캐싱한다 — soffice/unopkg/java 서브프로세스를 /health 요청마다 반복
# 실행하지 않기 위함(ai/rag/converters/diagnostics.py). 진단 자체가 예외를 던지지
# 않도록 설계돼 있지만(run_hwp_diagnostics 내부에서 흡수), 여기서도 한 번 더 감싸서
# 진단 실패가 서버 기동 실패로 이어지지 않게 한다.
#
# fail-safe 정책: run_hwp_diagnostics() 호출 자체가 여기서 또 실패하면(즉 그 함수
# 내부의 방어조차 뚫렸다면) config.enabled를 확인할 방법이 없다 — 이럴 때
# enabled=False로 두면 /health가 "의도적 비활성화"로 오인해 status="ok"를 반환해
# 실제 장애를 정상처럼 위장하게 된다. 그래서 enabled=True로 fail-safe 처리해
# status="degraded"가 뜨도록 한다(ai/rag/converters/diagnostics.py의
# _diagnostics_failed_result와 동일한 정책).
@app.on_event("startup")
async def startup():
    await connect_db()
    if settings.ENABLE_IDEATION_PREVIEW:
        await IdeationConversationSessionRepository().ensure_indexes()

    try:
        hwp_diagnostics = await run_in_threadpool(run_hwp_diagnostics)
    except Exception:
        logger.exception("[HWP_CONVERTER_DIAGNOSTICS_ERROR] 진단 실행 자체에 실패했습니다")
        hwp_diagnostics = HwpDiagnosticsResult(
            enabled=True,
            available=False,
            libreoffice=False,
            h2orestart=False,
            java=False,
            temp_dir_writable=False,
            reason="HWP diagnostics failed to run (see server logs)",
        )
    app.state.hwp_diagnostics = hwp_diagnostics
    log_hwp_diagnostics(hwp_diagnostics)

    logger.info("AI Review Board API 서버 시작")


@app.on_event("shutdown")
async def shutdown():
    await close_db()
    logger.info("AI Review Board API 서버 종료")


@app.get("/")
async def root():
    return {"message": "AI Review Board API is running"}


# 테스트 클라이언트 등 startup 이벤트가 실행되지 않은 상태에서 /health가 불려도
# 죽지 않도록 하는 안전한 기본값 — "진단이 아직 실행되지 않음"을 뜻하며, 실제 HWP
# 설정과 무관하므로 서버 status는 degraded로 만들지 않는다(ok로 둔다).
_HWP_DIAGNOSTICS_NOT_RUN = HwpDiagnosticsResult(
    enabled=False,
    available=False,
    libreoffice=False,
    h2orestart=False,
    java=False,
    temp_dir_writable=False,
    reason="HWP diagnostics have not run yet",
)


@app.get("/health")
async def health_check():
    hwp_diagnostics: HwpDiagnosticsResult = getattr(app.state, "hwp_diagnostics", _HWP_DIAGNOSTICS_NOT_RUN)

    # 의도적으로 비활성화된 상태(enabled=False)는 degraded로 취급하지 않는다 — 활성화됐는데
    # 필수 항목이 빠졌을 때만(enabled=True and not available) 서버 전체 status를 낮춘다.
    status_value = "degraded" if hwp_diagnostics.enabled and not hwp_diagnostics.available else "ok"

    return {
        "status": status_value,
        "capabilities": {
            "hwp_conversion": hwp_diagnostics.model_dump(),
        },
    }
