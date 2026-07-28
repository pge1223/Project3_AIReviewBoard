import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jose import jwt

# backend/ 를 sys.path에 추가해 `from app.main import app`이 되도록 한다
# (app.main 자신이 레포 루트를 추가하는 로직은 그 이후에 실행됨).
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# app.main import는 documents/meetings 라우트를 통해 PersistentClient를 즉시 만들 수 있다.
# 테스트가 실행 중인 개발 서버와 같은 ./chroma_db를 열면 별도 프로세스가 같은 HNSW 파일에
# 접근해 인덱스 손상 위험이 생기므로, app.config.Settings가 로드되기 전에 프로세스별 테스트
# 전용 경로로 강제 격리한다.
_TEST_CHROMA_DIR = Path(tempfile.gettempdir()) / f"ai-review-board-pytest-chroma-{os.getpid()}"
os.environ["CHROMA_PERSIST_DIR"] = str(_TEST_CHROMA_DIR)

from app.main import app
from app.config import settings


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_header() -> dict:
    token = jwt.encode({"sub": "tester@example.com"}, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return {"authorization": f"Bearer {token}"}
