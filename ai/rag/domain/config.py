"""
Shared Domain Constants
========================
embedding/과 retrieval/이 함께 참조하는 상수만 둔다 (레이어 역전 방지: 다른 모듈이
domain을 참조하는 것은 되지만, domain이 embedding/retrieval을 참조하지는 않는다).
"""

# 2026-07-27: 기존 v1·v2 컬렉션의 영속 HNSW 세그먼트가 Windows 로컬 Chroma에서
# "Error loading hnsw index"로 손상되어 get/query/upsert 모두 불가능해졌다.
# Chroma에는 손상된 HNSW를 제자리 재구축하는 공개 API가 없으므로 기존 컬렉션을
# 삭제하지 않고 보존한 채 새 버전으로 전환한다. MongoDB 원본 문서는 같은
# document_id/project_id로 새 컬렉션에 재색인할 수 있다.
# 2026-07-27: v3 세그먼트도 같은 오류가 재발했다. 손상된 v3는 보존하고 v4로 격리한다.
# backend 테스트가 개발용 Chroma 경로를 열던 문제도 backend/tests/conftest.py에서 별도
# 테스트 경로로 분리해, 테스트 실행이 이 운영/개발 컬렉션에 접근하지 않도록 함께 막는다.
DEFAULT_COLLECTION_NAME: str = "project_documents_kure_v4"

# Chroma 컬렉션 이름 규칙 (chromadb 실제 검증 메시지 기준: 3~512자, [a-zA-Z0-9._-], 시작/끝은 영숫자)
COLLECTION_NAME_PATTERN: str = r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$"
