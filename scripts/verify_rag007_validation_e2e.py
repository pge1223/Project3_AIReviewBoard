"""Run planning/dev idea_validation with real KURE, Chroma RAG-007, and the configured LLM.

용준/Claude(2026-07-30, 요청: RAG-007 색인 청크 품질 정제) — staging 컬렉션 검증을
위해 --collection/--output을 인자화했다. 기본값은 기존과 동일(운영 컬렉션, 기존
출력 경로)이라 인자 없이 실행하면 하위 호환이다."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
MEETING = ROOT / "ai" / "meeting"
for path in (ROOT, BACKEND, MEETING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai.rag.domain.config import DEFAULT_COLLECTION_NAME
from ai.rag.embedding import KUREEmbedder
from ai.rag.embedding.config import EMBEDDING_VERSION
from ai.rag.evidence_linking.claim_grounding import ground_claims
from ai.rag.external_research import (
    DatasetProvider,
    ExternalEvidenceRepository,
    ExternalResearchConfig,
    ExternalResearchService,
)
from ai.rag.orchestration.ideation_evidence_service import make_ideation_evidence_lookup
from ai.rag.orchestration.ideation_external_evidence_service import (
    make_ideation_external_evidence_lookup,
)
from ai.rag.orchestration.ideation_target_indexing_service import (
    index_selected_candidate_as_target,
)
from ai.rag.retrieval.chroma_store import ChromaVectorStore
from ai.rag.retrieval.service import RAGIndexingService
from ai.rag.role_retrieval.service import RoleAwareRetrievalService
from app.config import settings
from graph.ideation_conv_problem import (
    make_planning_validation_node,
    make_technical_validation_node,
)
from graph.ideation_conv_state import initial_conv_state
from graph.llm import make_openai_llm_call

OUTPUT_DIR = ROOT / "reports" / "rag007_e2e_20260730"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation_e2e.json"
TEMP_CHROMA = OUTPUT_DIR / "validation_chroma"
PRODUCTION_COLLECTION_NAME = "external_market_policy_evidence"
SOURCE_PROJECT_ID = "6a68b9b9a84aaa49019a34bd"
TEST_PROJECT_ID = "rag007-e2e-validation"
SESSION_ID = "RAG007-E2E-20260730"

IDEA = {
    "candidate_id": "public-ai-accessibility-assistant",
    "title": "고령자·장애인을 위한 공공 AI 민원 안내 서비스",
    "problem": "고령자와 장애인이 복잡한 공공서비스 정보를 찾고 신청하는 과정에서 접근성 장벽을 겪는다.",
    "target_user": "디지털 접근에 어려움을 겪는 고령자와 장애인",
    "solution": "접근성 지원 대화형 안내와 개인정보 최소수집 원칙을 적용하고, 제한된 민원 분야에서 실증한다.",
    "main_features": [
        "쉬운 문장과 단계별 민원 안내",
        "화면 읽기 및 키보드 접근성 지원",
        "개인정보 최소수집과 상담 기록 통제",
    ],
    "required_data": ["공식 민원 절차", "접근성 사용자 테스트 결과"],
    "technical_approach": "RAG 기반 공식 정보 검색, 접근권한 통제, 응답 근거 표시",
    "mvp_scope": "민원 1개 분야에서 안내 정확성·접근성·개인정보 보호를 검증",
}

NOTICE = {
    "competition_name": "2026 공공기관 AI 혁신 챌린지",
    "purpose": "공공기관의 AI 활용 혁신과 국민 체감형 서비스 발굴",
    "criteria": "실증 가능성, 타 기관 확산 가능성, 대국민 체감 효과와 안전한 AI 활용",
}

ROLE_KEYWORDS = {
    "planning_expert": ["접근성", "사용자", "공공서비스", "실증", "확산", "개인정보"],
    "dev_expert": ["AI", "데이터", "보안", "개인정보", "안전성", "구현"],
}


def _copy_project_criteria(source_client, target_collection) -> int:
    source = source_client.get_collection(DEFAULT_COLLECTION_NAME)
    raw = source.get(
        where={"project_id": SOURCE_PROJECT_ID},
        include=["documents", "metadatas", "embeddings"],
    )
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    for record_id, document, metadata, embedding in zip(
        raw["ids"], raw["documents"], raw["metadatas"], raw["embeddings"]
    ):
        metadata = dict(metadata or {})
        if metadata.get("document_role") != "criteria":
            continue
        ids.append(f"e2e::{record_id}")
        documents.append(document or "")
        metadatas.append({**metadata, "project_id": TEST_PROJECT_ID})
        embeddings.append(embedding)
    if ids:
        target_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
    return len(ids)


def _apply_update(state: dict, update: dict) -> dict:
    merged = dict(state)
    for key, value in update.items():
        if key in {"messages", "idea_evolution"}:
            merged[key] = list(state.get(key) or []) + list(value or [])
        else:
            merged[key] = value
    return merged


def _message_report(message: dict, retrieved_external: list[dict], grounding: dict) -> dict:
    linked = set(message.get("linked_external_evidence_refs") or [])
    evidence = message.get("evidence") or []
    ui_external = [
        {
            "title": item.get("title") or item.get("document_title"),
            "publisher": item.get("publisher") or item.get("organization"),
            "published_at": item.get("published_at") or item.get("reference_date"),
            "source_url": item.get("source_url"),
            "chunk_id": item.get("chunk_id"),
        }
        for item in evidence
        if item.get("chunk_id") in linked
    ]
    return {
        "retrieved_external_evidence": retrieved_external,
        "claims": message.get("claims") or [],
        "claim_evidence_links": grounding.get("claim_evidence_links") or [],
        "linked_external_evidence_refs": message.get("linked_external_evidence_refs") or [],
        "reviewed_target_refs": message.get("reviewed_target_refs") or [],
        "linked_criteria_refs": message.get("linked_criteria_refs") or [],
        "expert_judgment_without_external_evidence": (
            message.get("expert_judgment_without_external_evidence") or []
        ),
        "final_utterance": message.get("content"),
        "ui_external_evidence": ui_external,
        "ui_target_titles": [
            item.get("document_name") or item.get("document_title") or item.get("title")
            for item in evidence
            if item.get("chunk_id") in set(message.get("reviewed_target_refs") or [])
        ],
        "evidence_status": message.get("evidence_status"),
        "evidence_funnel": message.get("evidence_funnel") or {},
    }


def parse_args() -> argparse.Namespace:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=PRODUCTION_COLLECTION_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not settings.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY가 없어 실제 validation LLM 호출을 실행할 수 없습니다.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embedder = KUREEmbedder()
    active_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    temp_client = chromadb.PersistentClient(path=str(TEMP_CHROMA))
    temp_store = ChromaVectorStore(
        client=temp_client,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.embedding_dimension,
        embedding_version=EMBEDDING_VERSION,
    )
    copied_criteria_count = _copy_project_criteria(active_client, temp_store._collection)
    indexing_service = RAGIndexingService(embedder, temp_store)
    indexed_target = index_selected_candidate_as_target(
        indexing_service=indexing_service,
        project_id=TEST_PROJECT_ID,
        session_id=SESSION_ID,
        candidate_id=IDEA["candidate_id"],
        candidate=IDEA,
    )
    role_service = RoleAwareRetrievalService(retrieval_service=indexing_service)
    evidence_lookup = make_ideation_evidence_lookup(
        project_id=TEST_PROJECT_ID,
        role_retrieval_service=role_service,
        top_k=5,
        session_id=SESSION_ID,
        selected_candidate_document_id=indexed_target.document_id,
    )
    evidence_lookup.criteria_chunk_count = copied_criteria_count

    external_config = ExternalResearchConfig(
        collection_name=args.collection,
        enable_public_api_search=False,
        min_similarity_score=0.45,
    )
    external_repo = ExternalEvidenceRepository(
        client=active_client,
        collection_name=external_config.collection_name,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.embedding_dimension,
        embedding_version="embedding_v1",
    )
    external_service = ExternalResearchService(
        DatasetProvider(external_repo, embedder, config=external_config),
        config=external_config,
        embedder=embedder,
    )
    base_external_lookup = make_ideation_external_evidence_lookup(
        external_service, top_k=3
    )
    retrieved_by_persona: dict[str, list[dict]] = {}

    def external_lookup(persona_id: str, query: str) -> dict:
        result = base_external_lookup(persona_id, query)
        retrieved_by_persona[persona_id] = list(result.get("external_evidence") or [])
        return result

    grounding_by_persona: dict[str, dict] = {}

    def grounder(persona_id: str, claims, evidence: list[dict]) -> dict:
        result = ground_claims(
            claims,
            evidence,
            role_keywords=ROLE_KEYWORDS.get(persona_id),
        )
        grounding_by_persona[persona_id] = result
        return result

    llm_call = make_openai_llm_call(
        settings.reviewer_model(), api_key=settings.OPENAI_API_KEY
    )
    state = initial_conv_state(SESSION_ID, NOTICE, {"description": ""})
    state = {
        **state,
        "phase": "idea_validation",
        "project_id": TEST_PROJECT_ID,
        "selected_idea_document_id": indexed_target.document_id,
        "provisional_idea": IDEA,
    }
    planning_node = make_planning_validation_node(
        llm_call, evidence_lookup, external_lookup, ground_claims=grounder
    )
    state = _apply_update(state, planning_node(state))
    technical_node = make_technical_validation_node(
        llm_call, evidence_lookup, external_lookup, ground_claims=grounder
    )
    state = _apply_update(state, technical_node(state))

    planning_message = next(
        item for item in state["messages"] if item.get("speaker_id") == "planning_expert"
    )
    technical_message = next(
        item for item in state["messages"] if item.get("speaker_id") == "dev_expert"
    )
    planning_report = _message_report(
        planning_message,
        retrieved_by_persona.get("planning_expert", []),
        grounding_by_persona.get("planning_expert", {}),
    )
    technical_report = _message_report(
        technical_message,
        retrieved_by_persona.get("dev_expert", []),
        grounding_by_persona.get("dev_expert", {}),
    )
    reports = [planning_report, technical_report]

    linked_items = [
        item
        for report in reports
        for item in report["ui_external_evidence"]
    ]
    dpg_linked = any(
        item.get("chunk_id")
        and any(
            evidence.get("chunk_id") == item["chunk_id"]
            and (
                evidence.get("summary_only")
                or evidence.get("source_type") == "official_page_summary"
                or evidence.get("allow_grounded_claim") is False
            )
            for report in reports
            for evidence in report["retrieved_external_evidence"]
        )
        for item in linked_items
    )
    raw_pattern = re.compile(r"<\s*작성\s*요령\s*>|\.hwp[x]?|^\s*[○※]\s*$", re.MULTILINE)
    success = {
        "rag007_cited_by_at_least_one_expert": bool(linked_items),
        "ui_contains_only_cited_external_evidence": all(
            len(report["ui_external_evidence"])
            == len(set(report["linked_external_evidence_refs"]))
            for report in reports
        ),
        "target_is_reviewed_not_external": all(
            not (
                set(report["reviewed_target_refs"])
                & set(report["linked_external_evidence_refs"])
            )
            for report in reports
        ),
        "dpg_summary_only_not_linked": not dpg_linked,
        "raw_chunk_not_exposed_in_utterance": not any(
            raw_pattern.search(report["final_utterance"] or "") for report in reports
        ),
    }
    report = {
        "model": settings.reviewer_model(),
        "source_project_id": SOURCE_PROJECT_ID,
        "test_project_id": TEST_PROJECT_ID,
        "copied_criteria_count": copied_criteria_count,
        "target_document_id": indexed_target.document_id,
        "target_chunk_count": indexed_target.chunk_count,
        "planning": planning_report,
        "technical": technical_report,
        "success_conditions": success,
        "all_success": all(success.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
