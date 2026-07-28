# RAG-007 — 외부 시장·정책 자료 검색

## 1. 목적

AI 위원의 역할(마케팅/기획/재무/기술 등)과 평가 기준에 맞춰, 시장성·정책성 평가에
필요한 **외부 공개 통계·시장·정책 자료**를 검색해 발행기관·원본 출처·자료 기준일과
함께 제공한다.

## 2. RAG-003·RAG-006과의 차이

| | RAG-003 | RAG-006 | RAG-007 |
|---|---|---|---|
| 검색 대상 | 현재 프로젝트 업로드 문서 | 공개 수상작·선정 사례·가이드 | 외부 공개 통계·시장·정책 자료 |
| Chroma 컬렉션 | `project_documents_kure_v1` | `similar_success_cases` | `external_market_policy_evidence` |
| 용도 | 평가의 **직접 근거** | 비교·개선을 위한 **참고 자료** | 시장/정책 환경을 보여주는 **외부 보조 근거** |
| 역할 반영 | RoleProfile로 검색어 확장 + 재정렬 | 없음 | 위원 역할별 검색어 확장 + role_score 랭킹 |

세 모듈은 서로 다른 Chroma 컬렉션을 쓰므로 데이터가 섞이지 않는다. RAG-007은
`ai.rag.role_retrieval`, `ai.rag.evidence_linking`, `ai.rag.evidence_sufficiency`,
`ai.rag.similar_cases`, `ai.rag.integration.meeting_evidence_adapter`의 기존 코드를
전혀 수정하지 않는다.

## 3. 외부자료 유형

```python
class ExternalEvidenceType(str, Enum):
    STATISTICS = "statistics"
    MARKET = "market"
    POLICY = "policy"
    PUBLIC_DATA = "public_data"
    GUIDELINE = "guideline"
    LAW = "law"
    RESEARCH_REPORT = "research_report"
```

기본 검색 대상은 정부·공공기관, 공공데이터 포털, 통계기관, 지자체, 법령·정책
공식 사이트, 공식 연구기관, 출처가 명확한 산업 보고서로 제한한다. 개인 블로그,
커뮤니티 게시글, 출처 불분명 기사, 광고성 페이지, 원본을 확인할 수 없는 재가공
자료는 색인 대상이 아니며, 설령 섞여 들어와도 `source_validator`가 출처 필드를
검증해 걸러낸다(6번 참고).

## 4. 외부자료 데이터 스키마

`ExternalEvidenceDocument` (`schemas.py`) — 색인 대상 청크 1건:

```python
from ai.rag.external_research import ExternalEvidenceDocument, ExternalEvidenceType

doc = ExternalEvidenceDocument(
    source_id="KOSIS-POPULATION-2025",
    document_id="DOC-KOSIS-001",
    chunk_id="CHUNK-KOSIS-001-01",
    title="연령별 인구 통계",
    evidence_type=ExternalEvidenceType.STATISTICS,
    publisher="통계청",
    source_url="https://kosis.kr/...",
    domain="public_service",
    evaluation_criteria=["시장성", "사회적 가치"],
    supported_roles=["marketing", "planning"],
    content="2025년 12월 기준 전국 인구는 ...",
    reference_date="2025-12-31",
    published_at="2026-02-01",
    region="대한민국",
    metric_name="총인구",
    metric_value=51700000,
    metric_unit="명",
)
```

`source_id, document_id, chunk_id, title, evidence_type, publisher, source_url,
domain, content`는 필수이며 비어 있으면 `ExternalResearchValidationError`가
발생한다. 날짜 필드(`reference_date`/`published_at`/`retrieved_at`)는 `YYYY-MM-DD`
형식이 아니면 거부된다 — 형식을 모르면 아예 값을 넣지 않아야 한다(임의 생성 금지).
`metric_value`는 원문에 실제 수치가 있을 때만 채운다; 불명확하면 `metric_name`/
`metric_value`/`metric_unit`을 모두 `None`으로 둔다(섹션 19의 정규화 원칙,
`normalizer.py`는 값의 "형태"만 정리할 뿐 값을 추론해서 채우지 않는다).

## 5. 사전 수집 데이터 색인 방법

기존 `KUREEmbedder`와 Chroma 인프라를 그대로 재사용한다.

```python
import chromadb
from ai.rag.embedding.kure_embedder import KUREEmbedder
from ai.rag.external_research import (
    ExternalEvidenceIndexingService,
    ExternalEvidenceRepository,
    ExternalResearchConfig,
)

config = ExternalResearchConfig()  # RAG_EXTERNAL_* 환경변수로 오버라이드 가능
client = chromadb.PersistentClient(path="./chroma_data")
embedder = KUREEmbedder()

repository = ExternalEvidenceRepository(
    client=client,
    collection_name=config.collection_name,
    embedding_model=embedder.model_name,
    embedding_dimension=embedder.embedding_dimension,
    embedding_version="embedding_v1",
)
indexing_service = ExternalEvidenceIndexingService(repository, embedder)

summary = indexing_service.index_evidence([doc], trace_id="batch-001")
print(summary.indexed_count, summary.skipped_count, summary.warnings)
```

색인 시점에 `source_validator.validate_source_metadata()`로 출처를 먼저 검증하고,
검증에 실패하거나 content가 비어 있으면 그 항목만 건너뛴다(배치 전체를 실패시키지
않음). 동일한 `source_id::document_id::chunk_id`는 upsert로 덮어써지므로 재색인해도
중복되지 않는다.

## 6. 역할별 검색 질의 생성 방식

`query_builder.py`의 `ROLE_QUERY_TERMS`가 역할별 확장 검색어를 관리한다(서비스
코드에 하드코딩하지 않음):

```python
ROLE_QUERY_TERMS = {
    "marketing": ["시장 규모", "시장 성장률", "목표 고객", "수요 조사", "경쟁 현황"],
    "planning": ["정책 방향", "지원사업 목적", "사회적 가치", "공공 수요", "제도 적합성"],
    "finance": ["산업 매출", "예산", "비용", "경제 통계", "재정 규모"],
    "technology": ["기술 동향", "도입률", "기술 가이드", "표준", "보안 기준"],
}
```

`build_external_research_query()`는 도메인·평가 기준·위원 역할·역할별 확장
검색어·검색 문맥·지역·자료 유형을 문자열로 조합하는 순수 함수다(같은 입력이면
항상 같은 질의, LLM 미사용). 등록되지 않은 `reviewer_role`이 들어와도 예외를
던지지 않고 확장 검색어 없이 그대로 진행한다.

## 7. DatasetProvider 사용 방법

```python
from ai.rag.external_research import DatasetProvider, ExternalResearchService

dataset_provider = DatasetProvider(repository, embedder, config=config)
service = ExternalResearchService(dataset_provider, config=config)
```

`DatasetProvider`는 `ExternalResearchProvider` Protocol(`name` 프로퍼티 +
`search(request, query_text)`)을 구현하며, 도메인 필터 결과가 0건이고
`config.domain_filter_fallback_to_all=True`면 도메인 필터 없이 전체 컬렉션에서
다시 검색하고 `last_search_used_domain_fallback=True`를 남긴다 — 서비스가 이를
읽어 warning 문구를 만든다.

## 8. PublicApiProvider 활성화 조건

백엔드는 NAVER API HUB 뉴스 검색을 실시간 보조 자료 provider로 연결할 수 있다.
`backend/.env`에 아래 값을 설정해야 하며, 키는 Git에 커밋하지 않는다.

```env
NAVER_CLIENT_ID=
NAVER_CLIENT_SECRET=
RAG_EXTERNAL_ENABLE_PUBLIC_API=true
```

호출 주소는 `https://naverapihub.apigw.ntruss.com/search/v1/news`, 인증 헤더는
`X-NCP-APIGW-API-KEY-ID`와 `X-NCP-APIGW-API-KEY`다. API가 반환하는 제목·설명의
HTML 태그를 제거하고 원문 URL의 호스트를 발행처로 보존하며, `pubDate`를 ISO 날짜로
정규화한다. API 장애나 timeout은 기존 사전 수집 데이터 검색을 중단시키지 않는다.

`RAG_EXTERNAL_ENABLE_PUBLIC_API=false`이거나 두 키 중 하나라도 없으면 실시간 검색은
연결하지 않고 기존 Chroma 데이터셋만 사용한다.

## 9. 서비스 단독 호출 예시 (LangGraph 없이)

```python
from ai.rag.external_research import ExternalResearchRequest, ExternalResearchService

response = service.search(
    ExternalResearchRequest(
        domain="공공 AI 서비스",
        evaluation_criteria=["시장성", "정책 적합성"],
        reviewer_role="planning",
        query_context="공공기관 사업계획서 자동 평가 서비스",
        region="대한민국",
        top_k=5,
        trace_id="meeting-42",
    )
)
```

`ai.meeting.graph`, LangGraph runtime, reviewer/chair 노드를 import하지 않으며,
`pytest ai/rag/tests/test_external_research_*.py`만으로 완전히 독립적으로 테스트할
수 있다.

## 10. 반환 JSON 예시

```json
{
  "results": [
    {
      "source_id": "KOSIS-POPULATION-2025",
      "document_id": "DOC-KOSIS-001",
      "chunk_id": "CHUNK-KOSIS-001-01",
      "title": "연령별 인구 통계",
      "evidence_type": "statistics",
      "publisher": "통계청",
      "source_url": "https://kosis.kr/...",
      "domain": "public_service",
      "supported_roles": ["marketing", "planning"],
      "matched_criteria": ["시장성"],
      "quote": "2025년 12월 기준 전국 인구는 ...",
      "reference_date": "2025-12-31",
      "published_at": "2026-02-01",
      "retrieved_at": null,
      "date_status": "current",
      "region": "대한민국",
      "period": null,
      "metric_name": "총인구",
      "metric_value": 51700000,
      "metric_unit": "명",
      "semantic_score": 0.71,
      "role_score": 1.0,
      "criteria_score": 0.5,
      "freshness_score": 0.94,
      "final_score": 0.7715,
      "retrieval_source": "dataset",
      "reference_only": true
    }
  ],
  "total_results": 1,
  "query_text": "위원 역할: planning\n도메인: 공공 AI 서비스\n...",
  "reviewer_role": "planning",
  "used_dataset_search": true,
  "used_public_api_search": false,
  "trace_id": "meeting-42",
  "warnings": [],
  "reference_only": true
}
```

## 11. 자료 기준일과 최신성 처리

`freshness.py`가 `reference_date`(우선) 또는 `published_at`을 오늘 날짜와 비교해
자료 유형별 기준(`config.py`의 `DEFAULT_FRESHNESS_THRESHOLD_DAYS`, 통계/시장/정책/
가이드/공공데이터/연구보고서마다 다름, `RAG_EXTERNAL_*` 환경변수는 없지만
`FreshnessConfig`를 직접 생성/주입해 오버라이드 가능)과 비교해 `current`/`aging`/
`stale`을 판정한다. 두 날짜가 모두 없으면 `unknown` + 고정된 낮은 점수
(`UNKNOWN_FRESHNESS_SCORE=0.2`)를 준다 — 날짜를 모르는 자료를 "최신"으로 취급하지
않는다. 날짜를 임의로 만들어내는 코드는 없다.

**법령(LAW) 관련 한계**: "법령자료는 최신 개정일 기준"이라는 요구사항을 엄밀히
만족하려면 법령 개정 이력 데이터가 필요하지만, 이 프로젝트는 그런 데이터를 갖고
있지 않다. 현재는 자료가 표시하는 `reference_date`(통상 최신 개정일로 간주)를
그대로 쓰고 POLICY와 동일한 임계값(2년)을 적용한다 — 실제 법령 개정 여부 확인은
미검증 항목이다(17번 참고).

## 12. 출처 검증 방식

`source_validator.validate_source_metadata()`가 `source_url`/`publisher`/
`document_id`/`chunk_id`/본문/`evidence_type`/날짜 형식을 확인해 `(verified, reasons)`를
반환한다. 색인 시점(`indexing_service`)과 검색 시점(provider가 candidate를 만들 때)
양쪽에서 동일한 함수로 검증하며, `verified_source=False`인 후보는
`ExternalResearchService`가 최종 결과에서 제외하고 건수를 warning에 남긴다.
`source_url`/`publisher`는 항상 색인 시점 metadata 또는 provider의 원시 응답 그대로이며
**LLM이 생성하지 않는다.**

## 13. LangGraph 없이 실행하는 방법

이 패키지의 어떤 파일도 `ai.meeting.graph`, LangGraph, reviewer/chair 노드를
import하지 않는다. 9번 예시처럼 `ExternalEvidenceRepository` + `DatasetProvider` +
임베더만 있으면 바로 실행/테스트할 수 있다.

## 14. 회의 파이프라인 전달 예시

```python
response = service.search(request)
external_research_data = response.model_dump()
```

이 모듈은 `ai/meeting/graph`를 수정하지 않으며, 통합은 호출하는 쪽의 책임이다.

## 15. RAG-005 근거 충분도와 분리되는 이유

RAG-007 결과는 외부 시장·정책 **환경**을 설명하는 보조 근거일 뿐, 현재 제출 문서에
그 내용이 실제로 인용·반영됐는지를 보장하지 않는다. 예를 들어 외부 통계에 시장
규모 자료가 있어도, 그것이 곧 현재 문서에 시장 규모 근거가 있다는 뜻은 아니다.
그래서 `ExternalEvidenceResult.reference_only`는 항상 `true`이고, 이 패키지는
`ai.rag.evidence_sufficiency`(RAG-005) 코드를 어디에서도 import하지 않는다 — RAG-007
결과가 존재한다고 해서 RAG-005의 `allow_numeric_score`나 근거 충족도 상태가 자동으로
바뀌는 경로 자체가 코드상 존재하지 않는다.

## 16. 실시간 검색의 비용·범위 제한

기본값은 `enable_public_api_search=False`다(`RAG_EXTERNAL_ENABLE_PUBLIC_API`).
활성화하려면 호출자가 실제 승인된 API를 감싼 `fetch` 콜러블을 주입해야 하며,
`PublicApiProviderConfig`로 `timeout_seconds`/`max_results`를 제한한다. 실패(timeout
포함)는 예외로 명확히 구분되고, `ExternalResearchService`가 이를 잡아 사전 수집
데이터 결과는 유지한 채 warning만 추가한다 — 실시간 API 장애가 전체 검색 실패로
번지지 않는다.

## 17. 현재 한계와 미검증 항목

- 실제 공개 통계·시장·정책 데이터셋이 없어 **진짜 데이터로 색인·검색을 수행한 적이
  없다.** 모든 테스트는 fixture로 만든 가짜 외부자료 데이터와, KURE-v1 대신
  `conftest.py`의 `fake_kure_embedder`/자체 fake 임베더를 사용했다.
- NAVER 뉴스 검색 결과는 기사 제목과 검색 요약문만 제공하므로 기사 전문을 직접
  확인한 자료와 동일하게 취급할 수 없다. 항상 보조 참고자료로만 사용한다.
- 법령(LAW) 최신성 판정은 "개정 이력"이 아니라 `reference_date` 단일 값 + POLICY와
  동일한 임계값으로 근사한다(11번 참고) — 실제 법령 개정 여부를 검증하지 않는다.
- `criteria_score`/`role_score`는 문자열 정확 일치(대소문자만 무시) 기반이라
  유의어나 상위 카테고리는 반영하지 못한다.
- LLM은 이 모듈의 어떤 경로에서도 호출되지 않는다(섹션 21 요구사항을 "아예
  사용하지 않음"으로 만족) — "외부자료 설명 생성" 등 선택적 LLM 활용은 구현하지
  않았다(회의 파이프라인 쪽에서 필요하면 `response.model_dump()` 결과를 받아 별도로
  처리해야 한다).
