# RAG 품질 오프라인 평가 도구

`ai/rag/evaluation/`(레거시 배치 위원회 domain/persona_id 전용, `EvaluationCase`가
`ai.rag.orchestration.role_mapping.resolve_role_id`로 엄격 검증)와는 별도 병렬 패키지다.
이 패키지는 대화형 아이디어 회의(ideation, `planning_expert`/`dev_expert`)의
`evidence_lookup` 실제 경로(`ai.rag.orchestration.ideation_evidence_service.
search_ideation_evidence`)와, 그 근거로 실제로 생성되는 페르소나 발언을 평가 대상으로
삼는다.

## 왜 category/source_org가 아니라 project_id/role_id인가

요청 스키마 예시(`category`, `source_org`)는 실제로 존재하는 검색 경로에 없는 필터다.
`contest_works`는 임베딩되지 않은 MongoDB 원본 스크랩 컬렉션일 뿐 벡터 검색 대상이
아니고, `category`/`source_org`가 Chroma 메타데이터 필터로 걸리는 함수는 코드 어디에도
없다. 실제로 존재하는 필터는 `project_id`(어느 프로젝트 문서를 검색할지)와
`role_id`(`planning`/`technology`, persona_id로부터 자동 결정)뿐이다. 자세한 조사
결과는 계획 문서(대화 세션의 plan mode 기록)를 참고.

## 4개 지표

- **Recall@K / Hit@K**: `ai/rag/evaluation/metrics.py`의 순수 함수를 그대로 재사용.
  `retrieval_eval.py`가 청크 단위 결과를 문서 단위로 접어(`_dedupe_by_document`)
  `gold_document_ids`와 비교한다.
- **Faithfulness**: `(supported + 0.5*partially_supported) / (supported+partially_supported
  +unsupported+contradicted)`. `non_factual`은 분모에서 제외. 분모가 0이면
  `faithfulness_score=None`(not_applicable)로 남긴다.
- **Hallucination Rate**: `(unsupported+contradicted) / 같은 분모`.
- **Persona Evidence Fit**: 발언별 0~4점, `normalized_score = score/4`. 전체 점수는
  발언 점수 합계 / (발언 수 × 4).

## 정식 점수 vs 참고 점수

`human_verified=true`인 케이스만 정식 macro average(`recall_at_k_macro` 등)에 들어간다.
`rag_eval_v1.jsonl`의 모든 케이스는 Claude가 실제 청크 텍스트를 읽고 만든 초안이라
**전부 human_verified=false**다 — 첫 실행에서는 `reference_recall_at_k_macro` 등
"참고 점수"만 채워지고 정식 점수는 `null`이 정상이다(요청 10번: "첫 실행에서는 통과/실패
보다 현재 점수를 기록하는 것을 우선"). `dataset.py::extract_review_sample()`로 15%
표본을 뽑아 사람이 검수한 뒤 해당 케이스의 `human_verified`를 `true`로 바꾸면 그때부터
정식 점수에 들어간다.

## 실행

```bash
# 검색만(LLM 호출 없음)
python -m ai.rag.evaluation.rag_quality.cli \
  --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl \
  --mode retrieval --top-k 5 --output reports/rag_eval

# 생성 품질까지(실제 OpenAI 호출, 비용 발생 — --limit로 케이스 수 제한 권장)
python -m ai.rag.evaluation.rag_quality.cli \
  --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl \
  --mode generation --limit 10 --output reports/rag_eval

# 둘 다
python -m ai.rag.evaluation.rag_quality.cli \
  --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl \
  --mode all --output reports/rag_eval
```

옵션: `--top-k`, `--limit`, `--case-id`, `--persona`, `--output`(디렉터리),
`--no-cache`, `--human-verified-only`, `--chroma-path`(기본값은 backend `.env`의
`CHROMA_PERSIST_DIR`).

## generation 모드가 실제로 하는 일

케이스마다 `ai.meeting.graph.start_ideation_conversation`을 **실제로**
`max_rounds=1`로 짧게 실행해 진짜 `planning_expert`/`dev_expert` 발언을 만든다(기존
회의/스트리밍/후보 재생성/expert_delegation 그래프는 전혀 수정하지 않고 호출만 함).
`evidence_lookup`은 운영과 동일한 `make_ideation_evidence_lookup()`을 그대로 쓰되,
얇은 래퍼로 각 호출의 반환값(그 발언이 실제로 받은 근거)을 기록해 둔다 — 이 기록이
Faithfulness/Persona Evidence Fit 판정의 유일한 근거 컨텍스트다(LLM이 발언 안에 스스로
써 넣은 `evidence` 필드는 신뢰하지 않는다).

## 결과 파일

`--output DIR`을 지정하면 `DIR/report.json`(전체 원시 결과), `DIR/report.csv`(케이스별
4지표+실패 사유), `DIR/report.md`(사람이 읽는 요약)를 만든다.

## 남은 한계

- `expected_evidence_topics`는 데이터셋에 저장되지만 자동 채점에 쓰이지 않는다(사람 검수
  참고용).
- `SimilarCaseSearchService`("수상작 사례" 검색, RAG-006)는 이 dev 환경에 색인된 데이터가
  0건이라 평가 대상에서 제외했다 — 실제 사례 문서가 색인되면 별도 하위 모듈로 확장 가능.
- `estimated_cost_usd`는 정확한 토큰 사용량이 아니라 호출 1건당 대략적인 상수를 곱한
  근사치다.

## 다중 문서 품질 게이트

`multi_document_quality.py`는 기존 Recall 평가에 Planner 선택과 claim-grounding 검수를
추가한다. `datasets/ideation_multi_document_template.json`을 복사해 서로 다른 실제 프로젝트
3~5개, 프로젝트당 최소 5개 쟁점을 채우고 사람이 청크·주장 유형·지원 여부를 확인한 뒤
`human_verified=true`로 바꾼다.

운영 trace에서 실제 검색·선택·연결 결과 15건을 검수 대기열로 만들 수 있다. 이 명령은
정답을 자동 판정하지 않으며 생성 결과는 항상 `human_verified=false`다.

```bash
python -m ai.rag.evaluation.rag_quality.build_ideation_annotation_queue \
  logs/app.log logs/app.log.2026-07-22 \
  --output ai/rag/evaluation/rag_quality/datasets/ideation_multi_document_annotation_queue.json \
  --project-count 3 \
  --cases-per-project 5
```

각 케이스에서 원문을 대조해 `gold_relevant_chunk_ids`, `planner_relevant_selected_chunk_ids`,
claim 본문·실제/기대 유형·지원 여부,
`reviewer_id`, `reviewed_at`, `reviewer_notes`를 채워야 한다. 필수 검수값이 비었거나 case ID가
중복된 상태에서 `human_verified=true`로만 바꾸면 품질 게이트가 해당 케이스를 거부한다.
`planner_relevant_selected_chunk_ids`에는 선택된 chunk 중 Planner가 뽑은 quote 자체가 현재
쟁점과 직접 관련된 것만 기록한다. 이 필드가 없으면 구버전 데이터 호환을 위해 chunk 단위
gold 교집합으로 계산하지만, 큰 평가표 chunk에서는 Planner precision이 과대평가될 수 있다.

```bash
python -m ai.rag.evaluation.rag_quality.multi_document_quality \
  reports/ideation_multi_document_observations.json \
  --output reports/ideation_multi_document_quality.json
```

기본 통과 기준은 검수된 3개 프로젝트·프로젝트당 5케이스·총 15케이스, Recall@5 0.80, Planner precision/coverage
0.80, citation precision 및 claim type accuracy 0.90, unsupported document fact 0.05 이하,
issue match 0.90, Planner fallback 0.10 이하이다. 미검수 케이스는 참고용으로만 세고 점수에서
제외한다.
