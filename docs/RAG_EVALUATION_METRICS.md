# RAG 평가 지표 및 측정 가이드

## 1. 평가 목적

이 문서는 AI 아이디어 회의에서 기획위원과 개발위원이 사용하는 RAG 파이프라인의 품질을
반복 측정하기 위한 기준을 정의한다. 모델의 학습 성능이 아니라 다음 파이프라인 결과를
평가한다.

`사용자 질문 → 문서 검색 → 역할별 근거 선택 → 위원 발언 생성 → 근거 연결`

평가는 검색 품질과 생성 품질을 분리한다. 검색 결과가 나쁘면 생성 모델을 바꿔도 답변의
근거가 좋아지지 않으며, 검색 결과가 좋아도 생성 발언이 원문을 왜곡할 수 있기 때문이다.

## 2. 핵심 평가 지표

| 구분 | 지표 | 의미 | 현재 측정 | 권장 목표 |
|---|---|---|---|---:|
| 검색 | Recall@5 | 필요한 정답 문서를 상위 5개 안에서 얼마나 찾았는가 | 지원 | 0.85 이상 |
| 검색 | Hit@5 | 정답 문서를 하나라도 상위 5개 안에서 찾았는가 | 지원 | 0.90 이상 |
| 검색 | MRR@5 | 첫 번째 정답 문서가 검색 결과의 몇 위에 나타나는가 | 지원 | 0.80 이상 |
| 검색 | Context Precision | 검색된 근거 중 실제 질문과 관련 있는 근거의 비율 | 다중 문서 게이트에서 지원 | 0.80 이상 |
| 검색 | Context Recall | 정답에 필요한 근거를 빠짐없이 검색했는가 | Recall@K로 대체 측정 | 0.85 이상 |
| 생성 | Faithfulness | 위원 발언의 사실 주장이 검색 근거로 뒷받침되는가 | 지원 | 0.90 이상 |
| 생성 | Hallucination Rate | 근거가 없거나 근거와 모순되는 주장의 비율 | 지원 | 0.05 이하 |
| 생성 | Answer Relevance | 발언이 질문과 현재 회의 쟁점에 직접 답하는가 | 추가 권장 | 0.80 이상 |
| 역할 | Persona Evidence Fit | 기획·개발위원이 각 역할에 맞는 근거를 사용했는가 | 지원 | 0.80 이상 |
| 연결 | Citation Precision | 연결된 문서·청크가 실제 주장을 지원하는가 | 다중 문서 게이트에서 지원 | 0.90 이상 |
| 운영 | Retrieval Failure Rate | 근거가 있어야 하는데 검색 결과가 0건인 비율 | 지원 | 0.05 이하 |
| 운영 | 평균 검색 시간 | 질문 1건의 근거 검색 소요 시간 | 지원 | 환경별 기준선 대비 관리 |

## 3. 지표 계산 방법

### 3.1 Recall@K

정답 문서 집합 중 상위 K개 검색 결과에 포함된 문서의 비율이다.

```text
Recall@K = |정답 문서 ∩ 상위 K개 검색 문서| / |정답 문서|
```

한 질문에 필요한 문서가 2개이고 상위 5개 결과에서 1개만 찾았다면 Recall@5는 `0.5`다.
현재 구현은 같은 문서에서 검색된 여러 청크를 문서 단위로 중복 제거한 후 계산한다.

### 3.2 Hit@K

상위 K개 결과 안에 정답 문서가 하나라도 있으면 `1`, 없으면 `0`이다.

```text
Hit@K = 1 if 정답 문서 ∩ 상위 K개 검색 문서가 존재 else 0
```

Hit@K는 검색 성공 여부를 빠르게 확인하기 좋지만, 필요한 여러 문서를 모두 찾았는지는
보여주지 않으므로 Recall@K와 함께 사용한다.

### 3.3 MRR@K

첫 번째 정답 문서가 나타난 순위의 역수를 계산한다. 1위면 `1.0`, 2위면 `0.5`,
5위면 `0.2`, 상위 K개 안에 없으면 `0`이다. Recall@K가 같더라도 사용하기 좋은 근거가
앞쪽에 배치되는지를 구분한다.

```text
Reciprocal Rank = 1 / 첫 번째 정답 문서 순위
MRR@K = 전체 평가 케이스 Reciprocal Rank의 평균
```

### 3.4 Context Precision

검색 또는 Planner가 선택한 근거 중 실제 질문과 직접 관련된 근거의 비율이다.

```text
Context Precision = 관련 근거 수 / 선택된 전체 근거 수
```

점수가 낮으면 검색 결과에 공고문 작성 요령, 무관한 신청 양식, 다른 평가 부문의 내용이
섞이고 있다는 뜻이다.

### 3.5 Faithfulness

위원 발언을 사실 주장 단위로 나눈 후 각 주장을 다음 다섯 종류로 판정한다.

- `supported`: 근거가 직접 지원
- `partially_supported`: 일부만 지원
- `unsupported`: 근거에서 확인할 수 없음
- `contradicted`: 근거와 모순
- `non_factual`: 의견·제안처럼 사실 검증 대상이 아님

```text
Faithfulness =
  (supported + 0.5 × partially_supported)
  / (supported + partially_supported + unsupported + contradicted)
```

`non_factual`은 분모에서 제외한다. 검증할 사실 주장이 하나도 없으면 점수를 `N/A`로 둔다.

### 3.6 Hallucination Rate

검색 근거로 확인되지 않거나 근거와 모순되는 주장의 비율이다.

```text
Hallucination Rate =
  (unsupported + contradicted)
  / (supported + partially_supported + unsupported + contradicted)
```

파일명이나 HWP/HWPX 원문이 말풍선에 그대로 출력되는 문제는 환각과 별개다. 이는
`message`와 `evidence` 분리 여부를 확인하는 회귀 테스트로 따로 관리한다.

### 3.7 Persona Evidence Fit

위원 발언이 역할에 맞는 근거를 사용했는지 LLM Judge가 0~4점으로 평가한다.

- 기획위원: 정책 적합성, 사용자 문제, 확산 가능성, 공공 가치, 최근 사회·시장 동향
- 개발위원: 구현 가능성, 데이터·API, 보안, 성능, 운영 위험, 최신 기술 동향

```text
정규화 점수 = 역할 적합도 점수 / 4
```

### 3.8 Answer Relevance

발언이 검색 문서의 내용을 반복하는 데 그치지 않고 사용자의 질문과 현재 회의 쟁점에
직접 답하는지 평가한다. 현재 자동 리포트의 정식 지표에는 포함되지 않으므로 다음
루브릭으로 추가하는 것을 권장한다.

- `0`: 질문과 무관
- `1`: 관련 단어만 언급
- `2`: 일부 답변하지만 핵심 누락
- `3`: 핵심에 직접 답변
- `4`: 직접 답변하고 다음 판단·행동까지 제시

## 4. 평가 데이터셋 기준

평가 케이스 한 건에는 최소한 다음 값이 필요하다.

```json
{
  "id": "contest_001",
  "query": "실증·PoC 부문의 심사 항목과 배점은 무엇인가요?",
  "persona_id": "planning_expert",
  "filters": {
    "project_id": "실제 프로젝트 ID",
    "phase": "idea_validation"
  },
  "gold_document_ids": ["정답 문서 ID"],
  "expected_evidence_topics": ["평가 항목", "배점"],
  "forbidden_claims": [],
  "expect_no_evidence": false,
  "human_verified": true
}
```

자동 또는 AI로 만든 정답은 `human_verified=false`로 두고 참고 점수에만 사용한다. 사람이
원문과 정답 문서를 확인한 케이스만 `true`로 변경하여 정식 평균에 포함한다.

권장 구성은 다음과 같다.

- 최소 베이스라인: 10건
- 비교 가능한 1차 평가: 30건 이상
- 기획위원과 개발위원 케이스를 같은 비율로 구성
- 공고문, 평가 기준, 신청 양식, 최신 뉴스 근거를 분리하여 포함
- 정답 근거가 없는 질문도 포함하여 불필요한 근거 생성 여부 확인

## 5. 실행 방법

검색 평가만 실행하면 LLM 비용이 발생하지 않는다.

```bash
python -m ai.rag.evaluation.rag_quality.cli \
  --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl \
  --mode retrieval \
  --top-k 5 \
  --output reports/rag_eval
```

실제 위원 발언의 Faithfulness와 역할 적합도까지 평가할 때는 OpenAI 호출 비용이
발생하므로 처음에는 `--limit 10`을 권장한다.

```bash
python -m ai.rag.evaluation.rag_quality.cli \
  --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl \
  --mode all \
  --top-k 5 \
  --limit 10 \
  --output reports/rag_eval
```

결과는 다음 세 파일로 생성된다.

- `report.json`: 전체 원시 결과와 케이스별 판정
- `report.csv`: 케이스별 지표 비교
- `report.md`: 사람이 읽는 요약과 실패 케이스

> **Windows 로컬 주의:** 백엔드와 평가 CLI가 같은 `chroma_db`를 동시에 열면 HNSW
> 인덱스가 손상될 수 있다. 평가할 때는 백엔드를 종료하거나 Chroma 폴더를 복사한
> 평가 전용 경로를 `--chroma-path`로 지정한다.

## 6. 결과 해석 및 개선 순서

| 증상 | 우선 확인할 항목 | 개선 방향 |
|---|---|---|
| Recall@5가 낮음 | 청킹, 검색 쿼리, 문서 유형, top_k | 청킹·쿼리 확장·문서 유형 필터 개선 |
| Hit@5는 높지만 Precision이 낮음 | 중복 청크와 무관 문서 혼입 | 문서 중복 제거·rerank·단계별 quota 조정 |
| Faithfulness가 낮음 | 검색 근거와 발언의 주장 연결 | evidence만 프롬프트에 주입하고 근거 없는 주장 제한 |
| Hallucination Rate가 높음 | 빈 검색 결과의 LLM 폴백 | 근거 없음 안내, 주장 생성 제한, citation 검증 |
| Persona Fit이 낮음 | 기획·개발 역할 매핑과 target | 역할별 검색 쿼리·가중치·근거 유형 분리 |
| 최신성이 낮음 | 외부 뉴스 검색과 날짜 메타데이터 | 최신성 가중치, 중복 기사 제거, 게시일 검증 |
| 검색 시간이 증가함 | 후보 청크 수와 rerank 범위 | candidate_k, 배치 크기, 캐시 조정 |

## 7. 품질 게이트

초기 권장 통과 기준은 다음과 같다.

```text
Recall@5                  >= 0.85
Hit@5                     >= 0.90
MRR@5                     >= 0.80
Context/Planner Precision >= 0.80
Faithfulness              >= 0.90
Hallucination Rate        <= 0.05
Persona Evidence Fit      >= 0.80
Citation Precision        >= 0.90
Retrieval Failure Rate    <= 0.05
```

점수는 한 번의 최고값보다 같은 검수 데이터셋에서 이전 버전과 비교한 변화량을 중요하게
본다. 청킹, 임베딩 모델, Chroma 컬렉션, 역할별 가중치가 바뀌면 설정과 컬렉션 이름을
리포트에 반드시 기록한다.

## 8. 현재 프로젝트 참고 상태

- 자동 평가 구현: `ai/rag/evaluation/rag_quality/`
- 기준값 설정: `ai/rag/evaluation/rag_quality/thresholds.json`
- 기본 데이터셋: `ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl`
- 이전 10건 참고 측정: Recall@5 `1.0000`, Hit@5 `1.0000`
- 위 수치는 `project_documents_kure_v5`에서 `human_verified=false` 케이스로 측정한 참고
  결과이므로, 현재 `project_documents_kure_v6`의 정식 성능으로 간주하지 않는다.
