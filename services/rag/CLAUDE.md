# services/rag — RAG 인프라 + FIND 블록

담당: 김용준 (단일 소유권 — 다른 파트는 이슈로 변경 요청)

## 이 폴더 컨텍스트
- BM25 + 벡터 하이브리드 검색 파이프라인
- 청킹: 500~800 토큰, overlap 80~120
- 메타데이터 태깅: 출처 인용(citation)을 위해 각 청크에 공고 ID/출처 URL 유지
- 평가: Ragas (Faithfulness, Answer Relevance, Context Precision/Recall),
  Golden Set 20~50개 QA triple — `services/rag/eval/golden_set.json`
  (질문 유형: factual / synthesis / trap 혼합)

## 페르소나
- 공용 시스템 프롬프트는 `services/rag/persona/system_prompt.md`에서 관리
  (설계/내용은 민경 담당, 이 파일 위치만 공유)
- 다른 폴더(apps/server 등)에서는 이 파일을 import해서 사용, 복붙 금지

## 하지 말 것
- HWP 파싱 로직 추가 제안 금지 (팀 결정으로 제외됨)
- 청킹/임베딩 파라미터를 다른 파트에서 임의 변경 금지 — 김용준과 협의
