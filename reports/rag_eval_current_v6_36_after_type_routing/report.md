# RAG Evaluation

- 평가 일시: 2026-07-28T14:57:03.289939+00:00
- 데이터셋: rag_eval_current_v6_36 v1.0.0
- 데이터셋 경로: reports/rag_eval_current_v6_36.jsonl
- 검색 설정: top_k=5, collection=project_documents_kure_v6
- 생성 모델: N/A
- 평가 모델: N/A (prompt_version=faithfulness_judge_v1 / persona_fit_judge_v1)
- 검수 완료 케이스 수: 0 / 36

## Summary

- Recall@5: N/A (참고: 1.0000, 검수 0건 기준)
- Hit@5: N/A (참고: 1.0000)
- MRR@5: N/A (참고: 1.0000)
- 검색 실패율(예상 근거 있는데 결과 0건): 0.0000
- 근거 없음 예상 케이스 정확도: N/A (0건)
- 평균 검색 시간: 288.6ms
- 예상 비용: 미측정

## Worst Cases

### Recall@K가 낮은 질문
- `v6_001` recall=1.00 query='2026 공공기관 AI 혁신 챌린지를 개최하는 목적과 슬로건은 무엇인가요?'
- `v6_002` recall=1.00 query='공공기관 AI 혁신 챌린지의 주최·주관 기관은 어디인가요?'
- `v6_003` recall=1.00 query='공모 부문 두 가지와 지정 주제·자유주제 Track의 차이를 설명해 주세요.'
- `v6_004` recall=1.00 query='AI 활용 우수사례 부문에 제출할 수 있는 운영 사례의 조건은 무엇인가요?'
- `v6_005` recall=1.00 query='지정 주제에서 인정하는 사회 현안의 예시와 범위 제한은 무엇인가요?'

## Notes

v6-36-final-type-routing-and-mrr
