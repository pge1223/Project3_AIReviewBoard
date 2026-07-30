# Ragas 평가 전용 환경 (앱과 완전히 분리)

이 폴더는 앱(review-board conda 환경, `requirements.txt`)과 **의존성을 공유하지 않는다**.
`ragas` 패키지가 이미 삭제된 `langchain_community.chat_models.vertexai`를 무조건
import하는데(2026-07-29 실측, ragas 0.4.3/0.2.15 둘 다 동일), 앱이 쓰는 최신
langchain-core/langchain-community(langchain-huggingface·langchain-chroma가 끌고 옴)와
같은 환경에 두면 `import ragas`만 해도 `ModuleNotFoundError`가 난다. 그래서 이 폴더의
스크립트는 **앱 코드(`ai.*`, `backend.*`)를 import하지 않고**, 파일(JSONL)로만 앱과
통신한다.

```
[앱 환경: review-board]                    [평가 환경: ragas-eval]
ai/rag/evaluation/rag_quality/       →      ai/rag/evaluation/ragas_standalone/
  export_ragas_dataset.py                     run_ragas_eval.py
  (실제 회의 그래프 실행 → JSONL 저장)          (JSONL 읽어 Ragas 지표 계산 → JSON/CSV 저장)
```

## 1. 평가 환경 생성

```bash
conda env create -f ai/rag/evaluation/ragas_standalone/environment-ragas.yml
conda activate ragas-eval
pip install -r ai/rag/evaluation/ragas_standalone/requirements-ragas.txt
```

(conda 없이 venv를 쓰는 경우)

```bash
python3.11 -m venv .venv-ragas
source .venv-ragas/bin/activate  # Windows: .venv-ragas\Scripts\activate
pip install -r ai/rag/evaluation/ragas_standalone/requirements-ragas.txt
```

Python 3.11에서 확인했다(`ragas==0.2.15` + `langchain-community<0.4` 조합이 이 버전 기준
정상 import된다 — 최신 `ragas==0.4.3`은 같은 vertexai import 문제로 이 프로젝트
langchain 스택과 무관하게 fresh 환경에서도 동일하게 깨진다).

## 2. 앱 환경에서 평가 데이터 내보내기 (review-board 환경)

```bash
conda activate review-board
python -m ai.rag.evaluation.rag_quality.export_ragas_dataset \
  --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl \
  --top-k 5 \
  --limit 10 \
  --output reports/ragas_export/dataset.jsonl
```

OpenAI 호출(발언 생성)이 발생하므로 처음에는 `--limit`을 작게 준다. 결과:

- `reports/ragas_export/dataset.jsonl` — 평가 입력
- `reports/ragas_export/dataset.manifest.json` — 실행 설정 기록(모델, top_k, 케이스 수 등)

### 입력 JSONL 한 줄 예시

```json
{
  "case_id": "rag_eval_001",
  "message_id": "msg_ab12cd34",
  "persona_id": "dev_expert",
  "round": 1,
  "user_input": "chat K-Lawyer 서비스가 잘못된 답변(hallucination)을 예방하기 위해 어떤 방법을 쓰나요?",
  "response": "K-Lawyer는 질의 전에 관련 분야를 먼저 선택하도록 해서...",
  "retrieved_contexts": ["① 법제처 법령 학습... Hallucination 예방을 위해...", "..."],
  "reference": null
}
```

`reference`는 `ai/rag/evaluation/rag_quality/schemas.py`의 `RagEvalCase.reference_answer`
(선택 필드, 기본값 없음)에서 온다. 대부분의 케이스는 아직 사람이 작성한 정답 발언이
없으므로 `null`이다 — 그런 행은 Context Precision/Context Recall이 계산되지 않는다
(아래 4번 참고). 정답 발언을 붙이려면 평가셋 JSONL에 `"reference_answer": "..."` 필드를
추가하면 된다.

## 3. 평가 환경에서 Ragas 실행

```bash
conda activate ragas-eval
export OPENAI_API_KEY=sk-...   # Windows PowerShell: $env:OPENAI_API_KEY = "sk-..."
python ai/rag/evaluation/ragas_standalone/run_ragas_eval.py \
  --input reports/ragas_export/dataset.jsonl \
  --output-dir reports/ragas_eval \
  --model gpt-4o-mini \
  --embedding-model text-embedding-3-small
```

출력:

- `reports/ragas_eval/ragas_report.json` — 행 단위 원시 결과 + 집계(macro average)
- `reports/ragas_eval/ragas_report.csv` — 행 단위 결과(스프레드시트용)

## 4. 실제로 측정되는 지표

| 지표 | 필요한 필드 | reference 없어도 계산됨 |
|---|---|---|
| Faithfulness | `user_input`, `response`, `retrieved_contexts` | O |
| Answer Relevancy | `user_input`, `response` | O |
| Context Precision | `user_input`, `retrieved_contexts`, `reference` | **X — reference 없으면 `not_measurable`** |
| Context Recall | `user_input`, `retrieved_contexts`, `reference` | **X — reference 없으면 `not_measurable`** |

Context Precision/Context Recall은 Ragas 원래 정의(`LLMContextPrecisionWithReference`,
`LLMContextRecall`)상 정답 발언(`reference`)이 필수다. `reference`가 없는 행은 값을 만들어
채우거나 다른 계산식으로 대체하지 않고, `context_precision_status`/`context_recall_status`
필드에 `"not_measurable"`로 명시한다(`ragas_report.json`의 `aggregate.
context_precision_not_measurable_count` / `context_recall_not_measurable_count`로 개수
확인 가능). 정식으로 이 두 지표를 보려면 평가셋에 사람이 검수한 `reference_answer`를
채워 넣고 다시 내보내야 한다.

## 5. 앱 환경 원상복구 확인

이 폴더의 어떤 파일도 `requirements.txt`나 `review-board` conda 환경을 건드리지 않는다.
앱 쪽에서 바뀐 것은 `ai/rag/evaluation/rag_quality/`에 새 파일 1개
(`export_ragas_dataset.py`, ragas 미설치)와 `schemas.py`의 선택 필드 1개
(`RagEvalCase.reference_answer`)뿐이다.
