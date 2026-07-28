# AI Review Board 프롬프트 목록

프롬프트 기본 위치: [`ai/meeting/prompts`](../ai/meeting/prompts)

## 1. 현재 V2 신청서 코치

현재 아이디어 후보를 선택한 뒤 신청 양식을 대화로 작성하는 흐름에서 사용하는 핵심 프롬프트다.

| 프롬프트 | 목적 |
|---|---|
| [`ideation_form_coach_answer_assessment_01.txt`](../ai/meeting/prompts/ideation_form_coach_answer_assessment_01.txt) | 직전 사용자 답변의 단계 적합성·구체성·전문가 필요 여부를 화면 비노출 상태로 먼저 판정 |
| [`ideation_form_coach_expert_review_01.txt`](../ai/meeting/prompts/ideation_form_coach_expert_review_01.txt) | 판정 결과가 요구한 기존 기획·개발 페르소나와 RAG 근거를 사용해 사후 보완 조언 생성 |
| [`ideation_form_coach_01.txt`](../ai/meeting/prompts/ideation_form_coach_01.txt) | 답변 판정과 필요한 전문가 검토 결과를 받은 뒤 상태 갱신, 신청서 `draft_patch`, 다음 질문을 생성하는 메인 진행자 프롬프트 |

관련 코드:

- [`ai/meeting/form_coach/prompt_builder.py`](../ai/meeting/form_coach/prompt_builder.py)
- [`ai/meeting/form_coach/expert_review.py`](../ai/meeting/form_coach/expert_review.py)
- [`backend/app/api/routes/ideation_form_coach.py`](../backend/app/api/routes/ideation_form_coach.py)

전문가 화면 출력은 `post_answer_mentor_messages`만 사용한다. 직전 사용자 답변에 전문가
판단이 필요할 때만 표시하며, 첫 질문 전 사전 전문가 메시지는 생성하지 않는다.

## 2. 아이디어 후보 발굴

사용자에게 초기 아이디어가 없을 때 공고문을 바탕으로 후보를 생성하고 검토한다.

| 프롬프트 | 목적 |
|---|---|
| [`ideation_conv_candidate_planning.txt`](../ai/meeting/prompts/ideation_conv_candidate_planning.txt) | 공고문과 평가기준을 바탕으로 서로 다른 아이디어 후보 2~3개 생성 |
| [`ideation_conv_candidate_feasibility.txt`](../ai/meeting/prompts/ideation_conv_candidate_feasibility.txt) | 후보별 MVP, 기술, 데이터, 구현 가능성을 개발 관점에서 검토 |
| [`ideation_conv_candidate_selection.txt`](../ai/meeting/prompts/ideation_conv_candidate_selection.txt) | 후보 결합, 전문가 추천, 모호한 자연어 선택 답변을 해석 |

단순한 번호 또는 제목 선택은 코드에서 처리하므로 후보 선택 프롬프트를 호출하지 않을 수 있다.

## 3. 기존 대화형 아이디어 회의

기획 전문가, 개발 전문가, 진행자가 LangGraph 안에서 질문과 보완 의견을 주고받는 기존 회의 흐름이다.

| 프롬프트 | 목적 |
|---|---|
| [`ideation_conv_question.txt`](../ai/meeting/prompts/ideation_conv_question.txt) | 기획 또는 개발 전문가가 사용자에게 질문 하나 생성 |
| [`ideation_conv_sufficiency.txt`](../ai/meeting/prompts/ideation_conv_sufficiency.txt) | 사용자 응답을 답변, 설명 요청, 불충분한 답변으로 분류 |
| [`ideation_conv_discussion.txt`](../ai/meeting/prompts/ideation_conv_discussion.txt) | 사용자 답변 이후 기획·개발 전문가의 보완 의견 생성 |
| [`ideation_conv_discussion_facilitator_02.txt`](../ai/meeting/prompts/ideation_conv_discussion_facilitator_02.txt) | 전문가 의견을 사용자 결정과 신청 양식 작성으로 연결하는 진행자 02 |
| [`ideation_conv_discussion_facilitator.txt`](../ai/meeting/prompts/ideation_conv_discussion_facilitator.txt) | 합의점과 이견을 중심으로 회의를 정리하는 이전 진행자 프롬프트 |
| [`ideation_conv_canvas_update.txt`](../ai/meeting/prompts/ideation_conv_canvas_update.txt) | 라운드 결과를 오른쪽 아이디어 기획 캔버스의 구조화된 값으로 갱신 |
| [`ideation_conv_synthesis.txt`](../ai/meeting/prompts/ideation_conv_synthesis.txt) | 전체 대화를 바탕으로 최종 아이디어 제안서 초안 생성 |

현재 기존 그래프의 진행자 템플릿은
[`prompt_loader.py`](../ai/meeting/prompts/prompt_loader.py)에서
`ideation_conv_discussion_facilitator_02.txt`로 지정되어 있다.

롤백하려면 다음 상수를 이전 파일명으로 변경한다.

```python
IDEATION_CONV_DISCUSSION_FACILITATOR_TEMPLATE = (
    "ideation_conv_discussion_facilitator.txt"
)
```

## 4. 전문가 위임 처리

사용자가 직접 답을 정하기 어렵거나 전문가에게 판단을 맡겼을 때 사용하는 흐름이다.

| 프롬프트 | 목적 |
|---|---|
| [`ideation_conv_expert_delegation.txt`](../ai/meeting/prompts/ideation_conv_expert_delegation.txt) | 담당 전문가가 수정 가능한 임시 방향을 제안 |
| [`ideation_conv_expert_delegation_review.txt`](../ai/meeting/prompts/ideation_conv_expert_delegation_review.txt) | 반대 역할 전문가가 제안을 교차 검토 |
| [`ideation_conv_expert_delegation_facilitator.txt`](../ai/meeting/prompts/ideation_conv_expert_delegation_facilitator.txt) | 제안과 검토 결과를 하나의 최종 권고안으로 정리 |

## 5. 이전 배치형 아이디어 회의

사용자가 매 질문에 답하는 현재 대화형 구조 이전에 사용하던 라운드 기반 회의 프롬프트다.

| 프롬프트 | 목적 |
|---|---|
| [`ideation_common.txt`](../ai/meeting/prompts/ideation_common.txt) | 기획·개발 전문가가 함께 사용하는 공통 회의 규칙 |
| [`ideation_facilitator_prompt.txt`](../ai/meeting/prompts/ideation_facilitator_prompt.txt) | 라운드별 전문가 발언을 정리하고 다음 행동 결정 |
| [`ideation_synthesis_prompt.txt`](../ai/meeting/prompts/ideation_synthesis_prompt.txt) | 배치형 회의 전체 결과를 아이디어 제안서로 종합 |

## 6. 문서 심사

아이디어 발굴이 아니라 작성된 문서를 평가하는 AI Review Board 심사 흐름에서 사용한다.

| 프롬프트 | 목적 |
|---|---|
| [`reviewer_prompt.txt`](../ai/meeting/prompts/reviewer_prompt.txt) | 제출 문서를 평가기준과 RAG 근거로 심사 |
| [`chair_prompt.txt`](../ai/meeting/prompts/chair_prompt.txt) | 여러 심사위원 결과를 합의점, 이견, 리스크, 수정 우선순위로 종합 |

## 현재 흐름 요약

현재 주제 발굴과 신청서 작성의 중심 흐름은 다음과 같다.

```text
공고문 분석
  -> 아이디어 후보 생성
  -> 개발 가능성 검토
  -> 사용자 후보 선택
  -> V2 신청서 코치
  -> 사용자 답변별 신청서 초안 갱신
```

핵심적으로 관리할 프롬프트는 다음 5개다.

1. `ideation_conv_candidate_planning.txt`
2. `ideation_conv_candidate_feasibility.txt`
3. `ideation_conv_candidate_selection.txt`
4. `ideation_form_coach_answer_assessment_01.txt`
5. `ideation_form_coach_expert_review_01.txt`
6. `ideation_form_coach_01.txt`
