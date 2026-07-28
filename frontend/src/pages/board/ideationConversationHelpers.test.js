import test from 'node:test'
import assert from 'node:assert/strict'

import { discussionIdeaCountFor, nextStepLabelFor } from './ideationConversationHelpers.js'

test('phase에 따라 active 방향 또는 발전된 후보 수를 표시한다', () => {
  assert.equal(
    discussionIdeaCountFor({
      phase: 'idea_conflict_and_merge',
      solution_directions: [
        { status: 'active' },
        { status: 'merged' },
        { status: 'superseded' },
        { status: 'active' },
      ],
    }),
    2,
  )
  assert.equal(
    discussionIdeaCountFor({
      phase: 'awaiting_candidate_selection',
      solution_directions: [{ status: 'active' }, { status: 'active' }, { status: 'active' }],
      idea_candidates: [{}, {}],
    }),
    2,
  )
})

test('후보 선택 단계의 다음 단계는 기획·개발 검증이다', () => {
  assert.equal(nextStepLabelFor({ phase: 'awaiting_candidate_selection', idea_locked: false }), '기획·개발 검증')
  assert.equal(nextStepLabelFor({ phase: 'idea_validation', idea_locked: false }), '검증 결과 반영')
  assert.equal(nextStepLabelFor({ phase: 'awaiting_concept_confirmation', idea_locked: false }), '최종 아이디어 확정')
  assert.equal(nextStepLabelFor({ phase: 'expert_discussion', idea_locked: true }), '구현 구체화')
  assert.equal(nextStepLabelFor({ phase: 'finalized', idea_locked: true }), '신청서 초안')
})
