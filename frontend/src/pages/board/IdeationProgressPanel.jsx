import { CheckCircle2, Circle, Compass, HelpCircle, ShieldAlert, Users } from 'lucide-react'

// 용준/Claude(2026-07-27): "작성 전(discovery) 회의"의 진행 상태를 한눈에 보여주는 순수
// 표시 전용 패널. API 호출은 하지 않고, IdeationConversationScreen.jsx가 이미 들고 있는
// ideationConv(전체 세션 상태)를 그대로 prop으로 받아 읽기만 한다.
//
// 여기서 쓰는 phase 라벨은 ideationConversationHelpers.js::statusLabelFor가 아니라 이
// 파일 안의 로컬 맵(PHASE_LABEL)을 새로 둔다 — statusLabelFor는 이미 기존 화면 여러
// 곳(상단 배지, 입력창 placeholder 등)에서 starting/sending/finalizing 같은 로컬 진행
// 플래그와 얽혀 쓰이고 있어서, discovery phase를 넣으려고 그 함수를 건드리면 기존 호출부의
// 동작이 바뀔 위험이 있다. 이 패널은 phase 값만 보고 정적으로 라벨을 고른다.
const PHASE_LABEL = {
  // discovery 모드
  problem_discovery: '문제 탐색 중',
  awaiting_problem_focus_selection: '문제 영역 선택 대기',
  problem_focus_selection: '문제 영역 선택 반영 중',
  idea_divergence: '해결 방향 발산 중',
  idea_conflict_and_merge: '방향 수정·결합 중',
  awaiting_conflict_resolution: '해결 방향 조정 대기',
  conflict_resolution: '해결 방향 조정 반영 중',
  candidate_generation: '후보 정리 중',
  awaiting_candidate_selection: '기획·개발 검증 후보 선택',
  provisional_selection: '검증 후보 선택 반영 중',
  idea_validation: '기획·개발 검증 중',
  awaiting_concept_confirmation: '최종 확정 대기',
  concept_confirmation: '최종 확정 반영 중',
  // 기존(리파인먼트) 모드 — discovery 이후 합류하는 phase도 함께 안내한다.
  expert_discussion: '전문가 회의 진행 중',
  awaiting_planning_answer: '기획 의원 답변 대기',
  awaiting_developer_answer: '개발 의원 답변 대기',
  awaiting_user_decision: '위원 논의 완료 · 의견은 선택',
  discussion_complete: '위원 논의 완료',
  waiting_user_input: '사용자 의견 대기',
  candidate_selection: '후보 선택 대기',
  finalizing: '회의 내용 정리 중',
  completed: '회의 완료',
  finalized: '완료',
  failed: '오류 발생',
}

function phaseLabelFor(phase) {
  return PHASE_LABEL[phase] || (phase ? phase : '준비 중')
}

function Section({ icon: Icon, title, children }) {
  return (
    <div style={{ marginTop: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        {Icon && <Icon size={13} color="var(--purple)" />}
        <div style={{ fontSize: 13.5, fontWeight: 700, color: 'var(--text-2)' }}>{title}</div>
      </div>
      {children}
    </div>
  )
}

// unresolved_issues(문자열 배열) + validation_result.{planning,technical}.concerns(있으면)를
// 합쳐서 "검증되지 않은 가정" 목록을 만든다. 두 소스에 겹치는 문구가 있을 수 있어 단순
// 문자열 동등 비교로만 중복을 제거한다(요청: "dedupe simple").
function collectUnverifiedAssumptions(unresolvedIssues, validationResult) {
  const items = [...(unresolvedIssues || [])]
  const concernSources = [validationResult?.planning?.concerns, validationResult?.technical?.concerns]
  for (const concerns of concernSources) {
    if (!Array.isArray(concerns)) continue
    for (const c of concerns) {
      if (typeof c === 'string' && c.trim()) items.push(c.trim())
    }
  }
  const issueSources = [validationResult?.planning?.issues, validationResult?.technical?.issues]
  for (const issues of issueSources) {
    if (!Array.isArray(issues)) continue
    for (const issue of issues) {
      if (typeof issue?.description === 'string' && issue.description.trim()) {
        items.push(issue.description.trim())
      }
    }
  }
  const seen = new Set()
  const deduped = []
  for (const item of items) {
    const key = item.trim()
    if (!key || seen.has(key)) continue
    seen.add(key)
    deduped.push(key)
  }
  return deduped
}

const CHANGED_BY_LABEL = {
  planning_expert: '기획위원',
  dev_expert: '개발위원',
  ideation_facilitator: '진행자',
  user: '사용자',
}

export default function IdeationProgressPanel({ ideationConv }) {
  if (!ideationConv) return null

  const {
    phase,
    problem_definition: problemDefinition,
    problem_focus: problemFocus,
    unresolved_issues: unresolvedIssues,
    idea_evolution: ideaEvolution,
    idea_locked: ideaLocked,
    validation_result: validationResult,
    messages,
  } = ideationConv

  const problemSummary = problemDefinition?.problem
    || (problemFocus?.length > 0 ? problemFocus.map((p) => p.title).filter(Boolean).join(' · ') : '')

  const assumptions = collectUnverifiedAssumptions(unresolvedIssues, validationResult)

  const recentCritiques = (ideaEvolution || [])
    .filter((r) => r?.action_type === 'critique')
    .slice(-3)
    .reverse()

  const lastQuestionMessage = [...(messages || [])].reverse().find((m) => m?.message_type === 'question')

  return (
    <div className="card glass" style={{ padding: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-0)' }}>아이디어 진행 상황</div>
        <span className={`badge ${ideaLocked ? 'green' : 'amber'} mono`} style={{ fontSize: 12 }}>
          {ideaLocked ? '확정됨' : '미확정'}
        </span>
      </div>

      <Section icon={Compass} title="현재 단계">
        <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-0)' }}>{phaseLabelFor(phase)}</div>
      </Section>

      <Section icon={Circle} title="현재 문제 정의">
        <div style={{ fontSize: 14.5, color: problemSummary ? 'var(--text-0)' : 'var(--text-2)', lineHeight: 1.6 }}>
          {problemSummary || '문제 영역을 선택하면 회의를 통해 구체화됩니다.'}
        </div>
      </Section>

      {problemDefinition?.target_user && (
        <Section icon={Users} title="대상 사용자">
          <div style={{ fontSize: 14.5, color: 'var(--text-0)', lineHeight: 1.6 }}>
            {problemDefinition.target_user}
          </div>
        </Section>
      )}

      {assumptions.length > 0 && (
        <Section icon={ShieldAlert} title="검증되지 않은 가정">
          <ul style={{ margin: 0, paddingLeft: 16, fontSize: 14, color: 'var(--text-0)', lineHeight: 1.65 }}>
            {assumptions.map((a, i) => <li key={i}>{a}</li>)}
          </ul>
        </Section>
      )}

      <Section icon={ShieldAlert} title="전문가 간 주요 쟁점">
        {recentCritiques.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {recentCritiques.map((c) => (
              <div
                key={c.record_id}
                style={{
                  fontSize: 13.5, color: 'var(--text-1)', lineHeight: 1.55,
                  background: 'var(--bg-0)', border: '1px solid var(--glass-border)', borderRadius: 8, padding: '6px 8px',
                }}
              >
                <div style={{ fontWeight: 600, color: 'var(--text-2)', marginBottom: 2, fontSize: 12.5 }}>
                  {CHANGED_BY_LABEL[c.changed_by] || c.changed_by || '위원'}
                </div>
                {c.content || c.title}
              </div>
            ))}
          </div>
        ) : (
          <div style={{ fontSize: 14, color: 'var(--text-2)', lineHeight: 1.6 }}>
            아직 전문가 간 주요 쟁점이 생성되지 않았습니다.
          </div>
        )}
      </Section>

      <Section icon={HelpCircle} title="다음 회의 질문">
        <div style={{ fontSize: 14.5, color: lastQuestionMessage ? 'var(--text-0)' : 'var(--text-2)', lineHeight: 1.6 }}>
          {lastQuestionMessage?.content || `${phaseLabelFor(phase)} 단계입니다. 아직 새로운 회의 질문이 없습니다.`}
        </div>
      </Section>

      <Section icon={CheckCircle2} title="아이디어 확정 여부">
        <div style={{ fontSize: 14.5, fontWeight: 600, color: ideaLocked ? 'var(--purple)' : 'var(--text-2)' }}>
          {ideaLocked ? '확정됨' : '미확정 — 계속 논의 중'}
        </div>
      </Section>
    </div>
  )
}
