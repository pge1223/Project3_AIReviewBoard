import { useState } from 'react'
import { previewIdeationMeeting } from '../api/ideationApi'

// 용준/Claude(2026-07-20): 개발용 "아이디어 발전 회의" 프리뷰 화면. 정식 심사 화면
// (MentorSelectionPage 등)과는 완전히 분리된 별도 페이지이며, 기존 화면/라우팅은
// 건드리지 않는다. 목적은 planning_expert -> dev_expert -> planning_expert_revise ->
// facilitator 흐름이 실제 LLM 호출 및 화면까지 올바르게 이어지는지 눈으로 확인하는 것 —
// backend/app/api/routes/ideation_preview.py(ENABLE_IDEATION_PREVIEW=true일 때만 존재)를
// 그대로 호출한다.

const PERSONA_META = {
  planning_expert: { label: '기획 전문가', color: '#7c4dff' },
  planning_expert_revise: { label: '기획 전문가 (수정안)', color: '#ab47bc' },
  dev_expert: { label: '개발 전문가', color: '#00897b' },
  ideation_facilitator: { label: '진행위원', color: '#546e7a' },
}

// turns[]는 백엔드가 speaker_id를 항상 "planning_expert"/"dev_expert"로만 정규화한다
// (ideation_nodes.py::_normalize_turn — 별도 "revise" id가 없다). 화면에서 기획 전문가의
// 두 번째 발언(같은 라운드 안에서 dev_expert 다음에 오는 발언)만 "수정안"으로 구분해
// 표시한다.
function displayKeyFor(turn, indexInRound) {
  if (turn.speaker_id === 'planning_expert' && indexInRound > 0) return 'planning_expert_revise'
  return turn.speaker_id
}

function groupIndexInRound(turns) {
  const seenPerRound = {}
  return turns.map((turn) => {
    const round = turn.round
    const idx = seenPerRound[round] || 0
    seenPerRound[round] = idx + 1
    return displayKeyFor(turn, idx)
  })
}

function TurnCard({ turn, displayKey }) {
  const meta = PERSONA_META[displayKey] || { label: turn.speaker_id, color: '#8b8fa3' }
  return (
    <div style={{ ...styles.turnCard, borderLeftColor: meta.color }}>
      <div style={styles.turnHeader}>
        <span style={{ ...styles.personaBadge, background: meta.color }}>{meta.label}</span>
        <span style={styles.turnMeta}>
          {turn.speaker_name} · {turn.role} · 라운드 {turn.round} · {turn.stance}
        </span>
      </div>
      <p style={styles.turnSummary}>{turn.summary}</p>

      {turn.proposals?.length > 0 && (
        <div style={styles.turnField}>
          <span style={styles.turnFieldLabel}>제안</span>
          <ul style={styles.turnList}>
            {turn.proposals.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
      )}
      {turn.risks?.length > 0 && (
        <div style={styles.turnField}>
          <span style={styles.turnFieldLabel}>위험</span>
          <ul style={styles.turnList}>
            {turn.risks.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
      {turn.questions_for_expert?.length > 0 && (
        <div style={styles.turnField}>
          <span style={styles.turnFieldLabel}>상대 전문가에게</span>
          <ul style={styles.turnList}>
            {turn.questions_for_expert.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </div>
      )}
      {turn.questions_for_user?.length > 0 && (
        <div style={styles.turnField}>
          <span style={styles.turnFieldLabel}>사용자에게</span>
          <ul style={styles.turnList}>
            {turn.questions_for_user.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </div>
      )}
      {turn.evidence?.length > 0 && (
        <div style={styles.turnField}>
          <span style={styles.turnFieldLabel}>근거</span>
          <ul style={styles.turnList}>
            {turn.evidence.map((e, i) => (
              <li key={i}>
                {e.document_name || e.document_id || '출처 미상'}
                {e.page ? ` (p.${e.page})` : ''} — “{e.quote}”
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default function IdeationPreviewPage() {
  const [competitionName, setCompetitionName] = useState('')
  const [competitionDocument, setCompetitionDocument] = useState('')
  const [userIdea, setUserIdea] = useState('')
  const [useRag, setUseRag] = useState(false)
  const [projectId, setProjectId] = useState('')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [showRaw, setShowRaw] = useState(false)

  const canStart = competitionName.trim() && competitionDocument.trim() && userIdea.trim() && !loading

  function handleStart() {
    setLoading(true)
    setError('')
    setResult(null)
    previewIdeationMeeting({
      competitionName,
      competitionDocument,
      userIdea,
      useRag,
      projectId: useRag ? projectId : undefined,
    })
      .then((data) => setResult(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }

  const displayKeys = result ? groupIndexInRound(result.turns) : []

  return (
    <div style={styles.page}>
      <h1 style={styles.title}>아이디어 발전 회의 · 개발용 프리뷰</h1>
      <p style={styles.subtitle}>
        정식 기능이 아닙니다 — 결과는 저장되지 않고, planning_expert → dev_expert →
        planning_expert_revise → facilitator 흐름만 검증합니다.
      </p>

      <div style={styles.formBox}>
        <label style={styles.label}>공모전명</label>
        <input
          style={styles.input}
          value={competitionName}
          onChange={(e) => setCompetitionName(e.target.value)}
          placeholder="예: 소상공인 디지털 혁신 아이디어 공모전"
        />

        <label style={styles.label}>공고문</label>
        <textarea
          style={styles.textarea}
          value={competitionDocument}
          onChange={(e) => setCompetitionDocument(e.target.value)}
          placeholder="공고 목적, 평가 기준 등을 붙여넣으세요."
          rows={4}
        />

        <label style={styles.label}>사용자 아이디어</label>
        <textarea
          style={styles.textarea}
          value={userIdea}
          onChange={(e) => setUserIdea(e.target.value)}
          placeholder="초기 아이디어를 입력하세요."
          rows={4}
        />

        <label style={styles.checkboxRow}>
          <input type="checkbox" checked={useRag} onChange={(e) => setUseRag(e.target.checked)} />
          RAG 사용(공고문 등록된 project_id로 근거 검색)
        </label>
        {useRag && (
          <input
            style={styles.input}
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
            placeholder="project_id"
          />
        )}

        <button style={styles.startButton} onClick={handleStart} disabled={!canStart}>
          {loading ? '회의 진행 중...' : '회의 시작'}
        </button>
      </div>

      {loading && <p style={styles.muted}>실제 LLM 호출로 회의를 진행하는 중입니다. 잠시 기다려주세요...</p>}
      {error && <p style={styles.error}>{error}</p>}

      {result && (
        <div style={styles.resultBox}>
          <div style={styles.statusRow}>
            <span style={styles.statusBadge}>상태: {result.status}</span>
            <span style={styles.statusBadge}>현재 라운드: {result.current_round}</span>
          </div>

          {result.error && <p style={styles.error}>{result.error.message}</p>}
          {result.pending_question && (
            <div style={styles.pendingBox}>
              <strong>진행위원이 사용자에게 질문했습니다:</strong> {result.pending_question}
            </div>
          )}

          <div style={styles.sectionLabel}>발언 기록</div>
          {result.turns.map((turn, i) => (
            <TurnCard key={i} turn={turn} displayKey={displayKeys[i]} />
          ))}

          {result.facilitator_summary && (
            <>
              <div style={styles.sectionLabel}>진행위원 합의 내용</div>
              <div style={{ ...styles.turnCard, borderLeftColor: PERSONA_META.ideation_facilitator.color }}>
                <span
                  style={{ ...styles.personaBadge, background: PERSONA_META.ideation_facilitator.color }}
                >
                  {PERSONA_META.ideation_facilitator.label}
                </span>
                <div style={styles.turnField}>
                  <span style={styles.turnFieldLabel}>합의점</span>
                  <ul style={styles.turnList}>
                    {result.facilitator_summary.consensus.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                </div>
                <div style={styles.turnField}>
                  <span style={styles.turnFieldLabel}>미해결 쟁점</span>
                  <ul style={styles.turnList}>
                    {result.facilitator_summary.unresolved_issues.map((u, i) => (
                      <li key={i}>{u}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </>
          )}

          {result.final_proposal && (
            <>
              <div style={styles.sectionLabel}>최종 제안서</div>
              <div style={styles.proposalBox}>
                <h3 style={styles.proposalTitle}>{result.final_proposal.idea_name}</h3>
                <p style={styles.turnSummary}>{result.final_proposal.one_line_pitch}</p>
                <div style={styles.turnField}>
                  <span style={styles.turnFieldLabel}>문제 정의</span>
                  <p style={styles.turnSummary}>{result.final_proposal.problem_definition}</p>
                </div>
                <div style={styles.turnField}>
                  <span style={styles.turnFieldLabel}>핵심 기능</span>
                  <ul style={styles.turnList}>
                    {result.final_proposal.key_features.map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </div>
                <div style={styles.turnField}>
                  <span style={styles.turnFieldLabel}>다음 작업</span>
                  <ul style={styles.turnList}>
                    {result.final_proposal.next_actions.map((a, i) => (
                      <li key={i}>{a}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </>
          )}

          <button style={styles.rawToggle} onClick={() => setShowRaw((v) => !v)}>
            {showRaw ? '원본 JSON 숨기기' : '원본 JSON 보기'}
          </button>
          {showRaw && <pre style={styles.rawBox}>{JSON.stringify(result, null, 2)}</pre>}
        </div>
      )}
    </div>
  )
}

const ACCENT = '#7c4dff'

const styles = {
  page: { maxWidth: 720, margin: '0 auto', padding: '32px 20px', color: '#1f2333' },
  title: { fontSize: 22, fontWeight: 700, margin: '0 0 6px' },
  subtitle: { fontSize: 13, color: '#8b8fa3', marginBottom: 24 },
  formBox: { display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 20 },
  label: { fontSize: 12.5, fontWeight: 700, color: '#4b4f63', marginTop: 10 },
  input: {
    border: '1.5px solid #e5e3f0',
    borderRadius: 10,
    padding: '10px 12px',
    fontSize: 13.5,
    font: 'inherit',
  },
  textarea: {
    border: '1.5px solid #e5e3f0',
    borderRadius: 10,
    padding: '10px 12px',
    fontSize: 13.5,
    font: 'inherit',
    resize: 'vertical',
  },
  checkboxRow: { display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, marginTop: 10 },
  startButton: {
    marginTop: 16,
    padding: '12px 0',
    borderRadius: 12,
    border: 'none',
    background: ACCENT,
    color: '#fff',
    fontSize: 14,
    fontWeight: 700,
    cursor: 'pointer',
  },
  muted: { color: '#8b8fa3', fontSize: 13 },
  error: { color: '#d64545', fontSize: 13 },
  resultBox: { marginTop: 20 },
  statusRow: { display: 'flex', gap: 8, marginBottom: 16 },
  statusBadge: {
    fontSize: 12,
    fontWeight: 600,
    color: '#4b4f63',
    background: '#eef0f7',
    padding: '6px 12px',
    borderRadius: 999,
  },
  pendingBox: {
    background: '#fff8e1',
    border: '1px solid #ffe082',
    borderRadius: 10,
    padding: 12,
    fontSize: 13,
    marginBottom: 16,
  },
  sectionLabel: { fontSize: 13, fontWeight: 700, color: '#4b4f63', margin: '20px 0 10px' },
  turnCard: {
    background: '#fff',
    border: '1px solid #ece9f7',
    borderLeft: '4px solid #7c4dff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
  },
  turnHeader: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' },
  personaBadge: {
    fontSize: 11.5,
    fontWeight: 700,
    color: '#fff',
    padding: '4px 10px',
    borderRadius: 999,
  },
  turnMeta: { fontSize: 12, color: '#8b8fa3' },
  turnSummary: { fontSize: 13.5, lineHeight: 1.6, margin: '4px 0' },
  turnField: { marginTop: 8 },
  turnFieldLabel: { fontSize: 11.5, fontWeight: 700, color: '#8b8fa3' },
  turnList: { margin: '4px 0 0', paddingLeft: 18, fontSize: 13, lineHeight: 1.6 },
  proposalBox: {
    background: '#fff',
    border: '1px solid #ece9f7',
    borderRadius: 12,
    padding: 16,
  },
  proposalTitle: { fontSize: 16, fontWeight: 700, margin: '0 0 4px' },
  rawToggle: {
    marginTop: 16,
    border: 'none',
    background: 'none',
    color: ACCENT,
    fontWeight: 600,
    cursor: 'pointer',
    fontSize: 12.5,
    padding: 0,
  },
  rawBox: {
    marginTop: 10,
    background: '#1f2333',
    color: '#e5e3f0',
    borderRadius: 10,
    padding: 14,
    fontSize: 11.5,
    overflowX: 'auto',
    maxHeight: 400,
  },
}
