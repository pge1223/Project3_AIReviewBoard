import { useRef, useState } from 'react'
import {
  finalizeIdeationConversation,
  replyIdeationConversation,
  startIdeationConversation,
} from '../api/ideationConversationApi'

// 용준/Claude(2026-07-20): 개발용 "대화형 아이디어 발전 회의" 프리뷰 화면. 기존
// /ideation-preview(배치형 — 한 번에 끝까지 실행)와 완전히 분리된 별도 페이지다.
// 목적은 기획 질문 -> 사용자 답변 -> 개발 질문 -> 사용자 답변 -> 두 전문가 보완 ->
// (필요시 추가 질문 라운드) -> 사용자가 확정 버튼을 눌러야만 최종 제안서 생성이라는
// 대화형 흐름이 실제 LLM 호출 및 화면까지 올바르게 이어지는지 눈으로 확인하는 것이다.

const SPEAKER_META = {
  planning_expert: { label: '기획 전문가', color: '#7c4dff', align: 'left' },
  dev_expert: { label: '개발 전문가', color: '#00897b', align: 'left' },
  ideation_facilitator: { label: '진행위원', color: '#e08e45', align: 'left' },
  user: { label: '나', color: '#2f6fed', align: 'right' },
}

const PHASE_LABEL = {
  awaiting_candidate_selection: '후보 중 발전시킬 아이디어를 선택해 주세요',
  awaiting_planning_answer: '기획 전문가의 질문에 답변 대기 중',
  awaiting_developer_answer: '개발 전문가의 질문에 답변 대기 중',
  awaiting_user_decision: '두 전문가 논의 완료 — 더 이야기하거나 주제를 확정하세요',
  finalized: '주제 확정 완료',
  failed: '오류가 발생했습니다',
}

// 용준/Claude(2026-07-21): 배지는 ideation_mode(세션이 "최초 진입할 때"의 모드, 세션 내내
// 고정)가 아니라 active_stage(현재 진행 단계, 후보 선택 후 바뀜)를 기준으로 표시한다 —
// 그래야 discovery로 시작해도 후보를 선택한 뒤에는 "아이디어 발전 모드"로 정확히 바뀐다.
const ACTIVE_STAGE_LABEL = {
  candidate_discovery: '아이디어 발굴 모드',
  candidate_selection: '아이디어 발굴 모드',
  refinement: '아이디어 발전 모드',
  finalized: '아이디어 발전 모드',
}

const FEASIBILITY_LABEL = { high: '높음', medium: '보통', low: '낮음' }

const STEPS = ['공모전 입력', '공모전 분석', '주제 아이디어 회의', '주제 확정']

function stepIndexFor(result) {
  if (!result) return 0
  if (result.phase === 'finalized') return 3
  return 2
}

// 용준/Claude(2026-07-21, 전문가 의견 UX 개선): 실제 사용자 테스트에서 전문가 의견이 너무
// 길어 읽기 어렵다는 문제가 확인됐다 — 백엔드가 opinion/agreement/disagreement 메시지에
// 순수 추가한 message.structured(judgment/reason/suggestion/confirmed/unconfirmed)가 있으면
// 판단+제안만 기본 노출하고, 근거·확정·미확정 사항은 "상세 보기"로 접는다. structured가 없는
// 메시지(질문/답변/설명/요약, 또는 구버전 세션)는 기존처럼 content 전체를 그대로 보여준다 —
// 화면 디자인이나 빈 content 오류 처리는 건드리지 않는다.
const COLLAPSIBLE_MESSAGE_TYPES = new Set(['opinion', 'agreement', 'disagreement'])

function MessageBubble({ message }) {
  const meta = SPEAKER_META[message.speaker_id] || { label: message.speaker_name, color: '#8b8fa3', align: 'left' }
  const isRight = meta.align === 'right'
  const [detailsOpen, setDetailsOpen] = useState(false)
  const structured = message.structured
  const isCollapsible = !!structured && COLLAPSIBLE_MESSAGE_TYPES.has(message.message_type)

  return (
    <div style={{ ...styles.bubbleRow, justifyContent: isRight ? 'flex-end' : 'flex-start' }}>
      <div style={{ ...styles.bubble, borderColor: meta.color, background: isRight ? '#eef3ff' : '#fff' }}>
        <div style={styles.bubbleHeader}>
          <span style={{ ...styles.speakerBadge, background: meta.color }}>{meta.label}</span>
          <span style={styles.bubbleMeta}>
            R{message.round} · {message.message_type}
          </span>
        </div>
        {isCollapsible ? (
          <>
            <p style={styles.bubbleContent}>
              {structured.judgment}
              {structured.suggestion ? `\n\n[제안]\n${structured.suggestion}` : ''}
            </p>
            <button style={styles.detailToggle} onClick={() => setDetailsOpen((v) => !v)}>
              {detailsOpen ? '상세 접기' : '상세 보기(근거·확정·미확정)'}
            </button>
            {detailsOpen && (
              <div style={styles.detailBox}>
                {structured.reason && <p style={styles.bubbleContent}>{`[근거]\n${structured.reason}`}</p>}
                {structured.confirmed?.length > 0 && (
                  <div style={styles.turnField}>
                    <span style={styles.turnFieldLabel}>확정 사항</span>
                    <ul style={styles.turnList}>
                      {structured.confirmed.map((c, i) => (
                        <li key={i}>{c}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {structured.unconfirmed?.length > 0 && (
                  <div style={styles.turnField}>
                    <span style={styles.turnFieldLabel}>미확정 사항</span>
                    <ul style={styles.turnList}>
                      {structured.unconfirmed.map((u, i) => (
                        <li key={i}>{u}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <p style={styles.bubbleContent}>{message.content}</p>
        )}
        {message.evidence?.length > 0 && (
          <div style={styles.evidenceBox}>
            {message.evidence.map((e, i) => (
              <div key={i} style={styles.evidenceItem}>
                {e.document_name || e.document_id || '출처 미상'}
                {e.page ? ` (p.${e.page})` : ''} — “{e.quote}”
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function IdeationConversationPreviewPage() {
  const [competitionName, setCompetitionName] = useState('')
  const [competitionDocument, setCompetitionDocument] = useState('')
  const [userIdea, setUserIdea] = useState('')
  const [maxRounds, setMaxRounds] = useState(3)
  const [model, setModel] = useState('')

  const [starting, setStarting] = useState(false)
  const [sending, setSending] = useState(false)
  const [finalizing, setFinalizing] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [draft, setDraft] = useState('')
  const [showRaw, setShowRaw] = useState(false)

  const scrollRef = useRef(null)

  // 용준/Claude(2026-07-21): 초기 아이디어는 이제 선택 항목이다 — 비워두면 서버가
  // discovery(아이디어 발굴) 모드로 시작해 후보를 대신 제안해 준다. 필수 항목은
  // 공모전명뿐이다.
  const canStart = competitionName.trim() && !starting
  const isBusy = starting || sending || finalizing
  const canReply =
    result &&
    [
      'awaiting_candidate_selection',
      'awaiting_planning_answer',
      'awaiting_developer_answer',
      'awaiting_user_decision',
    ].includes(result.phase) &&
    !isBusy
  const canFinalize = result && result.phase === 'awaiting_user_decision' && !isBusy

  function scrollToBottom() {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
    })
  }

  function handleStart() {
    setStarting(true)
    setError('')
    setResult(null)
    startIdeationConversation({ competitionName, competitionDocument, userIdea, maxRounds, model })
      .then((data) => {
        setResult(data)
        scrollToBottom()
      })
      .catch((err) => setError(err.message))
      .finally(() => setStarting(false))
  }

  function handleSend() {
    if (!draft.trim() || !result) return
    setSending(true)
    setError('')
    replyIdeationConversation(result.session_id, draft.trim(), model)
      .then((data) => {
        setResult(data)
        setDraft('')
        scrollToBottom()
      })
      .catch((err) => setError(err.message))
      .finally(() => setSending(false))
  }

  function handleFinalize() {
    if (!result) return
    setFinalizing(true)
    setError('')
    finalizeIdeationConversation(result.session_id, model)
      .then((data) => {
        setResult(data)
        scrollToBottom()
      })
      .catch((err) => setError(err.message))
      .finally(() => setFinalizing(false))
  }

  const activeStep = stepIndexFor(result)

  return (
    <div style={styles.page}>
      <h1 style={styles.title}>아이디어 발전 회의 · 대화형 개발용 프리뷰</h1>
      <p style={styles.subtitle}>
        정식 기능이 아닙니다 — 결과는 저장되지 않고(서버 재시작 시 세션이 사라짐), 질문 하나마다
        사용자 답변을 기다리는 대화형 흐름만 검증합니다.
      </p>

      <div style={styles.layout}>
        {/* 왼쪽: 단계 */}
        <aside style={styles.leftPanel}>
          <div style={styles.panelTitle}>진행 단계</div>
          {STEPS.map((step, i) => (
            <div key={step} style={{ ...styles.stepItem, opacity: i <= activeStep ? 1 : 0.4 }}>
              <span style={{ ...styles.stepDot, background: i <= activeStep ? '#7c4dff' : '#d8d5ea' }} />
              {step}
            </div>
          ))}

          <div style={styles.panelTitle}>회의 설정</div>
          <label style={styles.label}>공모전명</label>
          <input
            style={styles.input}
            value={competitionName}
            onChange={(e) => setCompetitionName(e.target.value)}
            placeholder="예: 소상공인 디지털 혁신 아이디어 공모전"
            disabled={!!result}
          />
          <label style={styles.label}>공고문(선택)</label>
          <textarea
            style={styles.textarea}
            value={competitionDocument}
            onChange={(e) => setCompetitionDocument(e.target.value)}
            rows={3}
            disabled={!!result}
          />
          <label style={styles.label}>초기 아이디어(선택)</label>
          <textarea
            style={styles.textarea}
            value={userIdea}
            onChange={(e) => setUserIdea(e.target.value)}
            rows={3}
            disabled={!!result}
            placeholder="아이디어가 있다면 입력해 주세요. 비워두면 공고문과 심사 기준을 바탕으로 전문가들이 아이디어 후보를 제안합니다."
          />
          <label style={styles.label}>최대 라운드</label>
          <input
            style={styles.input}
            type="number"
            min={1}
            max={3}
            value={maxRounds}
            onChange={(e) => setMaxRounds(Number(e.target.value))}
            disabled={!!result}
          />
          <label style={styles.label}>모델(선택, 비우면 기본값)</label>
          <input
            style={styles.input}
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="예: gpt-4.1-mini"
            disabled={!!result}
          />

          {!result && (
            <button style={styles.startButton} onClick={handleStart} disabled={!canStart}>
              {starting ? '회의 시작 중...' : '회의 시작'}
            </button>
          )}
        </aside>

        {/* 중앙: 채팅 */}
        <main style={styles.centerPanel}>
          {result && (
            <div style={styles.statusBar}>
              <span style={{ ...styles.statusBadge, ...styles.modeBadge }}>
                {ACTIVE_STAGE_LABEL[result.active_stage] || result.active_stage}
              </span>
              <span style={styles.statusBadge}>{PHASE_LABEL[result.phase] || result.phase}</span>
              <span style={styles.statusBadge}>라운드 {result.round}/{result.max_rounds}</span>
            </div>
          )}

          {result?.idea_candidates?.length > 0 && (
            <div style={styles.candidatesBox}>
              <div style={styles.sectionLabel}>후보 아이디어</div>
              <div style={styles.candidatesGrid}>
                {result.idea_candidates.map((c) => (
                  <div key={c.candidate_id} style={styles.candidateCard}>
                    <div style={styles.candidateTitle}>{c.title}</div>
                    <div style={styles.turnField}>
                      <span style={styles.turnFieldLabel}>해결할 문제</span>
                      <p style={styles.bubbleContent}>{c.problem}</p>
                    </div>
                    <div style={styles.turnField}>
                      <span style={styles.turnFieldLabel}>목표 사용자</span>
                      <p style={styles.bubbleContent}>{c.target_user}</p>
                    </div>
                    <div style={styles.turnField}>
                      <span style={styles.turnFieldLabel}>핵심 가치</span>
                      <p style={styles.bubbleContent}>{c.core_value}</p>
                    </div>
                    {c.main_features?.length > 0 && (
                      <div style={styles.turnField}>
                        <span style={styles.turnFieldLabel}>주요 기능</span>
                        <ul style={styles.turnList}>
                          {c.main_features.map((f, i) => <li key={i}>{f}</li>)}
                        </ul>
                      </div>
                    )}
                    <div style={styles.turnField}>
                      <span style={styles.turnFieldLabel}>실현 가능성</span>
                      <p style={styles.bubbleContent}>{FEASIBILITY_LABEL[c.feasibility] || c.feasibility || '미상'}</p>
                    </div>
                    <div style={styles.turnField}>
                      <span style={styles.turnFieldLabel}>공모전 적합성</span>
                      <p style={styles.bubbleContent}>{c.contest_fit}</p>
                    </div>
                    {c.risks?.length > 0 && (
                      <div style={styles.turnField}>
                        <span style={styles.turnFieldLabel}>주요 위험</span>
                        <ul style={styles.turnList}>
                          {c.risks.map((r, i) => <li key={i}>{r}</li>)}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={styles.chatArea} ref={scrollRef}>
            {!result && <p style={styles.muted}>왼쪽에서 공모전을 입력하고 회의를 시작하세요.</p>}
            {result?.messages.map((m) =>
              m.content?.trim() ? (
                <MessageBubble key={m.message_id} message={m} />
              ) : (
                <p key={m.message_id} style={styles.error}>
                  {(SPEAKER_META[m.speaker_id]?.label || m.speaker_name)}의 응답을 만드는 중 오류가 발생했습니다.
                </p>
              ),
            )}
            {(sending || finalizing) && (
              <div style={styles.loadingRow}>
                <span style={styles.loadingDot} /> LLM 응답을 기다리는 중...
              </div>
            )}
          </div>

          {error && <p style={styles.error}>{error}</p>}

          {result?.phase === 'finalized' && result.idea_proposal && (
            <div style={styles.proposalBox}>
              <div style={styles.sectionLabel}>최종 제안서</div>
              <h3 style={styles.proposalTitle}>{result.idea_proposal.idea_name}</h3>
              <p style={styles.bubbleContent}>{result.idea_proposal.one_line_pitch}</p>
              <div style={styles.turnField}>
                <span style={styles.turnFieldLabel}>문제 정의</span>
                <p style={styles.bubbleContent}>{result.idea_proposal.problem_definition}</p>
              </div>
              <div style={styles.turnField}>
                <span style={styles.turnFieldLabel}>핵심 기능</span>
                <ul style={styles.turnList}>
                  {result.idea_proposal.key_features?.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>
              <div style={styles.turnField}>
                <span style={styles.turnFieldLabel}>다음 작업</span>
                <ul style={styles.turnList}>
                  {result.idea_proposal.next_actions?.map((a, i) => <li key={i}>{a}</li>)}
                </ul>
              </div>
            </div>
          )}

          {result && result.phase !== 'finalized' && (
            <div style={styles.inputRow}>
              <input
                style={styles.chatInput}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && canReply && handleSend()}
                placeholder={canReply ? '답변을 입력하세요' : '전문가 응답을 기다리는 중입니다'}
                disabled={!canReply}
              />
              <button style={styles.sendButton} onClick={handleSend} disabled={!canReply || !draft.trim()}>
                전송
              </button>
            </div>
          )}
        </main>

        {/* 오른쪽: 요약 */}
        <aside style={styles.rightPanel}>
          <div style={styles.panelTitle}>현재 주제 후보</div>
          <p style={styles.muted}>{userIdea || '아직 입력되지 않았습니다.'}</p>

          <div style={styles.panelTitle}>합의 사항</div>
          {result?.consensus?.length ? (
            <ul style={styles.turnList}>{result.consensus.map((c, i) => <li key={i}>{c}</li>)}</ul>
          ) : (
            <p style={styles.muted}>아직 없습니다.</p>
          )}

          <div style={styles.panelTitle}>미해결 쟁점</div>
          {result?.unresolved_issues?.length ? (
            <ul style={styles.turnList}>{result.unresolved_issues.map((u, i) => <li key={i}>{u}</li>)}</ul>
          ) : (
            <p style={styles.muted}>아직 없습니다.</p>
          )}

          <button style={styles.finalizeButton} onClick={handleFinalize} disabled={!canFinalize}>
            {finalizing ? '초안 생성 중...' : '주제 확정하고 초안 받기'}
          </button>
          {!canFinalize && result && result.phase !== 'finalized' && (
            <p style={styles.muted}>두 전문가의 첫 논의 라운드가 끝나야 확정할 수 있습니다.</p>
          )}

          {result && (
            <>
              <button style={styles.rawToggle} onClick={() => setShowRaw((v) => !v)}>
                {showRaw ? '원본 JSON 숨기기' : '원본 JSON 보기'}
              </button>
              {showRaw && <pre style={styles.rawBox}>{JSON.stringify(result, null, 2)}</pre>}
            </>
          )}
        </aside>
      </div>
    </div>
  )
}

const styles = {
  page: { maxWidth: 1280, margin: '0 auto', padding: '24px 20px', color: '#1f2333' },
  title: { fontSize: 22, fontWeight: 700, margin: '0 0 6px' },
  subtitle: { fontSize: 13, color: '#8b8fa3', marginBottom: 20 },
  layout: { display: 'grid', gridTemplateColumns: '260px 1fr 280px', gap: 16, alignItems: 'start' },
  leftPanel: { display: 'flex', flexDirection: 'column', gap: 6, position: 'sticky', top: 16 },
  centerPanel: { display: 'flex', flexDirection: 'column', gap: 10, minHeight: 480 },
  rightPanel: { display: 'flex', flexDirection: 'column', gap: 6, position: 'sticky', top: 16 },
  panelTitle: { fontSize: 12.5, fontWeight: 700, color: '#4b4f63', marginTop: 14 },
  stepItem: { display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5, color: '#4b4f63' },
  stepDot: { width: 8, height: 8, borderRadius: '50%', display: 'inline-block' },
  label: { fontSize: 12, fontWeight: 600, color: '#4b4f63', marginTop: 6 },
  input: { border: '1.5px solid #e5e3f0', borderRadius: 8, padding: '8px 10px', fontSize: 13, font: 'inherit' },
  textarea: {
    border: '1.5px solid #e5e3f0',
    borderRadius: 8,
    padding: '8px 10px',
    fontSize: 13,
    font: 'inherit',
    resize: 'vertical',
  },
  startButton: {
    marginTop: 14,
    padding: '10px 0',
    borderRadius: 10,
    border: 'none',
    background: '#7c4dff',
    color: '#fff',
    fontSize: 13.5,
    fontWeight: 700,
    cursor: 'pointer',
  },
  statusBar: { display: 'flex', gap: 8 },
  statusBadge: {
    fontSize: 11.5,
    fontWeight: 600,
    color: '#4b4f63',
    background: '#eef0f7',
    padding: '5px 10px',
    borderRadius: 999,
  },
  chatArea: {
    flex: 1,
    minHeight: 380,
    maxHeight: 560,
    overflowY: 'auto',
    border: '1px solid #ece9f7',
    borderRadius: 12,
    padding: 14,
    background: '#fafafe',
  },
  bubbleRow: { display: 'flex', marginBottom: 10 },
  bubble: { maxWidth: '80%', border: '1.5px solid', borderRadius: 12, padding: 10 },
  bubbleHeader: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 },
  speakerBadge: { fontSize: 11, fontWeight: 700, color: '#fff', padding: '2px 8px', borderRadius: 999 },
  bubbleMeta: { fontSize: 11, color: '#8b8fa3' },
  bubbleContent: { fontSize: 13.5, lineHeight: 1.6, margin: 0, whiteSpace: 'pre-wrap' },
  evidenceBox: { marginTop: 6, borderTop: '1px dashed #e5e3f0', paddingTop: 6 },
  evidenceItem: { fontSize: 11.5, color: '#8b8fa3' },
  detailToggle: {
    marginTop: 6,
    border: 'none',
    background: 'none',
    color: '#7c4dff',
    fontWeight: 600,
    cursor: 'pointer',
    fontSize: 11.5,
    padding: 0,
    textAlign: 'left',
  },
  detailBox: { marginTop: 6, borderTop: '1px dashed #e5e3f0', paddingTop: 6, display: 'flex', flexDirection: 'column', gap: 6 },
  loadingRow: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5, color: '#8b8fa3' },
  loadingDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    background: '#7c4dff',
    display: 'inline-block',
    animation: 'none',
  },
  muted: { color: '#8b8fa3', fontSize: 12.5 },
  error: { color: '#d64545', fontSize: 13 },
  inputRow: { display: 'flex', gap: 8 },
  chatInput: {
    flex: 1,
    border: '1.5px solid #e5e3f0',
    borderRadius: 10,
    padding: '10px 12px',
    fontSize: 13.5,
    font: 'inherit',
  },
  sendButton: {
    padding: '0 18px',
    borderRadius: 10,
    border: 'none',
    background: '#2f6fed',
    color: '#fff',
    fontSize: 13.5,
    fontWeight: 700,
    cursor: 'pointer',
  },
  finalizeButton: {
    marginTop: 14,
    padding: '10px 0',
    borderRadius: 10,
    border: 'none',
    background: '#00897b',
    color: '#fff',
    fontSize: 13,
    fontWeight: 700,
    cursor: 'pointer',
  },
  rawToggle: {
    marginTop: 14,
    border: 'none',
    background: 'none',
    color: '#7c4dff',
    fontWeight: 600,
    cursor: 'pointer',
    fontSize: 12,
    padding: 0,
    textAlign: 'left',
  },
  rawBox: {
    marginTop: 8,
    background: '#1f2333',
    color: '#e5e3f0',
    borderRadius: 10,
    padding: 12,
    fontSize: 11,
    overflowX: 'auto',
    maxHeight: 320,
  },
  proposalBox: { background: '#fff', border: '1px solid #ece9f7', borderRadius: 12, padding: 16 },
  sectionLabel: { fontSize: 12.5, fontWeight: 700, color: '#4b4f63', marginBottom: 6 },
  proposalTitle: { fontSize: 16, fontWeight: 700, margin: '0 0 4px' },
  turnField: { marginTop: 8 },
  turnFieldLabel: { fontSize: 11.5, fontWeight: 700, color: '#8b8fa3' },
  turnList: { margin: '4px 0 0', paddingLeft: 18, fontSize: 12.5, lineHeight: 1.6 },
  modeBadge: { background: '#f3e9ff', color: '#7c4dff' },
  candidatesBox: {
    background: '#fff',
    border: '1px solid #ece9f7',
    borderRadius: 12,
    padding: 16,
  },
  candidatesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: 12,
    marginTop: 4,
  },
  candidateCard: {
    border: '1.5px solid #ece9f7',
    borderRadius: 10,
    padding: 12,
    background: '#fafafe',
  },
  candidateTitle: { fontSize: 14, fontWeight: 700, marginBottom: 4 },
}
