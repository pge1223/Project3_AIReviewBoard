import { Fragment, useEffect, useRef, useState } from 'react'
import { AlertCircle, ChevronDown, ChevronUp, RefreshCw, Send, Sparkles } from 'lucide-react'
import {
  cancelIdeationConversation,
  continueIdeationExpertTurnStream,
  finalizeIdeationConversation,
  getIdeationConversation,
  replyIdeationConversation,
  replyIdeationConversationStream,
  startIdeationConversation,
  startIdeationConversationStream,
} from '../../api/ideationConversationApi'
import { getAnnouncementAnalysis, getApplicationFormAnalysis } from '../../api/documentApi'
import IdeaCanvasPanel from './IdeaCanvasPanel'
import IdeationAvatarStage from './IdeationAvatarStage'
import {
  EXPERT_RECOMMEND_MESSAGE,
  FEASIBILITY_LABEL,
  REGENERATE_MESSAGE,
  SPEAKER_META,
  buildCompetitionDocumentText,
  candidateSelectMessage,
  classifyIdeationConvError,
  competitionNameFrom,
  nextActionGuideFor,
  resolveUseRag,
  resolveRespondingToSpeakerId,
  speakerMetaFor,
  statusLabelFor,
} from './ideationConversationHelpers'
import {
  advanceDisplay,
  applyStreamEvent,
  charsPerTickFor,
  createEmptyStreamState,
  dedupeMessagesById,
  isFullyDisplayed,
  pendingCharCount,
} from './ideationStreamReducer'

// 작성자: 용준/Claude(2026-07-21)
// 목적: /board "작성 전 → 주제 발굴" 흐름의 실제 대화형 아이디어 회의 화면.
//       기존 ReviewBoardPrototype.jsx 안의 더미 IdeationScreen/IdeationResultScreen
//       (고정 문자열·setTimeout·고정 후보/결과)을 대체한다. 페르소나 로직·질문 생성·
//       답변 충분성 판정·후보 결합 해석은 전부 기존 백엔드(ai/meeting 그래프 +
//       ideation_conv_*.txt 프롬프트)가 그대로 수행하고, 이 화면은 그 결과(messages/
//       idea_candidates/phase/...)를 /board의 웜 화이트 디자인(card glass, badge,
//       btn-primary/ghost — ReviewBoardPrototype.jsx의 Shell이 정의)으로 그릴 뿐이다.
//       API 연동 방식은 IdeationConversationPreviewPage.jsx(개발용 프리뷰)를 참고했지만
//       그 화면의 디자인을 옮겨오지는 않았다.
//
//       session_id/최신 회의 결과(ideationConv)는 이 컴포넌트가 아니라 부모
//       (ReviewBoardPrototype)의 state로 관리된다 — "이전 단계로 갔다 돌아와도 회의 상태
//       유지"를 만족시키려면, 이 화면이 언마운트됐다 다시 마운트돼도(사이드바로 다른
//       단계를 갔다 오는 경우) 이미 진행 중이던 세션을 잃지 않아야 하기 때문이다. 그래서
//       "start API를 한 번만 호출"하는 가드도 두 겹이다: ① 부모가 이미 ideationConv를
//       들고 있으면(=한 번이라도 시작 성공) 이 컴포넌트는 절대 다시 start를 호출하지
//       않는다(항상 최우선으로 검사), ② 아직 없다면 이번 마운트에서 한 번만 시도하도록
//       useRef 가드를 둔다(React StrictMode의 개발 모드 이중 effect 호출에도 안전 —
//       ref는 같은 컴포넌트 인스턴스의 마운트/언마운트/재마운트 사이에 유지된다).

const REPLYABLE_PHASES = new Set([
  'awaiting_candidate_selection',
  'awaiting_planning_answer',
  'awaiting_developer_answer',
  'awaiting_user_decision',
  'discussion_complete',
])

// 재인/Claude(2026-07-23): 아바타 재생 대상 화자 - user는 당연히 제외, 그 외 3명
// (진행자/기획/개발)만 IdeationAvatarStage의 AVATAR_SLOTS에 얼굴·목소리가 등록돼 있다.
const AVATAR_SPEAKER_IDS = new Set(['ideation_facilitator', 'planning_expert', 'dev_expert'])

// 재인/Claude(2026-07-23, 실측: "선택된 아이디어... 이거는 아예 코랩에 태우지마"): 후보
// 선택 확정 직후 진행자가 말하는 이 메시지는 ai/meeting/graph/ideation_conv_discovery.py
// ::_resolve_selection이 항상 이 문구("선택된 아이디어: ...")로 시작하는 고정 템플릿으로
// 만든다(LLM 호출 없음, 회의 로직 쪽에서 이미 확인된 사실). 사용자가 방금 화면에서 고른
// 내용을 그대로 되짚어 보여주는 시스템 로그 성격이라 아바타가 "말할" 필요가 없다고 판단해
// 아바타 재생 대상에서만 제외한다(텍스트 채팅에는 그대로 보인다 - canonicalMessages는
// 안 건드림). message_type이 다른 진행자 메시지("오늘은...", 라운드 정리)와 똑같이
// "summary"라 타입으로는 구분이 안 되고, 이 접두어가 가장 안정적인 구분 방법이다.
const AVATAR_EXCLUDED_CONTENT_PREFIX = '선택된 아이디어:'

function ideationSessionStorageKey(projectId) {
  return projectId ? `ideation-conv-session:${projectId}` : null
}

// 커서 깜빡임 애니메이션 — Shell(ReviewBoardPrototype.jsx)의 전역 <style>을 건드리지 않고
// 이 컴포넌트 전용으로 한 번만 주입한다(기존 코드베이스가 Shell에서 이미 쓰는 "JSX 안에
// <style> 태그를 직접 렌더링"하는 패턴 그대로).
function StreamingCursorStyle() {
  return (
    <style>{`
      @keyframes rb-ideation-cursor-blink { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }
      .rb-ideation-cursor { display: inline-block; width: 2px; margin-left: 1px; background: currentColor; animation: rb-ideation-cursor-blink 1s step-start infinite; }
    `}</style>
  )
}

// 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — 말풍선 위
// "누구에게 응답 중인지" 작은 캡션. message.structured.responding_to_speaker_id는 이미
// 백엔드(ideation_conv_nodes.py::_responding_to_for)가 코드로 결정적으로 계산해 저장한
// 값이다(LLM이 지어낸 값이 아니다) — 여기서는 그 값을 사람이 읽을 라벨로 바꿔 보여주기만
// 한다. 값이 없으면(질문/답변/최초 발언 등) 아무것도 렌더링하지 않는다.
function RespondingToCaption({ message, allMessages = [] }) {
  const actualSpeakerId = resolveRespondingToSpeakerId(message, allMessages)
  const targetLabel = SPEAKER_META[actualSpeakerId]?.label
  if (!targetLabel) return null
  return (
    <div style={{ fontSize: 11, color: 'var(--text-2)', marginBottom: 3 }}>↳ {targetLabel}에게 응답</div>
  )
}

// 근거(evidence) 접기/펼치기 — 용준/Claude(2026-07-22, 요청: RAG 근거 실제 활용 강화)로
// message.evidence(검색·프롬프트 주입 전체) 대신 message.linked_evidence_refs(주장-근거
// 연결·관련성 검증을 통과한 chunk_id만)를 기준으로 필터링한다. injected_evidence_count가
// 아니라 linked_evidence_count가 성공 기준이라는 요청 취지를 화면에도 그대로 반영한다 —
// 검증을 통과하지 못한 근거는 여기서 아예 보이지 않는다.
function EvidenceToggle({ evidence, linkedEvidenceRefs, claims }) {
  const [open, setOpen] = useState(false)
  const linkedSet = new Set(linkedEvidenceRefs || [])
  const items = (evidence || []).filter((e) => e && e.chunk_id && linkedSet.has(e.chunk_id))
  // document_fact인데 근거 연결에 실패했거나(unsupported), expert_judgment로 남은 주장은
  // 문서 출처가 있는 것처럼 꾸미지 않고 "전문가 판단" 배지로만 구분해 보여준다.
  const unlinkedClaims = (claims || []).filter(
    (c) => c && (c.claim_type === 'expert_judgment' || !linkedSet.size)
  )
  if (items.length === 0 && unlinkedClaims.length === 0) return null
  if (items.length === 0) {
    return (
      <div
        style={{
          marginTop: 4,
          fontSize: 11.5,
          color: 'var(--text-2)',
          display: 'inline-flex',
          alignItems: 'center',
          border: '1px solid var(--glass-border)',
          borderRadius: 999,
          padding: '2px 7px',
        }}
      >
        전문가 판단 · 문서 근거 없음
      </div>
    )
  }
  const count = items.length
  return (
    <div style={{ marginTop: 4 }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          background: 'none', border: 'none', padding: 0, cursor: 'pointer',
          fontSize: 11.5, color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: 2,
        }}
      >
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        근거 {count}건 {open ? '접기' : '보기'}
      </button>
      {open && (
        <div style={{ marginTop: 4, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {items.map((e, i) => (
            <div
              key={e.chunk_id || i}
              style={{
                fontSize: 11.5, color: 'var(--text-1)', lineHeight: 1.5,
                background: 'var(--bg-0)', border: '1px solid var(--glass-border)', borderRadius: 8, padding: '6px 8px',
              }}
            >
              {e.document_name && (
                <div style={{ fontWeight: 600, color: 'var(--text-2)', marginBottom: 2 }}>
                  {e.document_name}
                  {e.page != null && ` / ${e.page}페이지`}
                </div>
              )}
              {e.section && (
                <div style={{ fontSize: 10.5, color: 'var(--text-2)', marginBottom: 2 }}>평가항목: {e.section}</div>
              )}
              {(e.text || e.quote) && <div style={{ marginBottom: 2 }}>"{e.text || e.quote}"</div>}
              {e.source_url && (
                <a href={e.source_url} target="_blank" rel="noreferrer" style={{ fontSize: 10.5, color: 'var(--purple-dim)' }}>
                  원문 보기
                </a>
              )}
            </div>
          ))}
          {items.length === 0 && (
            <div
              style={{
                fontSize: 11.5, color: 'var(--text-2)', fontStyle: 'italic',
                background: 'var(--bg-0)', border: '1px solid var(--glass-border)', borderRadius: 8, padding: '6px 8px',
              }}
            >
              현재 문서에서 직접 확인되지 않은 전문가 판단입니다.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// 진행자의 라운드 정리 메시지 아래에 붙는 작은 합의/쟁점 요약 카드. 채팅 본문(spoken_text)은
// 이미 1~2문장으로 자연스럽게 정리되어 있으므로, 이 카드는 그 문장이 어떤 근거(합의 사항·
// 남은 쟁점·사용자 결정 필요 여부)에서 나왔는지 보고 싶을 때만 펼쳐보는 보조 정보다.
function FacilitatorSummaryCard({ structured }) {
  const agreements = structured?.agreements || []
  const disagreements = structured?.disagreements || []
  if (agreements.length === 0 && disagreements.length === 0) return null
  return (
    <div
      style={{
        marginTop: 6, padding: '8px 10px', borderRadius: 10,
        background: 'var(--bg-0)', border: '1px solid var(--glass-border)', fontSize: 12,
      }}
    >
      {agreements.length > 0 && (
        <div style={{ marginBottom: disagreements.length > 0 ? 6 : 0 }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>합의 사항</strong>
          <ul style={{ margin: '2px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {agreements.map((a, i) => <li key={i}>{a}</li>)}
          </ul>
        </div>
      )}
      {disagreements.length > 0 && (
        <div>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>남은 쟁점</strong>
          <ul style={{ margin: '2px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {disagreements.map((d, i) => <li key={i}>{d}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

function MessageBubble({ message, streaming = false, interrupted = false, allMessages = [] }) {
  const meta = speakerMetaFor(message)
  const isRight = meta.align === 'right'
  // 스트리밍 중인 말풍선은 displayedContent(타이핑 큐가 드러낸 만큼)만 보여준다 —
  // content(서버에서 실제로 받은 전체 텍스트)를 그대로 쓰면 델타가 도착하는 순간
  // 문장이 통째로 튀어나와 타이핑 효과가 사라진다. canonical(완료된) 메시지는
  // displayedContent 필드가 없으므로 content를 그대로 쓴다.
  const text = streaming ? message.displayedContent ?? '' : message.content
  const hasContent = !!text?.trim()
  // done(message_end 수신)이 와도 displayedContent가 content를 따라잡기 전까지는
  // 커서를 유지한다 — "message_end가 와도 남은 글자 큐를 끝까지 표시"(요청 사항).
  const caughtUp = (message.displayedContent?.length ?? 0) >= (message.content?.length ?? 0)
  // 용준/Claude(2026-07-23, 요청: 스트리밍 UX 버그 수정) — 구조화 응답 검증 실패로
  // message_reset을 받은 스트리밍 말풍선. ideationStreamReducer.js가 더 이상 이 메시지를
  // 배열에서 지우지 않고 status:'reviewing'으로만 표시하므로, 여기서는 지우는 대신
  // 흐리게 보여주고 재검토 중이라는 라벨을 붙인다(요청: "화면이 비지 않게").
  const isReviewing = streaming && message.status === 'reviewing'
  const showCursor = streaming && !caughtUp && !isReviewing

  if (!hasContent && !streaming) {
    return (
      <div style={{ display: 'flex', justifyContent: isRight ? 'flex-end' : 'flex-start', marginBottom: 10 }}>
        <div style={{ fontSize: 12.5, color: 'var(--coral)' }}>
          <AlertCircle size={12} style={{ verticalAlign: -1, marginRight: 4 }} />
          {meta.label}의 응답을 만드는 중 오류가 발생했습니다.
        </div>
      </div>
    )
  }
  // 재인/Claude(2026-07-23, 실측: "저 빈 말풍선이 보기 싫어서 그래"): message_start가
  // 오면 스트리밍 메시지가 streamState.messages에 즉시 생기지만, 첫 delta가 도착하기
  // 전까지는 content가 빈 문자열이다 — 그 짧은 순간에 배지만 있고 속은 텅 빈 말풍선이
  // 보였다. 실제 글자가 하나라도 도착하기 전까지는 아예 아무것도 안 그린다(위 sending &&
  // streamState.messages.length === 0 상태 문구가 그 대기 시간 동안 대신 보여준다 - 다만
  // message_start 자체가 phaseLabel을 null로 리셋하므로, 이 순간엔 그 문구도 안 보이고
  // 완전히 비어 있게 된다 - 사용자가 원한 게 정확히 이거다).
  if (streaming && !hasContent) {
    return null
  }

  const isFacilitatorSummary = !streaming && message.speaker_id === 'ideation_facilitator' && message.message_type === 'summary'

  return (
    <div style={{ display: 'flex', justifyContent: isRight ? 'flex-end' : 'flex-start', marginBottom: 10 }}>
      <div style={{ maxWidth: '82%' }}>
        {meta.badgeClass && (
          <div className={`badge ${meta.badgeClass} mono`} style={{ marginBottom: 4 }}>
            {meta.label}
          </div>
        )}
        {!streaming && <RespondingToCaption message={message} allMessages={allMessages} />}
        <div
          style={{
            background: isRight ? 'var(--purple-dim)' : 'var(--bg-1)',
            border: '1px solid var(--glass-border)',
            borderRadius: 12,
            padding: '10px 14px',
            fontSize: 13.5,
            lineHeight: 1.6,
            whiteSpace: 'pre-wrap',
            opacity: interrupted || isReviewing ? 0.6 : 1,
          }}
        >
          {text}
          {showCursor && <span className="rb-ideation-cursor">▍</span>}
        </div>
        {/* 용준/Claude(2026-07-22, 요청: 중단된 메시지 처리) — 사용자가 이미 읽은 부분
            발언은 화면에 남기되, 완료된 발언과 명확히 구분한다. 이 블록은 화면 기록일
            뿐이며 agreements/decisions/resolved_issues 등 어떤 canonical 상태에도
            포함되지 않는다(서버가 애초에 저장하지 않았으므로). */}
        {interrupted && (
          <div style={{ fontSize: 11.5, color: 'var(--text-2)', marginTop: 3, fontStyle: 'italic' }}>
            [사용자에 의해 발언이 중단됐습니다]
          </div>
        )}
        {/* 용준/Claude(2026-07-23, 요청: 스트리밍 UX 버그 수정) — 검증 실패로 재검토
            대기 중임을 알린다. willRetry가 false면(재시도 없이 안전한 fallback을 기다리는
            중) 곧 회의 진행 자체가 멈추지 않는다는 것을 알 수 있게 문구를 살짝 다르게 준다. */}
        {isReviewing && (
          <div style={{ fontSize: 11.5, color: 'var(--text-2)', marginTop: 3, fontStyle: 'italic' }}>
            {message.willRetry === false ? '더 나은 답변으로 정리하고 있습니다…' : '응답을 다시 검토하고 있습니다…'}
          </div>
        )}
        {!streaming && (
          <EvidenceToggle
            evidence={message.evidence}
            linkedEvidenceRefs={message.linked_evidence_refs}
            claims={message.claims}
          />
        )}
        {isFacilitatorSummary && <FacilitatorSummaryCard structured={message.structured} />}
      </div>
    </div>
  )
}


const EXPERT_LABELS = {
  planning_expert: '기획 위원',
  dev_expert: '개발 위원',
}

function InterruptionMarker({ speakerId }) {
  const speakerLabel = EXPERT_LABELS[speakerId]
  return (
    <p
      style={{
        margin: '4px 0 8px',
        color: 'var(--text-2)',
        fontSize: 12,
        fontStyle: 'italic',
      }}
    >
      [{speakerLabel ? `${speakerLabel}의 ` : ''}발언이 사용자에 의해 중단됐습니다]
    </p>
  )
}

function CandidateCard({ candidate, index, onSelect, disabled }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="card glass" style={{ marginBottom: 10, padding: 14 }}>
      <div style={{ fontSize: 11.5, color: 'var(--text-2)', marginBottom: 2 }}>후보 {index + 1}</div>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 6 }}>{candidate.title}</div>
      <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.6, marginBottom: 4 }}>
        <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>해결할 문제 · </strong>
        {candidate.problem}
      </div>
      <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.6, marginBottom: 4 }}>
        <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>목표 사용자 · </strong>
        {candidate.target_user}
      </div>

      <button
        className="btn-ghost"
        style={{ padding: '4px 10px', fontSize: 11.5, marginTop: 4, marginBottom: expanded ? 8 : 0, display: 'flex', alignItems: 'center', gap: 4 }}
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        {expanded ? '간단히 보기' : '상세 보기'}
      </button>

      {expanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 6 }}>
          <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.6 }}>
            <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>핵심 가치 · </strong>
            {candidate.core_value}
          </div>
          {candidate.main_features?.length > 0 && (
            <div style={{ fontSize: 12.5, color: 'var(--text-1)' }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>주요 기능</strong>
              <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.7 }}>
                {candidate.main_features.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </div>
          )}
          <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.6 }}>
            <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>공모전 적합성 · </strong>
            {candidate.contest_fit || '확인되지 않음'}
          </div>
          {candidate.risks?.length > 0 && (
            <div style={{ fontSize: 12.5, color: 'var(--text-1)' }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>주요 위험</strong>
              <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.7 }}>
                {candidate.risks.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}

      <div style={{ fontSize: 11.5, color: 'var(--text-2)', marginBottom: 10 }}>
        실현 가능성 {FEASIBILITY_LABEL[candidate.feasibility] || '미상'}
      </div>

      <button className="btn-primary" style={{ width: '100%', padding: '8px 0', fontSize: 12.5 }} disabled={disabled} onClick={() => onSelect(index)}>
        이 후보 선택
      </button>
    </div>
  )
}

function MergeAnalysisPanel({ mergeAnalysis, sourceCandidates, userSelectionMessage }) {
  if (!mergeAnalysis) return null
  return (
    <div className="card glass" style={{ marginBottom: 12, padding: 14 }}>
      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>결합 분석</div>
      {userSelectionMessage && (
        <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 8 }}>
          원문 요청 · “{userSelectionMessage}”
        </div>
      )}
      {sourceCandidates?.length > 0 && (
        <div style={{ fontSize: 12.5, color: 'var(--text-1)', marginBottom: 8 }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>원본 후보 · </strong>
          {sourceCandidates.map((c) => c.title).join(' · ')}
        </div>
      )}
      <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.6, marginBottom: 4 }}>
        <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>공통 문제 · </strong>{mergeAnalysis.common_problem}
      </div>
      <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.6, marginBottom: 4 }}>
        <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>공통 가치 · </strong>{mergeAnalysis.common_value}
      </div>
      <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.6, marginBottom: 4 }}>
        <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>결합 적합도 · </strong>
        {FEASIBILITY_LABEL[mergeAnalysis.fit] || mergeAnalysis.fit || '미상'}
      </div>
      {mergeAnalysis.primary_features?.length > 0 && (
        <div style={{ fontSize: 12.5, color: 'var(--text-1)', marginBottom: 4 }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>주 기능</strong>
          <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {mergeAnalysis.primary_features.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      )}
      {mergeAnalysis.secondary_features?.length > 0 && (
        <div style={{ fontSize: 12.5, color: 'var(--text-1)', marginBottom: 4 }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>보조 기능</strong>
          <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {mergeAnalysis.secondary_features.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      )}
      {mergeAnalysis.conflicts?.length > 0 && (
        <div style={{ fontSize: 12.5, color: 'var(--text-1)', marginBottom: 4 }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>충돌 지점</strong>
          <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {mergeAnalysis.conflicts.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      )}
      {mergeAnalysis.open_questions?.length > 0 && (
        <div style={{ fontSize: 12.5, color: 'var(--text-1)' }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>미확정 사항</strong>
          <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {mergeAnalysis.open_questions.map((q, i) => <li key={i}>{q}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

function ErrorBanner({ error, onRetry }) {
  if (!error) return null
  return (
    <div
      className="card glass"
      style={{ borderColor: 'var(--coral-dim)', marginBottom: 14, display: 'flex', alignItems: 'flex-start', gap: 10, padding: 14 }}
    >
      <AlertCircle size={16} color="var(--coral)" style={{ marginTop: 1, flexShrink: 0 }} />
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, color: 'var(--text-0)', lineHeight: 1.6 }}>{error.message}</div>
        {(error.code || error.failedNode) && (
          <div style={{ marginTop: 5, fontSize: 11.5, color: 'var(--text-2)', fontFamily: 'var(--mono)' }}>
            {error.code && `오류 코드: ${error.code}`}
            {error.code && error.failedNode && ' · '}
            {error.failedNode && `실패 노드: ${error.failedNode}`}
          </div>
        )}
        {onRetry && (
          <button className="btn-ghost" style={{ marginTop: 10, padding: '6px 12px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }} onClick={onRetry}>
            <RefreshCw size={12} /> 다시 시도
          </button>
        )}
      </div>
    </div>
  )
}

export function IdeationScreen({
  projectId,
  criteriaDocuments,
  ideationConv,
  setIdeationConv,
  onFinalized,
  onBack,
  saving = false,
  saveError = '',
}) {
  const [starting, setStarting] = useState(!ideationConv)
  const [sending, setSending] = useState(false)
  const [finalizing, setFinalizing] = useState(false)
  const [error, setError] = useState(null)
  const [draft, setDraft] = useState('')
  // 용준/Claude(2026-07-21, 요청: 실시간 스트리밍) — 지금 스트리밍 중인(아직 canonical이
  // 아닌) 메시지만 별도로 들고 있는다. 서버가 최종 state 이벤트를 보내면 이 값은 통째로
  // createEmptyStreamState()로 비우고 ideationConv(canonical)만 그린다 — 그래서 스트리밍
  // 미리보기와 canonical 메시지가 동시에 화면에 남아 중복되는 경우가 구조적으로 없다.
  const [streamState, setStreamState] = useState(() => createEmptyStreamState())
  // 가은/Claude(2026-07-22, 요청: 회의 시작 대기 체감 개선 1단계): /start/stream이 흘려주는
  // 진행 문구("아이디어 후보를 만들고 있습니다" -> "후보의 실현 가능성을 검토하고 있습니다").
  // 비어 있으면 기존 고정 문구를 보여준다.
  const [startPhaseLabel, setStartPhaseLabel] = useState('')
  // 용준/Claude(2026-07-22, 요청: "잠시만" 버튼) — 취소 확인을 기다리는 동안(백엔드에 취소
  // 신호를 보내고 세션 락이 실제로 풀릴 때까지) true. 이 사이에는 새 reply를 보내지 않는다
  // (요청: "취소 완료 전에 새 reply를 보내 세션 lock 409가 발생하지 않게").
  const [interrupting, setInterrupting] = useState(false)
  // null이면 평소 입력창, 문자열("planning_expert"/"dev_expert"/"both")이면 그 위원에게
  // 보낼 질문을 입력받는 중 — 대상 선택 버튼 대신 안내 문구가 바뀐 입력창을 보여준다.
  const [interjectTarget, setInterjectTarget] = useState(null)
  // 중단된 발언자와 사용자가 의견을 남길 대상은 별개다. 예를 들어 개발 위원 발언 중
  // 멈춘 뒤에도 기획 위원의 직전 의견을 선택할 수 있으므로 두 값을 각각 보존한다.
  const [interruptedSpeakerId, setInterruptedSpeakerId] = useState(null)
  // 취소된 스트리밍 본문은 저장하지 않고 중단 시점 마커만 로컬에 보존한다. 서버가 저장하지
  // 않은 미완성 발언은 ideationConv(canonical)나 다음 프롬프트에 절대 섞이지 않는다.
  const [interruptionMarkers, setInterruptionMarkers] = useState([])
  // 재인/Claude(2026-07-23, 실측: "내가 보낸 채팅이 바로 안 올라가고 늦게 올라감"):
  // 사용자가 보낸 메시지는 서버가 canonical state에 echo해서 돌려줄 때(finalizeStream)까지
  // 화면 어디에도 안 보였다 — 그 사이(위원 응답을 실제로 생성하는 몇 초)에는 입력창에
  // 그대로 남아있는 것처럼 보였다. 보내는 즉시 이 로컬 전용 임시 말풍선으로 먼저 보여주고,
  // 진짜 canonical 메시지가 도착하면(finalizeStream/sendNonStreaming) 즉시 지운다 —
  // interruptionMarkers와 같은 원칙: 서버 state에는 절대 안 섞이는 순수 화면 표시용.
  const [optimisticUserMessage, setOptimisticUserMessage] = useState(null)
  // 가은/Claude(2026-07-22): 아이디어 기획 캔버스(IdeaCanvasPanel)의 "심사기준 대응 포인트"
  // 시드용. runStart는 원래 이 분석을 start API 페이로드를 만드는 데만 쓰고 버렸는데,
  // 캔버스가 계속 보여줘야 하므로 상태로 유지한다. 세션 재개(resume) 경로에서는 start를
  // 다시 부르지 않으므로 거기서도 별도로 채운다.
  const [announcementAnalysis, setAnnouncementAnalysis] = useState(null)
  // 재인/Claude(2026-07-23): 아이디어 회의 아바타 연동 — canonical 메시지 중 아바타
  // 3명(진행자/기획/개발)에 해당하는 것만 골라 재생 큐로 넘긴다. 이미 큐에 넣은
  // message_id는 queuedAvatarIdsRef로 추적해서, 메시지 목록이 리렌더될 때마다 같은
  // 발언을 중복으로 다시 큐잉하지 않게 막는다.
  const [avatarPlayQueue, setAvatarPlayQueue] = useState([])
  const queuedAvatarIdsRef = useRef(new Set())
  // 재인/Claude(2026-07-23): handleAvatarNeedNextSpeaker가 지금 진행 중인 continue-turn
  // fetch를 추적한다 — "잠시만"이 그 사이에 눌리면 abort()로 끊어서, 이미 중단한 뒤에
  // 뒤늦게 도착하는 응답이 canonical state를 다시 덮어쓰지 않게 막는다(handleInterject
  // 참고). handleAvatarNeedNextSpeaker 자신이 sending/interrupting/interjectTarget이 걸려
  // 있으면 애초에 새로 시작하지 않으므로, 세션 락 409 자체는 이 ref 없이도 이미 피한다 —
  // 이 ref는 그 이후(이미 시작된 호출)에 대한 정리용이다.
  const avatarTurnAbortRef = useRef(null)

  const startedRef = useRef(false)
  const chatEndRef = useRef(null)
  const streamAbortRef = useRef(null)
  const inputRef = useRef(null)
  // 스트리밍 엔드포인트가 비활성화(404)로 확인되면 이 세션 동안은 다시 시도하지 않고
  // 동기식 API로만 보낸다(요청: "플래그가 꺼져 있으면 기존 비스트리밍 API로 돌아갈 수
  // 있도록"). 가은/Claude(2026-07-22, 요청: 안내 배너 제거): 전환 사실은 이제 화면 배너
  // 없이 console.warn으로만 남긴다 — 스트리밍 기본값이 켜진 뒤로(backend/app/config.py::
  // ENABLE_IDEATION_STREAMING=True) 이 전환은 개발자가 의도적으로 플래그를 껐을 때만
  // 일어나므로, 일반 사용자에게 설정 안내 문구를 보여줄 이유가 없어졌다.
  const streamingSupportedRef = useRef(true)
  // 네트워크로부터 최종 'state'(또는 'error') 이벤트를 이미 받았지만, 아직 화면 타이핑이
  // 그 텍스트를 다 따라잡지 못해 canonical로 교체를 미루고 있는 상태를 담는다. ref인
  // 이유: 이 값 자체는 화면에 아무것도 그리지 않으므로 리렌더를 유발할 필요가 없고,
  // 아래 rAF 루프가 매 프레임 읽기만 하면 된다.
  const pendingFinalRef = useRef(null)
  // 방어 코드(요청: "중복 버그" 진단) — 실제 원인은 서버 쪽에서 재현하지 못했다(계획 문서
  // 1번 참고). 남은 유력 용의점은 이 rAF 루프 effect가 (StrictMode 이중 호출 등으로) 두 번
  // 동시에 도는 경우다 — 이 ref로 같은 컴포넌트 인스턴스에서 루프가 항상 하나만 돌게 막는다.
  const rafLoopActiveRef = useRef(false)

  useEffect(() => {
    return () => {
      // 요청: "컴포넌트 unmount 시 요청 취소".
      streamAbortRef.current?.abort()
      pendingFinalRef.current = null
    }
  }, [])

  // 실제 LLM 델타가 도착하는 즉시 content(수신 텍스트)는 이미 갱신돼 있다 — 이 루프는
  // "화면에 보여주는 속도"만 조절한다(요청: "requestAnimationFrame 또는 짧은 타이머로
  // 큐에서 1~2글자씩 displayedText에 추가"). sending이 true인 동안 계속 돌며, 네트워크가
  // 끝나 pendingFinalRef가 채워져도 화면 타이핑이 content를 다 따라잡을 때까지는 계속
  // 돈다 — 다 따라잡은 순간에만 canonical state로 교체한다(요청: "최종 state가 먼저
  // 도착해도 임시 스트림 메시지를 즉시 삭제하지 않음").
  useEffect(() => {
    if (!sending) return
    if (rafLoopActiveRef.current) return // 이미 다른 루프가 돌고 있으면 두 번째 루프를 시작하지 않는다.
    rafLoopActiveRef.current = true
    let rafId
    let cancelled = false

    function finalizeStream(finalState, errorEvent) {
      try {
        // state와 error가 함께 와도 canonical state를 먼저 보존해야 실제 오류 코드와
        // failed_node를 화면에 표시하고 동일 입력으로 다시 시작할 수 있다.
        if (finalState) {
          setIdeationConv(finalState)
          setDraft('')
          // 재인/Claude(2026-07-23, 실측: "기획위원 메시지가 2개 뜨고 립싱크 중간에 채팅이
          // 올라갔다 사라짐"): tick()이 이 함수를 부르기 직전에 streamState를 이미
          // createEmptyStreamState()로 비웠지만(위 tick() 참고), 그건 별도의
          // setStreamState 호출/렌더 커밋에 걸려 있어서 이 setIdeationConv와 정확히 같은
          // 타이밍에 화면에 반영된다는 보장이 없다 — 그 틈에 옛 스트리밍 임시 메시지와
          // 방금 확정된 canonical 메시지가 한 프레임이라도 동시에 보이면 중복으로 렌더링
          // 된다. canonical을 실제로 바꾸는 이 지점에서 다시 한번 명시적으로 비워
          // "canonical이 바뀌는 순간 streamState도 반드시 함께 비워진다"를 타이밍에
          // 의존하지 않고 보장한다.
          setStreamState(createEmptyStreamState())
        }
        if (errorEvent) {
          setError({
            ...classifyIdeationConvError(new Error(errorEvent.message)),
            code: errorEvent.code,
            failedNode: errorEvent.failed_node,
          })
        } else if (!finalState) {
          // state/error 이벤트를 하나도 못 받은 채 스트림이 끝났다(연결이 조기 종료된 경우
          // 등) — 회의 화면은 유지하되 무엇이 잘못됐는지 알린다.
          setError(classifyIdeationConvError(new Error('스트리밍 응답이 완료되지 않았습니다.')))
        }
      } finally {
        // 성공·서버 실패·연결 실패 어느 경로에서도 입력창이 sending 상태에 갇히지 않는다.
        // optimisticUserMessage도 마찬가지 — 성공하면 finalState.messages에 진짜 echo가
        // 들어있어 더 필요 없고, 실패해도 화면에 유령 말풍선이 영원히 남으면 안 된다.
        setOptimisticUserMessage(null)
        setSending(false)
      }
    }

    function tick() {
      if (cancelled) return
      setStreamState((prev) => {
        const pending = pendingCharCount(prev)
        const next = pending > 0 ? advanceDisplay(prev, charsPerTickFor(pending)) : prev
        if (pendingFinalRef.current && isFullyDisplayed(next)) {
          const { finalState, errorEvent } = pendingFinalRef.current
          pendingFinalRef.current = null
          // setState 업데이터 함수 안에서 다른 컴포넌트 상태를 직접 바꾸면 안 되므로
          // (React 경고 대상), 렌더 커밋 이후로 미룬다.
          queueMicrotask(() => finalizeStream(finalState, errorEvent))
          return createEmptyStreamState()
        }
        return next
      })
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)

    return () => {
      cancelled = true
      if (rafId) cancelAnimationFrame(rafId)
      rafLoopActiveRef.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sending])

  async function runStart() {
    setStarting(true)
    setError(null)
    setStartPhaseLabel('')
    try {
      let analysis = { has_announcement: false }
      if (projectId) {
        analysis = await getAnnouncementAnalysis(projectId)
      }
      setAnnouncementAnalysis(analysis)

      // 가은/Claude(2026-07-22, 요청: 업로드 영역 통합) — 신청서 양식 전용 문서 목록이 따로
      // 없다("공모전 공고 · 평가기준 · 신청서 양식"을 EntryScreen의 한 업로드 영역에서 함께
      // 올린다). criteriaDocuments가 준비된 상태(색인 완료 또는 확인 필요)일 때만 조회한다
      // — 신청서 양식을 실제로 안 올렸으면 백엔드가 items를 빈 배열로 돌려줄 뿐이다.
      // 실패해도(네트워크 오류 등) 회의 시작 자체를 막지 않는다 — "참고 자료"일 뿐 필수가
      // 아니기 때문이다.
      let applicationFormItems = []
      const hasReadyCriteriaDoc = criteriaDocuments.some((doc) => doc.status === 'done' || doc.status === 'warning')
      if (projectId && hasReadyCriteriaDoc) {
        try {
          const formAnalysis = await getApplicationFormAnalysis(projectId)
          applicationFormItems = formAnalysis.items || []
        } catch (err) {
          console.warn('[ideation-conv] 신청양식 항목 조회에 실패해 항목 없이 회의를 시작합니다.', err)
        }
      }

      const payload = {
        competitionName: competitionNameFrom(analysis),
        competitionDocument: buildCompetitionDocumentText(analysis),
        userIdea: '', // discovery 모드로 시작해야 하므로 반드시 빈 문자열로 보낸다.
        maxRounds: 3,
        useRag: resolveUseRag(projectId, criteriaDocuments),
        projectId,
        applicationFormItems,
      }

      // 가은/Claude(2026-07-22, 요청: 회의 시작 대기 체감 개선 1단계): 시작도 답장과 같은
      // 스트리밍 통로를 먼저 시도한다 — 후보 생성(10~30초+) 동안 고정 문구 대신 실제 진행
      // 단계(phase 이벤트)가 표시된다. 스트리밍이 꺼져 있으면(404) 기존 동기식 start로
      // 조용히 폴백하되 console.warn은 남긴다(reply 쪽과 같은 정책).
      let data = null
      if (streamingSupportedRef.current) {
        try {
          let finalState = null
          let streamError = null
          await startIdeationConversationStream(payload, {
            onEvent: (event) => {
              if (event.type === 'phase') setStartPhaseLabel(event.label || '')
              else if (event.type === 'state') finalState = event.state
              else if (event.type === 'error') streamError = event
            },
          })
          if (streamError) throw new Error(streamError.message || '대화형 회의 시작 중 오류가 발생했습니다.')
          if (!finalState) throw new Error('스트리밍이 최종 결과 없이 종료되었습니다. 다시 시도해 주세요.')
          data = finalState
        } catch (err) {
          if (classifyIdeationConvError(err).type !== 'disabled') throw err
          streamingSupportedRef.current = false
          console.warn(
            '[ideation-stream] POST /start/stream 이 비활성화(404) 응답을 반환해 동기식 /start로 전환합니다. ' +
              '백엔드 backend/.env의 ENABLE_IDEATION_STREAMING 값을 확인하세요.',
            err,
          )
        }
      }
      if (!data) {
        data = await startIdeationConversation(payload)
      }
      setIdeationConv(data)
      const key = ideationSessionStorageKey(projectId)
      if (key) sessionStorage.setItem(key, data.session_id)
    } catch (err) {
      setError(classifyIdeationConvError(err))
    } finally {
      setStarting(false)
      setStartPhaseLabel('')
    }
  }

  // 부모(ReviewBoardPrototype)가 이미 진행 중인 회의 결과를 들고 있으면(다른 단계로
  // 갔다 돌아온 경우) 절대 다시 시작하지 않는다. 처음 진입할 때만, 그리고 이번 마운트에서
  // 딱 한 번만 시도한다.
  useEffect(() => {
    if (ideationConv || startedRef.current) return
    startedRef.current = true

    const key = ideationSessionStorageKey(projectId)
    const savedSessionId = key ? sessionStorage.getItem(key) : null
    if (!savedSessionId) {
      runStart()
      return
    }
    setStarting(true)
    if (projectId) {
      getAnnouncementAnalysis(projectId).then(setAnnouncementAnalysis).catch(() => {})
    }
    getIdeationConversation(savedSessionId)
      .then((data) => setIdeationConv(data))
      .catch(() => {
        if (key) sessionStorage.removeItem(key)
        return runStart()
      })
      .finally(() => setStarting(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
    // 화면에 실제로 드러난 글자 수(displayedContent)가 늘어날 때(타이핑 진행)마다도
    // 스크롤해야 하므로, 배열 참조 자체가 아니라 지금까지 누적된 총 글자 수를 의존값으로
    // 쓴다(요청: "delta가 들어올 때 자동 스크롤" — content가 아니라 실제 화면 표시 기준).
  }, [ideationConv?.messages?.length, streamState.messages.reduce((n, m) => n + (m.displayedContent?.length || 0), 0)])

  // 재인/Claude(2026-07-23): canonical 메시지(ideationConv.messages)에 새로 추가된
  // 아바타 대상(진행자/기획/개발) 발언만 골라 재생 큐에 넘긴다. streamState(아직
  // 타이핑 중인 임시 메시지)가 아니라 canonical만 보는 이유: 아바타는 "확정된 발언"만
  // 말해야 한다 — 중단(잠시만)된 발언은 canonical에 절대 안 들어오므로(handleInterject
  // 참고) 이 큐에도 자동으로 안 들어온다.
  useEffect(() => {
    const messages = ideationConv?.messages || []
    const newItems = []
    for (const m of messages) {
      if (!AVATAR_SPEAKER_IDS.has(m.speaker_id) || queuedAvatarIdsRef.current.has(m.message_id)) continue
      queuedAvatarIdsRef.current.add(m.message_id)
      // 아바타는 안 태우지만(코랩 요청 자체를 안 보냄) 텍스트 채팅에는 그대로 남는다 —
      // canonicalMessages는 이 필터와 무관하게 ideationConv.messages를 그대로 쓴다.
      if ((m.content || '').startsWith(AVATAR_EXCLUDED_CONTENT_PREFIX)) continue
      newItems.push({ speakerId: m.speaker_id, text: m.content })
    }
    if (newItems.length > 0) {
      setAvatarPlayQueue((prev) => [...prev, ...newItems])
    }
  }, [ideationConv?.messages])

  // 재인/Claude(2026-07-23): 아바타 재생 도중 "재생 끝나기 3초 전" 타이머가 울릴 때
  // 호출된다. 새 사용자 발언 없이 다음 위원(기획/개발) 발언 1건만 더 요청한다
  // (continue-turn/stream — 백엔드 continue_ideation_expert_turn, ai/meeting/graph 쪽
  // 회의 로직은 전혀 안 건드리고 "언제 다음 턴을 요청하느냐"만 다루는 함수). 응답이 오면
  // canonical state를 그대로 갱신하고, 위의 큐잉 effect(ideationConv?.messages 감시)가
  // 새 메시지를 자동으로 avatarPlayQueue에 넣어 다음 위원 재생으로 이어진다.
  //
  // sending/interrupting/interjectTarget 중 하나라도 걸려 있으면 부르지 않는다 — 사용자가
  // 직접 reply/interject를 보내는 중이면 같은 세션에 동시 요청을 보내 백엔드 세션 락
  // 409를 유발할 수 있다(호출 자체를 막는 게 가장 안전 — 실패해도 치명적이지 않지만,
  // 아바타가 조용히 멈추는 것보다 애초에 안 보내는 편이 낫다). phase가 이미
  // "expert_discussion"이 아니면(라운드가 이미 끝났거나 사용자가 먼저 답해 다음 라운드로
  // 넘어간 경우) 마찬가지로 부르지 않는다 — 백엔드도 같은 조건을 다시 검사하지만, 프론트
  // 에서 먼저 걸러야 불필요한 요청/에러 로그를 줄인다.
  async function handleAvatarNeedNextSpeaker(finishedSpeakerId) {
    console.log('[avatar-debug] handleAvatarNeedNextSpeaker CALLED', {
      finishedSpeakerId,
      hasSessionId: !!ideationConv?.session_id,
      phase: ideationConv?.phase,
      sending,
      interrupting,
      interjectTarget,
    })
    if (!ideationConv?.session_id) {
      console.log('[avatar-debug] bail: no session_id')
      return
    }
    if (ideationConv.phase !== 'expert_discussion') {
      console.log('[avatar-debug] bail: phase !== expert_discussion', ideationConv.phase)
      return
    }
    if (sending || interrupting || interjectTarget) {
      console.log('[avatar-debug] bail: sending/interrupting/interjectTarget guard', { sending, interrupting, interjectTarget })
      return
    }

    console.log('[avatar-debug] calling continueIdeationExpertTurnStream', { sessionId: ideationConv.session_id })
    const controller = new AbortController()
    avatarTurnAbortRef.current = controller
    try {
      await continueIdeationExpertTurnStream(ideationConv.session_id, {
        signal: controller.signal,
        onEvent: (event) => {
          console.log('[avatar-debug] continue-turn event', event.type, event)
          if (event.type === 'state') {
            setIdeationConv(event.state)
            // 재인/Claude(2026-07-23, 실측: "기획위원 발언이 화면에 2번 뜸"): 이 canonical
            // 갱신은 handleSend의 타이핑 효과 루프(streamState/pendingFinalRef/tick)를 전혀
            // 거치지 않는다 - 그쪽이 아직 스트리밍 임시 메시지를 못 지운 상태(sending이
            // 막 false로 바뀐 직후처럼)에서 이 이벤트가 오면, 같은 발언이 streamState
            // (임시, 다른 message_id)랑 canonical(방금 갱신됨) 양쪽에 동시에 남아 화면에
            // 두 번 그려질 수 있다. ideationStreamReducer.js 상단 주석의 설계 의도
            // ("스트리밍 미리보기와 최종 메시지가 절대 동시에 렌더링될 수 없다")를 이
            // 경로에서도 그대로 지키려고, canonical을 바꿀 때 스트리밍 임시 상태도 같이
            // 확실하게 비운다.
            setStreamState(createEmptyStreamState())
          } else if (event.type === 'error') {
            console.warn('[ideation-avatar] continue-turn 실패, speaker=', finishedSpeakerId, event)
          }
        },
      })
      console.log('[avatar-debug] continueIdeationExpertTurnStream resolved OK')
    } catch (err) {
      // 조용히 무시한다 — 이 호출은 최선을 다해 다음 위원을 미리 준비시키는 배경 동작이라,
      // 실패해도(400 phase 불일치, 세션 만료, abort 등) 화면에 에러를 띄우지 않는다.
      if (err?.name !== 'AbortError') {
        console.warn('[ideation-avatar] continue-turn 요청 실패, speaker=', finishedSpeakerId, err)
      }
    } finally {
      if (avatarTurnAbortRef.current === controller) avatarTurnAbortRef.current = null
    }
  }

  const phase = ideationConv?.phase
  const phaseFailure = phase === 'failed'
    ? {
        message: ideationConv?.error?.message || '회의 처리 중 오류가 발생했습니다.',
        code: ideationConv?.error?.code || 'IDEATION_CONV_NODE_FAILED',
        failedNode: ideationConv?.failed_node || ideationConv?.error?.failed_node,
      }
    : null
  const canonicalMessages = dedupeMessagesById(ideationConv?.messages)
  // 재인/Claude(2026-07-23, 실측: "메시지가 2개 겹쳐 나옴" — 서버 확인 결과 실제 메시지는
  // 1건뿐이었다): canonical과 streamState는 서로 겹칠 수 있다 — 특히 single_turn 응답처럼
  // 아주 짧은 메시지는 canonical(finalizeStream)로 넘어가는 순간과 streamState가 비워지는
  // 순간 사이에 한 프레임이라도 둘 다 남아있으면 같은 message_id가 두 배열에 동시에 존재해
  // 화면에 두 번 그려진다. dedupeMessagesById로 한 번 더 걸러 canonical(먼저 오는 쪽)을
  // 우선하고 streamState 쪽의 같은 id는 제거한다. interruptionMarkers는 메시지가 아니라
  // 렌더링용 마커라 여기(교차 참조용 allMessages)에는 안 넣는다.
  const visibleMessages = dedupeMessagesById([...canonicalMessages, ...streamState.messages])
  const busy = starting || sending || finalizing || saving
  // awaiting_user_decision도 입력을 막지 않는다("더 이야기하기") — 백엔드
  // apply_user_answer가 이 경우도 받아 두 전문가 보완 의견으로 이어간다.
  const canReplyOrContinue = !!ideationConv && REPLYABLE_PHASES.has(phase) && !busy
  const canFinalize = !!ideationConv
    && (phase === 'awaiting_user_decision' || phase === 'discussion_complete')
    && !busy
  const hasCandidates = (ideationConv?.idea_candidates?.length || 0) > 0
  const hasSelected = !!ideationConv?.selected_idea
  // 용준/Claude(2026-07-22, 요청: "잠시만" 버튼) — 실제로 기획/개발 위원이 발언을
  // 스트리밍하는 동안에만(말풍선이 하나 이상 생겨야) 활성화한다. 이미 취소 확인을 기다리는
  // 중이면(interrupting) 다시 누를 수 없다.
  const canInterject = sending && !interrupting && streamState.messages.length > 0
  const awaitingInterjectTarget = interjectTarget === '__choosing__'
  const chosenInterjectTarget = interjectTarget && interjectTarget !== '__choosing__' ? interjectTarget : null
  const interjectPlaceholder =
    chosenInterjectTarget === 'planning_expert'
      ? '기획 위원의 의견에 대한 질문이나 의견을 입력해 주세요.'
      : chosenInterjectTarget === 'dev_expert'
        ? '개발 위원의 의견에 대한 질문이나 의견을 입력해 주세요.'
        : chosenInterjectTarget === 'both'
          ? '두 위원의 의견을 함께 다룰 질문이나 의견을 입력해 주세요.'
          : null

  async function sendNonStreaming(text) {
    try {
      const data = await replyIdeationConversation(ideationConv.session_id, text)
      setIdeationConv(data)
      setDraft('')
      setOptimisticUserMessage(null)
    } catch (err) {
      setOptimisticUserMessage(null)
      setError(classifyIdeationConvError(err))
    }
  }

  // 용준/Claude(2026-07-21, 요청: 실시간 스트리밍) — 실제 OpenAI 토큰이 도착하는 즉시
  // content(streamState)가 갱신된다(완성된 응답을 받은 뒤 재생하는 가짜 타이핑 아님).
  // 화면에 얼마나 드러낼지(displayedContent)는 위 rAF 루프가 별도로 조절한다. 이 함수는
  // 네트워크가 끝나도 곧바로 canonical로 교체하지 않는다 — 최종 state/error를
  // pendingFinalRef에 넘겨두기만 하고, 실제 교체·setSending(false)는 화면 타이핑이 다
  // 따라잡은 뒤 rAF 루프의 finalizeStream이 수행한다.
  // 용준/Claude(2026-07-22, 요청: "잠시만" 버튼 — 질문 대상 선택): interjectTargetOverride가
  // 있으면(대상 선택 버튼을 눌렀을 때) canReplyOrContinue 게이트를 우회한다 — "잠시만" 재개는
  // 라운드 중간(expert_discussion)에서도 허용돼야 하는 별도 경로이기 때문이다. 대상 문자열
  // 자체는 content를 분석해 추측하지 않고, 버튼 선택 결과를 그대로 API에 전달한다.
  async function handleSend(overrideText, interjectTargetOverride) {
    const text = (overrideText ?? draft).trim()
    const target = interjectTargetOverride ?? null
    if (!text || !ideationConv) return
    if (!target && !canReplyOrContinue) return
    setSending(true)
    setError(null)
    setInterjectTarget(null)
    // 보내는 즉시 화면에 반영 — 서버 왕복(위원 응답 생성)이 끝나기를 기다리지 않는다.
    setOptimisticUserMessage({ message_id: 'LOCAL-OPTIMISTIC-USER', speaker_id: 'user', message_type: 'answer', content: text })
    setDraft('')

    if (!streamingSupportedRef.current && !target) {
      await sendNonStreaming(text)
      setSending(false)
      return
    }

    setStreamState(createEmptyStreamState())
    pendingFinalRef.current = null
    const controller = new AbortController()
    streamAbortRef.current = controller
    let finalState = null
    let streamErrorEvent = null
    try {
      await replyIdeationConversationStream(ideationConv.session_id, text, {
        signal: controller.signal,
        targetSpeakerId: target || undefined,
        opinionTargetSpeakerId: target || undefined,
        interruptedSpeakerId: target ? interruptedSpeakerId || undefined : undefined,
        // 재인/Claude(2026-07-23): 이 화면은 항상 아바타를 재생하므로, 라운드를 새로 여는
        // reply라도 위원 발언이 한 번에 다 몰려오지 않고 첫 발언부터 페이싱 대상이 되도록
        // 매번 true로 보낸다(handleAvatarNeedNextSpeaker가 나머지 턴을 이어서 요청함).
        singleTurn: true,
        onEvent: (event) => {
          if (event.type === 'state') {
            finalState = event.state
          } else if (event.type === 'error') {
            streamErrorEvent = event
          } else {
            setStreamState((prev) => applyStreamEvent(prev, event))
          }
        },
      })
    } catch (err) {
      streamAbortRef.current = null
      setStreamState(createEmptyStreamState())
      pendingFinalRef.current = null
      if (err?.name === 'AbortError') {
        // 사용자가 화면을 벗어나 요청을 취소한 경우 — 오류로 취급하지 않는다.
        setOptimisticUserMessage(null)
        setSending(false)
        return
      }
      const classified = classifyIdeationConvError(err)
      if (classified.type === 'disabled') {
        // 스트리밍 자체가 꺼져 있다(/reply/stream 404) — 이번 세션은 이후 계속 동기식
        // API만 쓴다. 가은/Claude(2026-07-22, 요청: 안내 배너 제거): "조용히 fallback 금지"
        // 취지는 console.warn으로 유지하되, 사용자용 화면 배너는 더 이상 띄우지 않는다
        // (스트리밍이 기본 활성화된 뒤로는 개발자가 의도적으로 끈 경우에만 오는 경로다).
        streamingSupportedRef.current = false
        console.warn(
          '[ideation-stream] POST /reply/stream 이 비활성화(404) 응답을 반환해 동기식 /reply로 전환합니다. ' +
            '백엔드 backend/.env의 ENABLE_IDEATION_STREAMING 값을 확인하세요.',
          err,
        )
        await sendNonStreaming(text)
        setSending(false)
        return
      }
      setOptimisticUserMessage(null)
      setError(classified)
      setSending(false)
      return
    }

    streamAbortRef.current = null
    // 요청: "최종 state가 먼저 도착해도 임시 스트림 메시지를 즉시 삭제하지 않음" — 여기서는
    // canonical로 바꾸지 않고 rAF 루프가 화면 타이핑을 다 끝낸 뒤 처리하도록 넘겨둔다.
    pendingFinalRef.current = { finalState, errorEvent: streamErrorEvent }
    if (target) setInterruptedSpeakerId(null)
  }

  // 용준/Claude(2026-07-22, 요청: "잠시만" 버튼) — 위원이 실제로 발언을 스트리밍하는 동안만
  // 호출된다. 순서: ① 화면 타이핑/실제 스트리밍 중단 요청 → ② "회의를 잠시 멈추고
  // 있어요..." 표시 → ③ 백엔드 취소 확인(세션 lock이 실제로 풀릴 때까지 대기) → ④ 사용자
  // 입력창 활성화 + 자동 포커스 → ⑤ 질문 대상 선택 UI 표시.
  async function handleInterject() {
    if (!canInterject) return
    setInterrupting(true)
    // ① 실제 네트워크 스트림 연결을 즉시 끊는다(화면 표시만 멈추는 게 아니라).
    streamAbortRef.current?.abort()
    // 재인/Claude(2026-07-23): 아바타가 백그라운드로 미리 요청해둔 continue-turn도 함께
    // 끊는다 — 안 끊으면 사용자가 "잠시만"으로 이미 중단한 뒤에, 뒤늦게 도착한 응답이
    // canonical state를 다시 덮어써 방금 중단한 발언이 유령처럼 다시 나타날 수 있다.
    avatarTurnAbortRef.current?.abort()

    // 미완성 스트리밍 본문은 보존하지 않는다. 취소가 끝난 뒤 서버에서 완료된 partial state를
    // 다시 받아온 다음, 그 마지막 canonical 메시지 뒤에 중단 마커만 배치한다.
    const lastPartial = streamState.messages[streamState.messages.length - 1]
    const hasInterruptedContent = Boolean(lastPartial && (lastPartial.content || '').trim())
    const requestIdToCancel = streamState.requestId
    let refreshedState = null

    try {
      // ③ 취소 완료(세션 lock 해제)까지 기다린다 — 그래야 곧바로 다음 reply를 보내도
      // 409가 나지 않는다.
      await cancelIdeationConversation(ideationConv.session_id, requestIdToCancel)
    } catch {
      // 취소 API 자체가 실패해도(네트워크 등) 사용자가 다시 입력할 수 있게는 해준다 —
      // 백엔드는 결국 워커 스레드 종료 시 세션 lock을 해제한다.
    } finally {
      try {
        // 같은 스트림 요청 안에서 이미 완료된 기획/개발 발언은 서버 partial state에 저장돼
        // 있다. 이를 다시 조회하지 않으면 프론트의 오래된 canonical 목록에는 보이지 않는다.
        refreshedState = await getIdeationConversation(ideationConv.session_id)
        setIdeationConv(refreshedState)
        // 재인/Claude(2026-07-23): "잠시만"을 누를 수 있는 시점(canInterject는 스트리밍
        // 메시지가 이미 하나 이상 생긴 뒤에만 true)이면 이번 라운드를 시작한 사용자
        // 메시지는 이미 서버에 저장돼 있어, 방금 refetch한 목록에도 진짜 버전이 들어있다.
        // optimisticUserMessage를 안 지우면 그 진짜 버전과 함께 두 번 보인다.
        setOptimisticUserMessage(null)
      } catch {
        // 조회가 실패해도 입력 재개는 보장하고, 현재 canonical 목록을 fallback으로 사용한다.
      }
      if (hasInterruptedContent) {
        const completedMessages = dedupeMessagesById(refreshedState?.messages || canonicalMessages)
        const afterMessageId = completedMessages[completedMessages.length - 1]?.message_id ?? null
        setInterruptionMarkers((prev) => [
          ...prev,
          {
            markerId: `interrupted-${lastPartial.message_id || Date.now()}`,
            afterMessageId,
            speakerId: lastPartial.speaker_id || null,
          },
        ])
      }
      setStreamState(createEmptyStreamState())
      pendingFinalRef.current = null
      setSending(false)
      setInterrupting(false)
      setInterruptedSpeakerId(
        lastPartial?.speaker_id === 'planning_expert' || lastPartial?.speaker_id === 'dev_expert'
          ? lastPartial.speaker_id
          : null,
      )
      // ⑤ 대상 선택 UI를 보여준다(아직 특정 위원을 고르지 않은 상태).
      setInterjectTarget('__choosing__')
      // ④ 입력창 자동 포커스(다음 페인트 이후).
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }

  async function handleFinalize() {
    if (!canFinalize) return
    setFinalizing(true)
    setError(null)
    try {
      const data = await finalizeIdeationConversation(ideationConv.session_id)
      setIdeationConv(data)
      if (data.phase === 'finalized') {
        await onFinalized(data)
      }
    } catch (err) {
      setError(classifyIdeationConvError(err))
    } finally {
      setFinalizing(false)
    }
  }

  function handleRestart() {
    const key = ideationSessionStorageKey(projectId)
    if (key) sessionStorage.removeItem(key)
    setIdeationConv(null)
    setInterruptionMarkers([])
    setInterruptedSpeakerId(null)
    setInterjectTarget(null)
    setError(null)
    startedRef.current = false
    runStart()
  }

  // 이미 확정까지 끝난 세션으로 이 화면에 돌아온 경우(사이드바 재진입) — 다시 채팅하지
  // 않고 바로 결과로 넘어갈 수 있게만 안내한다.
  if (ideationConv?.phase === 'finalized') {
    return (
      <div style={{ maxWidth: 860 }}>
        <div className="badge green mono" style={{ marginBottom: 10 }}>주제 확정 완료</div>
        <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 16 }}>이미 이 회의로 주제를 확정했어요</h2>
        {saveError && <p style={{ color: 'var(--coral)', fontSize: 13, marginBottom: 12 }}>{saveError}</p>}
        <button
          className="btn-primary"
          style={{ display: 'flex', alignItems: 'center', gap: 8 }}
          onClick={() => onFinalized(ideationConv)}
          disabled={saving}
        >
          {saving ? '프로젝트 저장 중...' : saveError ? '프로젝트 저장 다시 시도' : '확정 결과 보기'}
        </button>
      </div>
    )
  }

  return (
    <div className="rb-grid-2" style={{ maxWidth: 1570, display: 'grid', gridTemplateColumns: '1fr 80px 468px 80px 320px', gap: 20 }}>
      <div>
        <div className="badge coral mono" style={{ marginBottom: 10 }}>주제 아이디어 회의</div>
        {onBack && (
          <button className="btn-ghost" style={{ marginBottom: 10, padding: '5px 10px', fontSize: 12 }} onClick={onBack} disabled={busy}>
            ← 이전
          </button>
        )}
        {ideationConv?.competition_name ? (
          <>
            <div style={{ fontSize: 12, color: 'var(--text-2)', fontFamily: 'var(--mono)', marginBottom: 4 }}>공모전 주제</div>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>{ideationConv.competition_name}</h2>
            <div style={{ fontSize: 13, color: 'var(--text-2)', marginBottom: 10 }}>기획 위원 · 개발 위원과 함께 좁혀가는 중</div>
          </>
        ) : (
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 10 }}>기획 위원 · 개발 위원과 함께 좁혀가는 중</h2>
        )}
        <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
          {ideationConv && (
            <span className="badge amber mono">
              {statusLabelFor({ phase, starting, sending, finalizing, interrupting })}
            </span>
          )}
        </div>

        <ErrorBanner error={phaseFailure || error} onRetry={handleRestart} />
        <StreamingCursorStyle />

        <div className="card glass" style={{ minHeight: 360, maxHeight: 520, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4, padding: 16 }}>
          {starting && !ideationConv && (
            <p style={{ color: 'var(--text-2)', fontSize: 13 }}>
              {startPhaseLabel ? `${startPhaseLabel}...` : '공모전 분석을 바탕으로 아이디어 후보를 만들고 있어요...'}
            </p>
          )}
          {interruptionMarkers
            .filter((marker) => marker.afterMessageId === null)
            .map((marker) => <InterruptionMarker key={marker.markerId} speakerId={marker.speakerId} />)}
          {canonicalMessages.map((m) => (
            <Fragment key={m.message_id}>
              <MessageBubble message={m} allMessages={visibleMessages} />
              {interruptionMarkers
                .filter((marker) => marker.afterMessageId === m.message_id)
                .map((marker) => <InterruptionMarker key={marker.markerId} speakerId={marker.speakerId} />)}
            </Fragment>
          ))}
          {/* 재인/Claude(2026-07-23): 사용자가 방금 보낸 메시지 — 서버 왕복이 끝나
              canonical에 진짜 echo가 도착하기 전까지 임시로 보여준다(handleSend/
              finalizeStream/sendNonStreaming이 도착 즉시 지운다 - 그래서 진짜 메시지와
              동시에 두 번 보이는 순간은 없다). */}
          {optimisticUserMessage && (
            <MessageBubble message={optimisticUserMessage} allMessages={visibleMessages} />
          )}
          {/* 스트리밍 임시 메시지 — message_start를 받는 즉시 말풍선이 생기고, 실제 LLM
              델타가 도착하는 대로 안에서 텍스트가 자란다(완성 후 재생하는 효과 아님).
              streamState는 최종 state 이벤트가 오면 즉시 비워지므로, 이 목록과 위
              ideationConv.messages가 같은 내용으로 동시에 남아 중복되는 순간은 없다. */}
          {streamState.messages.map((m) => (
            <MessageBubble key={m.message_id} message={m} streaming allMessages={visibleMessages} />
          ))}
          {sending && streamState.messages.length === 0 && (
            <p style={{ fontSize: 12.5, color: 'var(--text-2)' }}>
              {streamState.phaseLabel || `${statusLabelFor({ phase, starting, sending, finalizing })}...`}
            </p>
          )}
          {finalizing && (
            <p style={{ fontSize: 12.5, color: 'var(--text-2)' }}>
              {statusLabelFor({ phase, starting, sending, finalizing })}...
            </p>
          )}
          {/* 가은/Claude(2026-07-23, 요청: 후보 패널을 대화창 안으로) — 진행자가 선택을
              요청하는 첫 버블("제안된 후보 중... 답할 수 있습니다") 바로 아래에 후보
              카드를 띄운다. 이전엔 오른쪽 사이드 패널에 항상 떠 있었는데, 대화 흐름
              밖이라 어느 버블에 대한 응답인지 연결이 약했다. phase가
              awaiting_candidate_selection인 동안은 항상 그 마지막 메시지가 이 선택
              질문이므로, 메시지 목록 맨 끝에 붙이면 자연히 그 버블 바로 아래에 온다.
              선택하면 phase가 바뀌어(hasSelected) 이 블록 자체가 사라진다. */}
          {phase === 'awaiting_candidate_selection' && hasCandidates && !hasSelected && (
            <div style={{ marginTop: 4 }}>
              <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                주제 후보
              </div>
              {ideationConv.idea_candidates.map((c, i) => (
                <CandidateCard key={c.candidate_id || i} candidate={c} index={i} onSelect={(idx) => handleSend(candidateSelectMessage(idx))} disabled={!canReplyOrContinue} />
              ))}
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
                <button
                  aria-label="다시 추천"
                  title="다시 추천"
                  disabled={!canReplyOrContinue}
                  onClick={() => handleSend(REGENERATE_MESSAGE)}
                  style={{
                    background: 'none',
                    border: '1px solid var(--glass-border)',
                    borderRadius: 999,
                    width: 30,
                    height: 30,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: canReplyOrContinue ? 'pointer' : 'default',
                    opacity: canReplyOrContinue ? 1 : 0.5,
                    color: 'var(--text-2)',
                  }}
                >
                  <RefreshCw size={14} />
                </button>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* 용준/Claude(2026-07-22, 요청: "잠시만" 버튼): 위원이 실제로 발언을 스트리밍하는
            동안만 노출된다 — 클릭 시 실제 활성 스트리밍 요청을 취소한다(표시만 멈추는
            효과 아님). */}
        {canInterject && (
          <div style={{ marginTop: 8 }}>
            <button className="btn-ghost" style={{ fontSize: 12, borderColor: 'var(--amber, var(--coral))' }} onClick={handleInterject}>
              잠시만
            </button>
          </div>
        )}

        {/* 의견 대상 선택 UI — 답변자를 고르는 것처럼 보이지 않도록 "누구의 의견에
            반응하는지"를 명시한다. 선택된 위원이 먼저 답하고 상대 위원이 이어서 검토한다. */}
        {awaitingInterjectTarget && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 6 }}>
              어느 의견에 대해 말씀하시겠어요?
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button className="btn-ghost" style={{ fontSize: 12 }} onClick={() => setInterjectTarget('planning_expert')}>
                기획 위원 의견에
              </button>
              <button className="btn-ghost" style={{ fontSize: 12 }} onClick={() => setInterjectTarget('dev_expert')}>
                개발 위원 의견에
              </button>
              <button className="btn-ghost" style={{ fontSize: 12 }} onClick={() => setInterjectTarget('both')}>
                두 의견 모두에
              </button>
            </div>
          </div>
        )}

        {ideationConv && phase !== 'finalized' && (
          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key !== 'Enter') return
                if (chosenInterjectTarget) handleSend(undefined, chosenInterjectTarget)
                else if (canReplyOrContinue) handleSend()
              }}
              placeholder={
                interjectPlaceholder ||
                (awaitingInterjectTarget
                  ? '먼저 위쪽에서 의견 대상을 선택해 주세요.'
                  : phase === 'failed'
                    ? '회의 처리 오류가 발생했습니다. 위의 다시 시도 버튼을 눌러주세요.'
                  : !canReplyOrContinue
                    ? '전문가 응답을 기다리는 중입니다'
                    : phase === 'awaiting_user_decision'
                      ? '필요하면 의견을 남겨주세요 (선택 사항)'
                      : '답변을 입력하세요')
              }
              disabled={!canReplyOrContinue && !chosenInterjectTarget && !awaitingInterjectTarget}
              style={{ flex: 1, background: 'var(--bg-1)', border: '1px solid var(--glass-border)', borderRadius: 10, padding: '10px 14px', color: 'var(--text-0)', fontSize: 13 }}
            />
            <button
              className="btn-primary"
              style={{ padding: '10px 14px' }}
              disabled={(!canReplyOrContinue && !chosenInterjectTarget) || !draft.trim()}
              onClick={() => (chosenInterjectTarget ? handleSend(undefined, chosenInterjectTarget) : handleSend())}
            >
              <Send size={14} />
            </button>
          </div>
        )}

        {/* 가은/Claude(2026-07-23, 요청: "다시 추천"을 후보 패널의 새로고침 아이콘으로 이동) —
            이 버튼은 후보 카드 아래(위 채팅 영역)로 옮겼다. "전문가 추천"은 후보 자체를
            새로 만드는 게 아니라 다른 답변 경로라 그대로 입력창 아래에 남긴다. */}
        {phase === 'awaiting_candidate_selection' && (
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button className="btn-ghost" style={{ fontSize: 12 }} disabled={!canReplyOrContinue} onClick={() => handleSend(EXPERT_RECOMMEND_MESSAGE)}>
              전문가 추천
            </button>
          </div>
        )}
      </div>

      {/* 재인/Claude(2026-07-24, 실측: "아바타가 채팅 쪽에 쏠려 보임"): 처음엔 아바타-캔버스
          사이에만 스페이서(160px)를 뒀더니 채팅-아바타 간격(그리드 기본 gap 20px)보다
          아바타-캔버스 간격이 훨씬 넓어서 아바타 블록이 왼쪽(채팅 쪽)으로 쏠려 보였다.
          스페이서를 양쪽에 80px씩 나눠서 아바타 블록이 채팅-캔버스 사이에서 가운데
          오도록 했다(위 gridTemplateColumns '1fr 80px 468px 80px 320px'). */}
      <div aria-hidden="true" />

      <IdeationAvatarStage
        playQueue={avatarPlayQueue}
        onNeedNextSpeaker={handleAvatarNeedNextSpeaker}
      />

      <div aria-hidden="true" />
      <div>
        <MergeAnalysisPanel
          mergeAnalysis={ideationConv?.merge_analysis}
          sourceCandidates={ideationConv?.source_candidates}
          userSelectionMessage={ideationConv?.user_selection_message}
        />

        <IdeaCanvasPanel ideationConv={ideationConv} analysis={announcementAnalysis} />

        {/* 가은/Claude(2026-07-23, 요청: "후보 선택하면 아이디어 선택 패널 없애줘") — 후보
            카드는 대화창 안(선택 질문 버블 아래)에서 보여주고, 선택 후에는 이 자리에
            요약 카드도 더 이상 띄우지 않는다. 선택 결과는 아래 IdeaCanvasPanel(문제
            상황/타깃 사용자 등)과 대화 자체로 계속 드러난다. */}

        {ideationConv && (ideationConv.consensus?.length > 0 || ideationConv.unresolved_issues?.length > 0) && (
          <div className="card glass" style={{ marginBottom: 12, padding: 14 }}>
            {ideationConv.consensus?.length > 0 && (
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-2)', marginBottom: 4 }}>합의 사항</div>
                <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, lineHeight: 1.7 }}>
                  {ideationConv.consensus.map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              </div>
            )}
            {ideationConv.unresolved_issues?.length > 0 && (
              <div>
                <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-2)', marginBottom: 4 }}>미해결 쟁점</div>
                <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, lineHeight: 1.7 }}>
                  {ideationConv.unresolved_issues.map((u, i) => <li key={i}>{u}</li>)}
                </ul>
              </div>
            )}
          </div>
        )}

        <button className="btn-primary" style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }} disabled={!canFinalize} onClick={handleFinalize}>
          <Sparkles size={14} />
          {finalizing ? '초안 생성 중...' : '주제 확정하고 이어서 받기'}
        </button>
        {!canFinalize && ideationConv && phase !== 'finalized' && phase !== 'failed' && (
          <p style={{ fontSize: 11.5, color: 'var(--text-2)', marginTop: 8 }}>{nextActionGuideFor(phase)}</p>
        )}
      </div>
    </div>
  )
}

const PROPOSAL_ROWS = [
  ['problem_definition', '문제 정의'],
  ['target_user', '목표 사용자'],
  ['core_user_value', '핵심 사용자 가치'],
  ['key_features', '주요 기능'],
  ['required_data', '필요한 데이터'],
  ['tech_direction', '기술 구현 방향'],
  ['mvp_scope', 'MVP 범위'],
  ['differentiation', '차별성'],
  ['risks_and_mitigations', '위험 요소와 대응 방안'],
  ['success_metrics', '성공 지표'],
  ['expert_final_opinions', '전문가별 최종 판단'],
  ['unverified_assumptions', '검증이 필요한 가정'],
  ['final_recommendation', '최종 추천 여부'],
]

const NOT_FINALIZED = '아직 확정되지 않음'

function proposalValueDisplay(value) {
  if (value === null || value === undefined || value === '') return NOT_FINALIZED
  if (Array.isArray(value)) {
    if (value.length === 0) return NOT_FINALIZED
    return (
      <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
        {value.map((item, i) => (
          <li key={i}>
            {typeof item === 'object' && item !== null
              ? [item.risk, item.mitigation].filter(Boolean).join(' → ') || JSON.stringify(item)
              : String(item)}
          </li>
        ))}
      </ul>
    )
  }
  if (typeof value === 'object') {
    return (
      <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
        {Object.entries(value).map(([k, v]) => (
          <li key={k}>
            <strong style={{ fontWeight: 600 }}>{k}</strong> · {String(v)}
          </li>
        ))}
      </ul>
    )
  }
  return String(value)
}

export function IdeationResultScreen({ ideationConv, onBack }) {
  if (!ideationConv || ideationConv.phase !== 'finalized' || !ideationConv.idea_proposal) {
    return (
      <div style={{ maxWidth: 760 }}>
        <div className="badge amber mono" style={{ marginBottom: 12 }}>아직 확정되지 않음</div>
        <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 16 }}>주제 발전 회의를 먼저 완료해 주세요</h2>
        <p style={{ fontSize: 13, color: 'var(--text-2)' }}>
          "주제 아이디어 회의" 단계에서 후보를 선택하고 전문가 질문에 답한 뒤, 확정 버튼을 눌러야 결과가 만들어져요.
        </p>
      </div>
    )
  }

  const proposal = ideationConv.idea_proposal
  const originalCandidates = ideationConv.original_idea_candidates || []
  const hasDiscoveryHistory = ideationConv.ideation_mode === 'discovery' && originalCandidates.length > 0

  return (
    <div style={{ maxWidth: 780 }}>
      <div className="badge green mono" style={{ marginBottom: 12 }}>주제 확정 · 기획서 작성 출발점</div>
      {onBack && (
        <button className="btn-ghost" style={{ marginBottom: 12, padding: '5px 10px', fontSize: 12 }} onClick={onBack}>
          ← 이전
        </button>
      )}
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 20 }}>{proposal.idea_name || '확정된 주제'}</h2>

      <div className="card glass">
        {PROPOSAL_ROWS.map(([key, label], i) => (
          <div key={key} style={{ display: 'grid', gridTemplateColumns: '150px 1fr', gap: 16, padding: '14px 0', borderTop: i > 0 ? '1px solid var(--glass-border)' : 'none' }}>
            <div style={{ fontSize: 12, color: 'var(--text-2)', fontFamily: 'var(--mono)' }}>{label}</div>
            <div style={{ fontSize: 13.5, lineHeight: 1.6 }}>{proposalValueDisplay(proposal[key])}</div>
          </div>
        ))}
      </div>

      {hasDiscoveryHistory && (
        <div className="card glass" style={{ marginTop: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>아이디어 발굴 이력</div>

          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-2)', marginBottom: 6 }}>최초 후보</div>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5, lineHeight: 1.7 }}>
              {originalCandidates.map((c) => <li key={c.candidate_id}>{c.title}</li>)}
            </ul>
          </div>

          {ideationConv.selected_idea && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-2)', marginBottom: 6 }}>선택하거나 결합한 후보</div>
              <div style={{ fontSize: 12.5, lineHeight: 1.6 }}>{ideationConv.selected_idea.title}</div>
            </div>
          )}

          {ideationConv.selection_reason && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-2)', marginBottom: 6 }}>선택 이유</div>
              <div style={{ fontSize: 12.5, lineHeight: 1.6 }}>{ideationConv.selection_reason}</div>
            </div>
          )}

          {ideationConv.user_selection_message && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-2)', marginBottom: 6 }}>사용자 원문 선택 요청</div>
              <div style={{ fontSize: 12.5, lineHeight: 1.6 }}>“{ideationConv.user_selection_message}”</div>
            </div>
          )}

          {ideationConv.merge_analysis && (
            <MergeAnalysisPanel mergeAnalysis={ideationConv.merge_analysis} sourceCandidates={ideationConv.source_candidates} />
          )}
        </div>
      )}
    </div>
  )
}
