import { Fragment, useEffect, useRef, useState } from 'react'
import { AlertCircle, ArrowRight, CheckCircle2, ChevronDown, ChevronUp, Circle, Download, Lightbulb, ListChecks, RefreshCw, Send, Sparkles, Users } from 'lucide-react'
import {
  cancelIdeationConversation,
  continueIdeationExpertTurnStream,
  finalizeIdeationConversation,
  getIdeationConversation,
  getLatestIdeationConversation,
  replyIdeationConversation,
  replyIdeationConversationStream,
  startIdeationConversation,
  startIdeationConversationStream,
} from '../../api/ideationConversationApi'
import { getAnnouncementAnalysis, getApplicationFormAnalysis } from '../../api/documentApi'
import IdeaCanvasPanel from './IdeaCanvasPanel'
import IdeationAvatarStage from './IdeationAvatarStage'
import ApplicationFormFieldSelectModal from './ApplicationFormFieldSelectModal'
import ApplicationFormPanel from './ApplicationFormPanel'
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

// 가은/Claude(2026-07-26, 요청: "영상 연결 URL 없으면 대화가 진행이 안 되는 것처럼
// 보인다 — 개발자용으로 영상 없이 디버깅할 수 있게") — 아래 revealCutoffIndex 게이팅은
// avatarRevealedCount가 실제 영상 재생 시작(video.play() 성공, IdeationAvatarStage.jsx)
// 에서만 올라간다. 로컬에 MEDIA_SERVICE_WS_URL(Colab MuseTalk 서버)이 없으면 영상이
// 영원히 재생되지 않아 avatarRevealedCount가 0에 머물고, 첫 위원 발언 이후 모든 메시지가
// 화면에서 영원히 숨겨진다 — 회의 자체(백엔드 phase/messages)는 정상 진행 중인데
// 화면만 멈춘 것처럼 보인다. 개발 중 영상 서버 없이 회의 로직만 확인하고 싶을 때
// frontend/.env(또는 .env.local)에 VITE_IDEATION_AVATAR_ENABLED=false를 넣으면 이
// 게이팅과 아바타 소켓 연결 시도 자체를 건너뛴다. 값을 안 주면(기본) 지금까지와 동일하게
// 항상 켜져 있다 — 운영/일반 개발 흐름에 영향 없음.
const IDEATION_AVATAR_ENABLED = import.meta.env.VITE_IDEATION_AVATAR_ENABLED !== 'false'

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
// 용준/Claude(2026-07-25, 요청: 3열 레이아웃 — 왼쪽 전체 Stepper(Shell navrail, 220px)는
// 그대로 두고, 이 화면 안에서는 "중앙 대화 / 오른쪽 참여 위원·아이디어·캔버스" 2열만 새로
// 짠다. 데스크톱은 고정 420px 오른쪽 패널 + sticky, 태블릿(<=1180px)은 오른쪽 패널이 중앙
// 아래로 내려오며 sticky를 끄고, 모바일(<=780px)은 이미 Shell이 navrail을 숨기므로 이
// 화면 안에서는 그대로 단일 열이 유지된다(요청한 모바일 순서 "단계 → 대화 → 제안 목록 →
// CTA"는 센터 다음에 오른쪽 패널이 오는 DOM 순서 그대로 만족된다). .badge.blue는 개발
// 위원 배지 색(요청: "파랑 계열")을 위해 이 페이지에만 추가한다 — Shell(공용 파일)의
// .badge.purple/.coral/.green/.amber/.grey 정의는 건드리지 않는다.
function StreamingCursorStyle() {
  return (
    <style>{`
      @keyframes rb-ideation-cursor-blink { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }
      .rb-ideation-cursor { display: inline-block; width: 2px; margin-left: 1px; background: currentColor; animation: rb-ideation-cursor-blink 1s step-start infinite; }
      .rb-root .badge.blue{ background: rgba(59,130,246,0.12); color: #2f6fd6; }
      .rb-ideation-layout .badge{ font-size:12.5px; font-weight:600; padding:4px 10px; }
      .rb-ideation-layout .badge.green{ color:#087557; background:#dcf7ee; }
      .rb-ideation-layout .badge.purple{ color:#5e3ec8; background:#eee9ff; }
      .rb-ideation-layout .badge.blue{ color:#245fb7; background:#e5efff; }
      .rb-ideation-layout .badge.amber{ color:#805800; background:#f8edcf; }
      .rb-ideation-layout .rb-ideation-meta-item{ border-color:rgba(28,26,46,0.14); background:rgba(255,255,255,0.86); }
      .rb-ideation-layout .rb-ideation-meta-label{ font-size:12.5px; font-weight:600; color:#625d72; }
      .rb-ideation-layout .rb-ideation-meta-label svg{ color:var(--purple); stroke-width:2.2; }
      .rb-ideation-layout .rb-ideation-meta-value{ margin-top:7px; font-size:18px; line-height:1.2; }
      .rb-ideation-layout .btn-primary{ min-height:46px; color:#fff; font-weight:700; }
      .rb-ideation-layout .btn-primary:disabled{
        opacity:1; background:#eceef1; color:#8a8f98; border:1px solid #dcdfe4;
        box-shadow:none; cursor:not-allowed;
      }
      .rb-ideation-layout .btn-ghost:disabled{ opacity:1; color:#6c6578; background:#f1eef5; cursor:not-allowed; }
      .rb-ideation-notice{
        margin:4px 0; padding:10px 12px; border-radius:10px; border:1px solid #ddd5f4;
        background:#f7f4ff; color:#514a61; font-size:14px; font-weight:600; line-height:1.55;
      }
      /* 용준/Claude(2026-07-26, 요청: "아이디어 기획 캔버스는 참여 위원 오른쪽에") —
         참여 위원 패널과 캔버스를 한 열에 위아래로 쌓지 않고, 채팅 열 옆에 각각
         독립된 열로 나란히 배치한다(3열: 회의 대화 / 참여 위원 / 기획 캔버스). */
      .rb-ideation-layout{
        --text-2:#6f697d;
        display:grid; grid-template-columns:minmax(0,1fr) 480px 340px;
        gap:20px; align-items:start; max-width:1680px;
      }
      .rb-ideation-side{ position:sticky; top:24px; margin-top:40px; display:flex; flex-direction:column; gap:12px; }
      .rb-ideation-canvas-col{ position:sticky; top:24px; margin-top:40px; display:flex; flex-direction:column; gap:12px; }
      .rb-ideation-meta{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }
      .rb-ideation-meta-item{ flex:1 1 120px; border-radius:12px; padding:10px 12px; }
      .rb-ideation-meta-label{ display:flex; align-items:center; gap:5px; margin-bottom:3px; }
      .rb-ideation-meta-value{ font-weight:700; color:var(--text-0); }
      @media (max-width: 1500px){
        .rb-ideation-layout{ grid-template-columns: minmax(0,1fr) 460px; }
        .rb-ideation-canvas-col{ position:static; grid-column: 1 / -1; margin-top:0; }
      }
      @media (max-width: 1180px){
        .rb-ideation-layout{ grid-template-columns: minmax(0,1fr); }
        .rb-ideation-side{ position:static; margin-top:0; }
        .rb-ideation-canvas-col{ grid-column: auto; margin-top:0; }
      }
      .rb-ideation-candidate-card{ transition: transform .18s ease, box-shadow .18s ease; }
      .rb-ideation-candidate-card--interactive:hover{ transform: translateY(-4px); box-shadow: 0 10px 24px rgba(124,92,234,0.18); }
      .rb-ideation-candidate-card--interactive:active{ transform: translateY(-1px); box-shadow: 0 4px 10px rgba(124,92,234,0.16); transition-duration: .06s; }
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
    <div style={{ fontSize: 13.5, color: 'var(--text-2)', marginBottom: 3 }}>↳ {targetLabel}에게 응답</div>
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
          fontSize: 13,
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
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          background: 'none', border: 'none', padding: 0, cursor: 'pointer',
          fontSize: 13, color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: 2,
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
                fontSize: 13, color: 'var(--text-1)', lineHeight: 1.5,
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
                <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 2 }}>평가항목: {e.section}</div>
              )}
              {(e.text || e.quote) && <div style={{ marginBottom: 2 }}>"{e.text || e.quote}"</div>}
              {e.source_url && (
                <a href={e.source_url} target="_blank" rel="noreferrer" style={{ fontSize: 13.5, fontWeight: 600, color: '#5e3ec8' }}>
                  원문 보기
                </a>
              )}
            </div>
          ))}
          {items.length === 0 && (
            <div
              style={{
                fontSize: 13, color: 'var(--text-2)', fontStyle: 'italic',
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
        background: 'var(--bg-0)', border: '1px solid var(--glass-border)', fontSize: 14.5,
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

function MessageBubble({ message, streaming = false, interrupted = false, allMessages = [], isLatest = false }) {
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
        <div style={{ fontSize: 14, color: 'var(--coral)' }}>
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
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span className={`badge ${meta.badgeClass} mono`}>{meta.label}</span>
          </div>
        )}
        {!streaming && <RespondingToCaption message={message} allMessages={allMessages} />}
        <div
          style={{
            background: isRight ? 'var(--purple-dim)' : 'var(--bg-1)',
            border: '1px solid var(--glass-border)',
            borderRadius: 12,
            padding: '12px 15px',
            color: 'var(--text-0)',
            fontSize: 16,
            fontWeight: 500,
            lineHeight: 1.7,
            whiteSpace: 'pre-wrap',
            wordBreak: 'keep-all',
            boxShadow: isLatest ? '0 4px 14px rgba(28,26,46,0.06)' : 'none',
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
          <div style={{ fontSize: 13, color: 'var(--text-2)', marginTop: 3, fontStyle: 'italic' }}>
            [사용자에 의해 발언이 중단됐습니다]
          </div>
        )}
        {/* 용준/Claude(2026-07-23, 요청: 스트리밍 UX 버그 수정) — 검증 실패로 재검토
            대기 중임을 알린다. willRetry가 false면(재시도 없이 안전한 fallback을 기다리는
            중) 곧 회의 진행 자체가 멈추지 않는다는 것을 알 수 있게 문구를 살짝 다르게 준다. */}
        {isReviewing && (
          <div style={{ fontSize: 13, color: 'var(--text-2)', marginTop: 3, fontStyle: 'italic' }}>
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
        fontSize: 14.5,
        fontStyle: 'italic',
      }}
    >
      [{speakerLabel ? `${speakerLabel}의 ` : ''}발언이 사용자에 의해 중단됐습니다]
    </p>
  )
}

// 용준/Claude(2026-07-25, 요청: "각 후보마다 큰 보라색 버튼을 반복해서 표시하지 말고,
// 카드 전체를 클릭 가능한 선택 카드로") — 반복되던 "이 후보 선택" 버튼을 없애고 카드
// 전체를 role="button"으로 만들었다. 선택 확정 API 호출(onSelect → handleSend →
// candidateSelectMessage)은 그대로다 — 프론트가 후보를 자체 해석하지 않고 항상 "n번"
// 문구로 백엔드 candidate_selection 로직을 거치는 기존 방식을 그대로 유지한다. "상세
// 보기" 토글은 카드 클릭(선택)과 별개 동작이라 stopPropagation으로 분리한다.
function CandidateCard({ candidate, index, onSelect, disabled, selected = false }) {
  const [expanded, setExpanded] = useState(false)

  function handleCardActivate() {
    if (disabled || selected) return
    onSelect(index)
  }

  return (
    <div
      role="button"
      tabIndex={disabled || selected ? -1 : 0}
      aria-pressed={selected}
      aria-disabled={disabled || selected}
      onClick={handleCardActivate}
      onKeyDown={(e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return
        e.preventDefault()
        handleCardActivate()
      }}
      className={`card glass rb-ideation-candidate-card${disabled || selected ? '' : ' rb-ideation-candidate-card--interactive'}`}
      style={{
        marginBottom: 10,
        padding: 14,
        cursor: disabled || selected ? 'default' : 'pointer',
        border: selected ? '2px solid var(--purple)' : '1px solid var(--glass-border)',
        background: selected ? '#f5f1ff' : disabled ? '#faf9fc' : 'var(--bg-1)',
        opacity: 1,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 2, gap: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#625d72' }}>후보 {index + 1}</div>
        {selected ? (
          <CheckCircle2 size={18} color="var(--purple)" style={{ flexShrink: 0 }} />
        ) : (
          <Circle size={16} color="var(--glass-border)" style={{ flexShrink: 0 }} />
        )}
      </div>
      {selected && (
        <span className="badge purple mono" style={{ fontSize: 12, marginBottom: 6, display: 'inline-block' }}>선택됨</span>
      )}
      <div style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-0)', marginBottom: 7 }}>{candidate.title}</div>
      {candidate.problem && (
        <div style={{ fontSize: 15.5, fontWeight: 500, color: 'var(--text-0)', lineHeight: 1.7, marginBottom: 5 }}>
          <strong style={{ color: '#514a61', fontWeight: 700 }}>해결할 문제 · </strong>
          {candidate.problem}
        </div>
      )}
      {candidate.target_user && (
        <div style={{ fontSize: 15.5, fontWeight: 500, color: 'var(--text-0)', lineHeight: 1.7, marginBottom: 5 }}>
          <strong style={{ color: '#514a61', fontWeight: 700 }}>목표 사용자 · </strong>
          {candidate.target_user}
        </div>
      )}

      <button
        type="button"
        className="btn-ghost"
        style={{ padding: '4px 10px', fontSize: 13, marginTop: 4, marginBottom: expanded ? 8 : 0, display: 'flex', alignItems: 'center', gap: 4 }}
        onClick={(e) => { e.stopPropagation(); setExpanded((v) => !v) }}
      >
        {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        {expanded ? '간단히 보기' : '상세 보기'}
      </button>

      {expanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 6 }}>
          {candidate.core_value && (
            <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>핵심 가치 · </strong>
              {candidate.core_value}
            </div>
          )}
          {candidate.main_features?.length > 0 && (
            <div style={{ fontSize: 14, color: 'var(--text-1)' }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>주요 기능</strong>
              <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.7 }}>
                {candidate.main_features.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </div>
          )}
          {candidate.contest_fit && (
            <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>차별점 · </strong>
              {candidate.contest_fit}
            </div>
          )}
          {candidate.risks?.length > 0 && (
            <div style={{ fontSize: 14, color: 'var(--text-1)' }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>주요 위험</strong>
              <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.7 }}>
                {candidate.risks.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}

      {candidate.feasibility && (
        <div style={{ fontSize: 13, color: 'var(--text-2)' }}>
          실현 가능성 {FEASIBILITY_LABEL[candidate.feasibility] || '미상'}
        </div>
      )}
    </div>
  )
}

function MergeAnalysisPanel({ mergeAnalysis, sourceCandidates, userSelectionMessage }) {
  if (!mergeAnalysis) return null
  return (
    <div className="card glass" style={{ marginBottom: 12, padding: 14 }}>
      <div style={{ fontSize: 13.5, fontWeight: 700, marginBottom: 8 }}>결합 분석</div>
      {userSelectionMessage && (
        <div style={{ fontSize: 13.5, color: 'var(--text-2)', marginBottom: 8 }}>
          원문 요청 · “{userSelectionMessage}”
        </div>
      )}
      {sourceCandidates?.length > 0 && (
        <div style={{ fontSize: 14, color: 'var(--text-1)', marginBottom: 8 }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>원본 후보 · </strong>
          {sourceCandidates.map((c) => c.title).join(' · ')}
        </div>
      )}
      <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6, marginBottom: 4 }}>
        <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>공통 문제 · </strong>{mergeAnalysis.common_problem}
      </div>
      <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6, marginBottom: 4 }}>
        <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>공통 가치 · </strong>{mergeAnalysis.common_value}
      </div>
      <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6, marginBottom: 4 }}>
        <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>결합 적합도 · </strong>
        {FEASIBILITY_LABEL[mergeAnalysis.fit] || mergeAnalysis.fit || '미상'}
      </div>
      {mergeAnalysis.primary_features?.length > 0 && (
        <div style={{ fontSize: 14, color: 'var(--text-1)', marginBottom: 4 }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>주 기능</strong>
          <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {mergeAnalysis.primary_features.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      )}
      {mergeAnalysis.secondary_features?.length > 0 && (
        <div style={{ fontSize: 14, color: 'var(--text-1)', marginBottom: 4 }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>보조 기능</strong>
          <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {mergeAnalysis.secondary_features.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      )}
      {mergeAnalysis.conflicts?.length > 0 && (
        <div style={{ fontSize: 14, color: 'var(--text-1)', marginBottom: 4 }}>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>충돌 지점</strong>
          <ul style={{ margin: '4px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {mergeAnalysis.conflicts.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      )}
      {mergeAnalysis.open_questions?.length > 0 && (
        <div style={{ fontSize: 14, color: 'var(--text-1)' }}>
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
        <div style={{ fontSize: 14.5, color: 'var(--text-0)', lineHeight: 1.6 }}>{error.message}</div>
        {(error.code || error.failedNode) && (
          <div style={{ marginTop: 5, fontSize: 13, color: 'var(--text-2)', fontFamily: 'var(--mono)' }}>
            {error.code && `오류 코드: ${error.code}`}
            {error.code && error.failedNode && ' · '}
            {error.failedNode && `실패 노드: ${error.failedNode}`}
          </div>
        )}
        {onRetry && (
          <button type="button" className="btn-ghost" style={{ marginTop: 10, padding: '6px 12px', fontSize: 13.5, display: 'flex', alignItems: 'center', gap: 6 }} onClick={onRetry}>
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
  // 가은/Claude(2026-07-23): 신청서 양식 항목(JSON export에도 포함) — runStart()가 이미
  // getApplicationFormAnalysis()로 받아오던 값을 상태로 유지한다. 세션 재개(resume)
  // 경로에서는 start를 다시 부르지 않으므로 ideationConv.application_form_items로
  // 채워진다(아래 useEffect 없이도 렌더 시점에 반영되도록 초기값에서부터 시도).
  const [applicationFormItems, setApplicationFormItems] = useState(
    () => ideationConv?.application_form_items || [],
  )
  // 가은/Claude(2026-07-24, 요청: 회의에 쓸 신청서 항목을 사용자가 고르게) — null이면
  // 팝업이 안 보이고, {items, analysis}면 회의 시작 전 확인 팝업이 뜬다. 실제 /start
  // 호출(무거운 후보 생성 LLM)은 이 팝업을 사용자가 확정할 때까지 미룬다 — 뒷 화면은
  // 계속 starting=true 로딩 상태를 보여준다.
  const [formSelectionPrompt, setFormSelectionPrompt] = useState(null)
  // 재인/Claude(2026-07-23): 아이디어 회의 아바타 연동 — canonical 메시지 중 아바타
  // 3명(진행자/기획/개발)에 해당하는 것만 골라 재생 큐로 넘긴다. 이미 큐에 넣은
  // message_id는 queuedAvatarIdsRef로 추적해서, 메시지 목록이 리렌더될 때마다 같은
  // 발언을 중복으로 다시 큐잉하지 않게 막는다.
  const [avatarPlayQueue, setAvatarPlayQueue] = useState([])
  const queuedAvatarIdsRef = useRef(new Set())
  // 재인/Claude(2026-07-24, 요청: "채팅 텍스트도 아바타 재생 순서에 맞춰 공개"): 지금까지
  // 실제로 재생을 시작한 아바타 발언 개수. IdeationAvatarStage가 각 항목을 화면에 틀기
  // 시작하는 순간(video.play() 성공 시점)마다 onRevealed로 하나씩 늘려준다 — 재생은
  // 항상 큐 순서대로(=메시지 순서대로) 진행되므로 개수만 세도 순서가 정확히 맞는다.
  const [avatarRevealedCount, setAvatarRevealedCount] = useState(0)
  // 재인/Claude(2026-07-23, 2026-07-24 갱신): 아래 eager fetch effect가 지금 진행 중인
  // continue-turn fetch를 추적한다 — "잠시만"이 그 사이에 눌리면 abort()로 끊어서, 이미
  // 중단한 뒤에 뒤늦게 도착하는 응답이 canonical state를 다시 덮어쓰지 않게 막는다
  // (handleInterject 참고). 그 effect 자신이 sending/interrupting/interjectTarget이 걸려
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
      let items = []
      const hasReadyCriteriaDoc = criteriaDocuments.some((doc) => doc.status === 'done' || doc.status === 'warning')
      if (projectId && hasReadyCriteriaDoc) {
        try {
          const formAnalysis = await getApplicationFormAnalysis(projectId)
          items = formAnalysis.items || []
        } catch (err) {
          console.warn('[ideation-conv] 신청양식 항목 조회에 실패해 항목 없이 회의를 시작합니다.', err)
        }
      }
      setApplicationFormItems(items)

      if (items.length > 0) {
        // 가은/Claude(2026-07-24, 요청: 회의에 쓸 신청서 항목을 사용자가 고르게) — 실제
        // /start 호출(무거운 후보 생성 LLM)은 사용자가 팝업에서 항목을 확정한 뒤로
        // 미룬다. starting은 여기서 끄지 않는다 — 팝업 뒤 화면은 계속 로딩 상태로 보인다.
        setFormSelectionPrompt({ items, analysis })
        return
      }

      await startConversation(analysis, [])
    } catch (err) {
      setError(classifyIdeationConvError(err))
      setStarting(false)
      setStartPhaseLabel('')
    }
  }

  // 가은/Claude(2026-07-24): runStart()의 "실제 회의 시작 호출" 부분을 분리했다 — 신청서
  // 항목이 있으면 확인 팝업 이후에, 없으면 runStart()가 곧바로 호출한다. starting/error
  // 상태는 이 함수가 끝까지 책임진다(성공/실패 모두 finally에서 정리).
  async function startConversation(analysis, selectedFormItems) {
    try {
      const payload = {
        competitionName: competitionNameFrom(analysis),
        competitionDocument: buildCompetitionDocumentText(analysis),
        userIdea: '', // discovery 모드로 시작해야 하므로 반드시 빈 문자열로 보낸다.
        maxRounds: 3,
        useRag: resolveUseRag(projectId, criteriaDocuments),
        projectId,
        applicationFormItems: selectedFormItems,
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

  function handleConfirmFormSelection(selectedItems) {
    const prompt = formSelectionPrompt
    setFormSelectionPrompt(null)
    setApplicationFormItems(selectedItems)
    startConversation(prompt?.analysis, selectedItems)
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
      if (!projectId) {
        runStart()
        return
      }
      setStarting(true)
      getAnnouncementAnalysis(projectId).then(setAnnouncementAnalysis).catch(() => {})
      getLatestIdeationConversation(projectId)
        .then((data) => {
          if (!data) return runStart()
          setIdeationConv(data)
          if (key) sessionStorage.setItem(key, data.session_id)
        })
        .catch((err) => setError(classifyIdeationConvError(err)))
        .finally(() => setStarting(false))
      return
    }
    setStarting(true)
    if (projectId) {
      getAnnouncementAnalysis(projectId).then(setAnnouncementAnalysis).catch(() => {})
    }
    getIdeationConversation(savedSessionId)
      .then((data) => setIdeationConv(data))
      .catch((err) => {
        if (err?.status !== 404) {
          setError(classifyIdeationConvError(err))
          return
        }
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

  // 재인/Claude(2026-07-24, 요청: "텍스트는 미리 다 뽑아두고 코랩만 순서대로 처리"):
  // 예전엔 아바타 pacing timer가 "재생 끝나기 8초 전"에 불러줘야만 다음 턴을 요청했다
  // (handleAvatarNeedNextSpeaker가 IdeationAvatarStage의 onNeedNextSpeaker로 연결).
  // 이제 텍스트 생성은 아바타 재생 타이밍과 완전히 분리한다 — 응답이 도착하는 즉시,
  // phase가 여전히 expert_discussion이면 곧바로 다음 턴을 요청해서 회의 로직이 허용하는
  // 한 최대한 빠르게 이어붙인다(예전 아바타 연동 전 방식과 동일). 실제로 사용자에게
  // 보여주는 시점(채팅 텍스트 노출 + 아바타 재생)은 IdeationAvatarStage 쪽에서 별도로
  // 조절한다(아래 avatarRevealedCount 참고) — 여기서는 "다음 턴이 뭔지 최대한 빨리
  // 알아내는" 역할만 한다. 회의 로직(누가 다음에 말할지, 언제 라운드가 끝나는지) 자체는
  // continue_ideation_expert_turn을 그대로 재사용하므로 전혀 안 건드렸다.
  //
  // sending/interrupting/interjectTarget 중 하나라도 걸려 있으면 부르지 않는다 — 사용자가
  // 직접 reply/interject를 보내는 중이면 같은 세션에 동시 요청을 보내 백엔드 세션 락
  // 409를 유발할 수 있다. avatarTurnAbortRef.current가 이미 걸려 있으면(이전 호출이 아직
  // 진행 중) 또 시작하지 않는다 — 이 ref는 handleInterject("잠시만")가 그대로 재사용해서
  // 중단 시 함께 끊는다(아래 avatarTurnAbortRef 선언부 참고, 코드 변경 없음).
  //
  // 재인/Claude(2026-07-25, 실측: "기획위원 한 번 말하고 그대로 멈춤" - 콘솔/백엔드 로그로
  // 확정): 처음엔 이 effect가 ideationConv(객체 전체)를 deps로 삼아서, 자기 자신의
  // setIdeationConv 호출로 매번 다시 실행되며 "다음 턴 요청"을 이어갔다. 근데 그러면
  // 매번 이 effect의 클린업(controller.abort())과 async 콜백의 finally
  // (avatarTurnAbortRef.current = null)가 경쟁한다 - 클린업이 먼저 실행된 시점엔 아직
  // finally가 ref를 안 지웠을 수 있어서, 바로 이어서 실행되는 새 effect가
  // "avatarTurnAbortRef.current가 있으니 이미 진행 중"으로 오판하고 아무것도 안 하고
  // 끝나버린다 - 그러면 그 뒤로는 아무도 다시 안 불러서 대화가 영원히 멈춘다(실측
  // 재현: 기획위원 발언 후 continue-turn 요청 자체가 다시 안 나감). 고친 방법: 렌더
  // 사이클(effect 재실행)에 의존해서 다음 턴을 잇지 않고, 이 안에서 while 루프로
  // 직접 이어서 요청한다 - deps에서도 ideationConv(전체 객체)를 빼고 session_id/phase만
  // 남겨서, 이 루프 자신의 setIdeationConv 호출로 effect가 다시 트리거되는 일 자체가
  // 없게 했다(그러면 클린업도 안 도니 경쟁 자체가 발생하지 않는다). session_id/phase가
  // "진짜로" 바뀌거나 sending/interrupting/interjectTarget이 바뀔 때만 재평가한다.
  useEffect(() => {
    if (!ideationConv?.session_id) return
    if (ideationConv.phase !== 'expert_discussion') return
    if (sending || interrupting || interjectTarget) return
    if (avatarTurnAbortRef.current) return

    const controller = new AbortController()
    avatarTurnAbortRef.current = controller
    let cancelled = false
    let currentSessionId = ideationConv.session_id

    ;(async () => {
      try {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          let nextState = null
          console.log('[avatar-debug] eager fetch: calling continueIdeationExpertTurnStream', { sessionId: currentSessionId })
          // eslint-disable-next-line no-await-in-loop
          await continueIdeationExpertTurnStream(currentSessionId, {
            signal: controller.signal,
            onEvent: (event) => {
              if (cancelled) return
              console.log('[avatar-debug] continue-turn event', event.type, event)
              if (event.type === 'state') {
                nextState = event.state
                setIdeationConv(event.state)
                // 재인/Claude(2026-07-23, 실측: "기획위원 발언이 화면에 2번 뜸"): 이 canonical
                // 갱신은 handleSend의 타이핑 효과 루프(streamState/pendingFinalRef/tick)를 전혀
                // 거치지 않는다 - 그쪽이 아직 스트리밍 임시 메시지를 못 지운 상태에서 이
                // 이벤트가 오면, 같은 발언이 streamState(임시, 다른 message_id)랑
                // canonical(방금 갱신됨) 양쪽에 동시에 남아 화면에 두 번 그려질 수 있다.
                // canonical을 바꿀 때 스트리밍 임시 상태도 같이 확실하게 비운다.
                setStreamState(createEmptyStreamState())
              } else if (event.type === 'error') {
                console.warn('[ideation-avatar] continue-turn 실패', event)
              }
            },
          })
          console.log('[avatar-debug] eager fetch: continueIdeationExpertTurnStream resolved OK', { phase: nextState?.phase })
          if (cancelled || !nextState || nextState.phase !== 'expert_discussion') break
          currentSessionId = nextState.session_id
        }
      } catch (err) {
        // 조용히 무시한다 — 이 호출은 최선을 다해 다음 턴을 미리 준비시키는 배경 동작이라,
        // 실패해도(400 phase 불일치, 세션 만료, abort 등) 화면에 에러를 띄우지 않는다.
        if (err?.name !== 'AbortError') {
          console.warn('[ideation-avatar] continue-turn 요청 실패', err)
        }
      } finally {
        if (avatarTurnAbortRef.current === controller) avatarTurnAbortRef.current = null
      }
    })()

    return () => {
      cancelled = true
      controller.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ideationConv?.session_id, ideationConv?.phase, sending, interrupting, interjectTarget])

  const phase = ideationConv?.phase
  const phaseFailure = phase === 'failed'
    ? {
        message: ideationConv?.error?.message || '회의 처리 중 오류가 발생했습니다.',
        code: ideationConv?.error?.code || 'IDEATION_CONV_NODE_FAILED',
        failedNode: ideationConv?.failed_node || ideationConv?.error?.failed_node,
      }
    : null
  // 재인/Claude(2026-07-24, 요청: "채팅 텍스트도 아바타 재생 순서에 맞춰 공개"): 텍스트은
  // 이제 아바타 재생과 무관하게 미리 다 뽑혀 ideationConv.messages(canonical)에 들어와
  // 있을 수 있다. avatarRevealedCount(IdeationAvatarStage가 각 아바타 발언을 실제로
  // 재생하기 시작할 때마다 하나씩 늘려줌 — 순서 보장됨, 아래 setAvatarRevealedCount 참고)
  // 만큼만 "아바타 대상 발언"을 공개하고, 아직 안 밝혀진 아바타 발언 지점부터는(그
  // 뒤에 온 비아바타 메시지 포함) 전부 숨긴다 — 순서를 지키기 위해서다(나중에 온 문구가
  // 아직 재생 안 한 위원 발언보다 먼저 보이면 안 됨). 아바타 대상이 아닌 메시지(고정
  // 문구 중 코랩 제외 대상, 사용자 메시지 등)는 게이팅 없이 즉시 보인다.
  const rawMessages = ideationConv?.messages || []
  let avatarSeenCount = 0
  // IDEATION_AVATAR_ENABLED=false(개발자용)면 영상 재생을 기다리지 않고 항상 -1(=자르지
  // 않음)로 취급한다 — 아바타 서버가 없어도 회의 메시지가 즉시 전부 보인다.
  const revealCutoffIndex = IDEATION_AVATAR_ENABLED
    ? rawMessages.findIndex((m) => {
        const isAvatarTracked = AVATAR_SPEAKER_IDS.has(m.speaker_id) && !(m.content || '').startsWith(AVATAR_EXCLUDED_CONTENT_PREFIX)
        if (!isAvatarTracked) return false
        avatarSeenCount += 1
        return avatarSeenCount > avatarRevealedCount
      })
    : -1
  const revealedMessages = revealCutoffIndex === -1 ? rawMessages : rawMessages.slice(0, revealCutoffIndex)
  const canonicalMessages = dedupeMessagesById(revealedMessages)
  // 재인/Claude(2026-07-23, 실측: "메시지가 2개 겹쳐 나옴" — 서버 확인 결과 실제 메시지는
  // 1건뿐이었다): canonical과 streamState는 서로 겹칠 수 있다 — 특히 single_turn 응답처럼
  // 아주 짧은 메시지는 canonical(finalizeStream)로 넘어가는 순간과 streamState가 비워지는
  // 순간 사이에 한 프레임이라도 둘 다 남아있으면 같은 message_id가 두 배열에 동시에 존재해
  // 화면에 두 번 그려진다. dedupeMessagesById로 한 번 더 걸러 canonical(먼저 오는 쪽)을
  // 우선하고 streamState 쪽의 같은 id는 제거한다. interruptionMarkers는 메시지가 아니라
  // 렌더링용 마커라 여기(교차 참조용 allMessages)에는 안 넣는다.
  const visibleMessages = dedupeMessagesById([...canonicalMessages, ...streamState.messages])
  const latestVisibleMessageId = [
    ...canonicalMessages,
    ...(optimisticUserMessage ? [optimisticUserMessage] : []),
    ...streamState.messages,
  ]
    .filter((message) => (message?.displayedContent ?? message?.content ?? '').trim())
    .at(-1)?.message_id
  const busy = starting || sending || finalizing || saving
  // awaiting_user_decision도 입력을 막지 않는다("더 이야기하기") — 백엔드
  // apply_user_answer가 이 경우도 받아 두 전문가 보완 의견으로 이어간다.
  const canReplyOrContinue = !!ideationConv && REPLYABLE_PHASES.has(phase) && !busy
  const canFinalize = !!ideationConv
    && (phase === 'awaiting_user_decision' || phase === 'discussion_complete')
    && !busy
  const hasCandidates = (ideationConv?.idea_candidates?.length || 0) > 0
  const hasSelected = !!ideationConv?.selected_idea
  // 용준/Claude(2026-07-25, 요청: 아이디어 목록에서 선택 상태를 명확히 보여주기) —
  // selected_idea/source_candidates는 백엔드가 이미 확정한 값이다(단일 선택은 원본
  // candidate_id를 그대로 갖고, 결합은 source_candidates에 원본 두 후보가 담긴다). 프론트는
  // candidate_id 우선, 없으면 title로 후보 카드와 대조해 "선택됨" 표시만 붙인다.
  const selectedCandidateKeys = new Set(
    [ideationConv?.selected_idea, ...(ideationConv?.source_candidates || [])]
      .filter(Boolean)
      .flatMap((idea) => [idea.candidate_id, idea.title])
      .filter(Boolean),
  )
  const isCandidateSelected = (candidate) =>
    selectedCandidateKeys.has(candidate.candidate_id) || selectedCandidateKeys.has(candidate.title)
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
        // 재인/Claude(2026-07-23, 2026-07-24 갱신): 이 화면은 항상 아바타를 재생하므로,
        // 라운드를 새로 여는 reply라도 위원 발언이 한 번에 다 몰려오지 않고 딱 1건만
        // 오게 매번 true로 보낸다(위쪽 eager fetch effect가 나머지 턴을 이어서 요청함 —
        // 텍스트는 이제 아바타 재생과 무관하게 미리 다 뽑히지만, 채팅/아바타 노출은
        // avatarRevealedCount로 여전히 하나씩 순서대로 공개된다).
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

  // 가은/Claude(2026-07-24, 요청: 분석용 JSON 내보내기) — 아이디어 기획 캔버스(idea_canvas),
  // 대화 전체(발화자·내용, UI 토글로 숨겨진 발언도 전부 포함 — 분석 목적이라 화면 표시
  // 필터를 따르지 않는다), 신청서 양식 항목을 한 파일로 내려받는다. 세션 루프·회귀
  // 디버깅용으로 실제 대화 로그를 그대로 봐야 할 때 쓴다(구두 요청, 2026-07-26 dev
  // #131-167 merge 후 재이식 — merge 전 이 함수가 참조하던 confirmed_plan/
  // application_form_draft/candidate_seed 등 form_coach_v2 전용 필드는 dev 쪽 데이터
  // 모델에 없어 뺐다. message.structured는 스키마가 계속 바뀌므로 가공하지 않고
  // 그대로 담는다).
  function handleExportAnalysisJson() {
    if (!ideationConv) return
    const payload = {
      exported_at: new Date().toISOString(),
      session_id: ideationConv.session_id,
      competition_name: ideationConv.competition_name || null,
      phase: ideationConv.phase || null,
      round: ideationConv.round || null,
      stop_reason: ideationConv.stop_reason || null,
      idea_canvas: ideationConv.idea_canvas || null,
      last_user_selection: ideationConv.user_selection_message || null,
      conversation: (ideationConv.messages || []).map((m) => ({
        round: m.round ?? null,
        speaker: speakerMetaFor(m).label,
        speaker_id: m.speaker_id,
        role: m.role || null,
        message_type: m.message_type,
        content: m.content,
        structured: m.structured || null,
      })),
      application_form_items: applicationFormItems,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    // 가은/Claude(2026-07-24, 요청: "저장 제목 기준이 뭐야? 시간 넣어줘") — 이전엔
    // session_id만 써서 같은 세션을 여러 번 내보내면 파일명이 계속 같았다(다운로드 폴더에서
    // 몇 번째로 내보낸 건지 구분 불가). 내보낸 시각(로컬 시간, YYYYMMDD-HHmmss)을 붙인다.
    const now = new Date()
    const pad = (n) => String(n).padStart(2, '0')
    const timestamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
    a.download = `ideation-${ideationConv.session_id || 'export'}-${timestamp}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
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
        <h2 style={{ fontSize: 21, fontWeight: 700, marginBottom: 16 }}>이미 이 회의로 주제를 확정했어요</h2>
        {saveError && <p style={{ color: 'var(--coral)', fontSize: 14.5, marginBottom: 12 }}>{saveError}</p>}
        <button
          type="button"
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

  // 용준/Claude(2026-07-25, 요청: 상단 상태 요약 바) — "회의 진행 시간"은 서버가 주는
  // 값이 없어(세션에 시작 타임스탬프 필드 자체가 없음) 하드코딩하지 않고 뺐다. 대신 실제로
  // 있는 값(라운드, 참여 위원 수, 논의 중인 아이디어 수, 다음 단계 라벨)만 보여준다.
  const nextStepLabel = phase === 'finalized' ? '완료' : '주제 확정'
  const metaItems = ideationConv
    ? [
        // 용준/Claude(2026-07-26, 요청: "/3은 필요없고 실시간 라운드 숫자만, 4라운드
        // 넘어가면 4로") — max_rounds는 백엔드가 무한 루프를 막는 안전 상한일 뿐
        // "정확히 이 라운드까지"라는 목표치가 아니므로 더 이상 분모로 보여주지 않는다.
        // ideationConv.round 값 자체를 그대로 보여주면 라운드가 늘어날 때마다 자동으로
        // 최신 숫자가 표시된다.
        { icon: ListChecks, label: '진행 라운드', value: `${ideationConv.round ?? 0}` },
        { icon: Users, label: '참여 위원', value: '3명' },
        { icon: Lightbulb, label: '논의 아이디어', value: `${ideationConv.idea_candidates?.length ?? 0}개` },
        { icon: ArrowRight, label: '다음 단계', value: nextStepLabel },
      ]
    : []

  return (
    <div className="rb-ideation-layout">
      <StreamingCursorStyle />
      {formSelectionPrompt && (
        <ApplicationFormFieldSelectModal
          items={formSelectionPrompt.items}
          onConfirm={handleConfirmFormSelection}
        />
      )}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
          {onBack && (
            <button type="button" className="btn-ghost" style={{ padding: '5px 10px', fontSize: 13.5 }} onClick={onBack} disabled={busy}>
              ← 이전
            </button>
          )}
          {ideationConv && (
            <button
              type="button"
              className="btn-ghost"
              style={{ padding: '5px 10px', fontSize: 12.5, display: 'inline-flex', alignItems: 'center', gap: 5 }}
              onClick={handleExportAnalysisJson}
              title="분석용 JSON 내보내기"
            >
              <Download size={13} /> JSON 내보내기
            </button>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 2 }}>
          <h2 style={{ fontSize: 27, fontWeight: 700, color: 'var(--text-0)', letterSpacing: '-0.02em' }}>AI 아이디어 회의</h2>
          {ideationConv && (
            <span className="badge amber mono">
              {statusLabelFor({ phase, starting, sending, finalizing, interrupting })}
            </span>
          )}
        </div>
        <div style={{ fontSize: 16, fontWeight: 500, color: '#625d72', marginBottom: 14, lineHeight: 1.5 }}>
          공모전 분석 결과를 바탕으로 진행위원, 기획 위원, 개발 위원이 함께 아이디어를 논의하고 있습니다.
        </div>
        {ideationConv?.competition_name && (
          <div
            style={{
              display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16,
              padding: '12px 16px', borderRadius: 10,
              background: '#fff', border: '1px solid var(--border-1, #e4e0ee)',
            }}
          >
            <span
              style={{
                flexShrink: 0, fontSize: 12, fontWeight: 700, color: '#918d9f',
                letterSpacing: '0.02em', border: '1px solid #dcd7ea', borderRadius: 6,
                padding: '3px 8px',
              }}
            >
              이번 회의 주제
            </span>
            <span style={{ fontSize: 20, fontWeight: 800, color: '#111', letterSpacing: '-0.01em' }}>
              {ideationConv.competition_name}
            </span>
          </div>
        )}

        <ErrorBanner error={phaseFailure || error} onRetry={handleRestart} />

        {/* 용준/Claude(2026-07-25, 요청: "회의 대화 영역 상단에 가로형 상태 요약 카드") —
            round/idea_candidates.length/phase는 전부 ideationConv에 이미 있는 실제 값이다. */}
        {metaItems.length > 0 && (
          <div className="rb-ideation-meta">
            {metaItems.map(({ icon: Icon, label, value }) => (
              <div key={label} className="card glass rb-ideation-meta-item">
                <div className="rb-ideation-meta-label"><Icon size={11} /> {label}</div>
                <div className="rb-ideation-meta-value">{value}</div>
              </div>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-0)' }}>회의 대화</div>
          {ideationConv && phase !== 'finalized' && phase !== 'failed' && (
            <span className="badge green mono" style={{ fontSize: 12 }}>실시간</span>
          )}
        </div>
        <div className="card glass" style={{ minHeight: 360, maxHeight: 520, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4, padding: 16 }}>
          {starting && !ideationConv && (
            <p className="rb-ideation-notice">
              {startPhaseLabel ? `${startPhaseLabel}...` : '공모전 분석을 바탕으로 아이디어 후보를 만들고 있어요...'}
            </p>
          )}
          {interruptionMarkers
            .filter((marker) => marker.afterMessageId === null)
            .map((marker) => <InterruptionMarker key={marker.markerId} speakerId={marker.speakerId} />)}
          {canonicalMessages.map((m) => (
            <Fragment key={m.message_id}>
              <MessageBubble
                message={m}
                allMessages={visibleMessages}
                isLatest={m.message_id === latestVisibleMessageId}
              />
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
            <MessageBubble
              message={optimisticUserMessage}
              allMessages={visibleMessages}
              isLatest={optimisticUserMessage.message_id === latestVisibleMessageId}
            />
          )}
          {/* 스트리밍 임시 메시지 — message_start를 받는 즉시 말풍선이 생기고, 실제 LLM
              델타가 도착하는 대로 안에서 텍스트가 자란다(완성 후 재생하는 효과 아님).
              streamState는 최종 state 이벤트가 오면 즉시 비워지므로, 이 목록과 위
              ideationConv.messages가 같은 내용으로 동시에 남아 중복되는 순간은 없다. */}
          {streamState.messages.map((m) => (
            <MessageBubble
              key={m.message_id}
              message={m}
              streaming
              allMessages={visibleMessages}
              isLatest={m.message_id === latestVisibleMessageId}
            />
          ))}
          {sending && streamState.messages.length === 0 && (
            <p className="rb-ideation-notice">
              {streamState.phaseLabel || `${statusLabelFor({ phase, starting, sending, finalizing })}...`}
            </p>
          )}
          {finalizing && (
            <p className="rb-ideation-notice">
              {statusLabelFor({ phase, starting, sending, finalizing })}...
            </p>
          )}
          {/* 용준/Claude(2026-07-25, 요청: "아이디어 후보를 채팅 영역 안으로") — 오른쪽은
              참여 위원 전용 공간으로 비우고, 후보 카드는 진행자의 선택 안내 메시지
              바로 아래(대화 흐름 안)에 특수 메시지 형태로 보여준다. phase가
              awaiting_candidate_selection인 동안은 항상 그 마지막 메시지가 이 선택
              질문이므로, 메시지 목록 맨 끝에 붙이면 자연히 그 버블 바로 아래에 온다.
              선택 확정(handleSend → candidateSelectMessage)은 기존 로직 그대로다 —
              카드 클릭도 텍스트 "n번" 입력과 같은 API 호출을 탄다. */}
          {phase === 'awaiting_candidate_selection' && hasCandidates && (
            <div style={{ marginTop: 4 }}>
              <div style={{ fontSize: 13.5, color: 'var(--text-2)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                주제 후보
              </div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: ideationConv.idea_candidates.length > 1 ? 'repeat(auto-fit, minmax(220px, 1fr))' : '1fr',
                  gap: 10,
                }}
              >
                {ideationConv.idea_candidates.map((c, i) => (
                  <CandidateCard
                    key={c.candidate_id || i}
                    candidate={c}
                    index={i}
                    selected={isCandidateSelected(c)}
                    onSelect={(idx) => handleSend(candidateSelectMessage(idx))}
                    disabled={!canReplyOrContinue || hasSelected}
                  />
                ))}
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
            <button type="button" className="btn-ghost" style={{ fontSize: 13.5, borderColor: 'var(--amber, var(--coral))' }} onClick={handleInterject}>
              잠시만
            </button>
          </div>
        )}

        {/* 의견 대상 선택 UI — 답변자를 고르는 것처럼 보이지 않도록 "누구의 의견에
            반응하는지"를 명시한다. 선택된 위원이 먼저 답하고 상대 위원이 이어서 검토한다. */}
        {awaitingInterjectTarget && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 13.5, color: 'var(--text-2)', marginBottom: 6 }}>
              어느 의견에 대해 말씀하시겠어요?
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button type="button" className="btn-ghost" style={{ fontSize: 13.5 }} onClick={() => setInterjectTarget('planning_expert')}>
                기획 위원 의견에
              </button>
              <button type="button" className="btn-ghost" style={{ fontSize: 13.5 }} onClick={() => setInterjectTarget('dev_expert')}>
                개발 위원 의견에
              </button>
              <button type="button" className="btn-ghost" style={{ fontSize: 13.5 }} onClick={() => setInterjectTarget('both')}>
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
                    : phase === 'awaiting_candidate_selection'
                      ? '답변을 입력하거나 후보를 선택해 주세요.'
                      : phase === 'awaiting_user_decision'
                        ? '필요하면 의견을 남겨주세요 (선택 사항)'
                        : hasSelected
                          ? '선택한 아이디어에 대해 추가 의견을 입력해 주세요.'
                          : '답변을 입력하세요')
              }
              disabled={!canReplyOrContinue && !chosenInterjectTarget && !awaitingInterjectTarget}
              style={{ flex: 1, background: 'var(--bg-1)', border: '1px solid var(--glass-border)', borderRadius: 10, padding: '10px 14px', color: 'var(--text-0)', fontSize: 15.5 }}
            />
            <button
              type="button"
              aria-label="메시지 보내기"
              className="btn-primary"
              style={{ padding: '10px 14px' }}
              disabled={(!canReplyOrContinue && !chosenInterjectTarget) || !draft.trim()}
              onClick={() => (chosenInterjectTarget ? handleSend(undefined, chosenInterjectTarget) : handleSend())}
            >
              <Send size={14} />
            </button>
          </div>
        )}

        {/* 선택한 아이디어 안내는 채팅 입력창 아래에 표시한다. selected_idea는 백엔드가
            확정한 값이며, 선택 취소 API나 phase 되돌리기 동작은 제공하지 않는다. */}
        {ideationConv?.selected_idea?.title && (
          <div
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
              marginTop: 12, padding: '13px 16px', borderRadius: 12,
              background: 'var(--bg-1)', border: '1px solid var(--glass-border)',
            }}
          >
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)' }}>선택한 아이디어</div>
              <div style={{ marginTop: 2, fontSize: 17, fontWeight: 600, color: 'var(--text-0)' }}>
                {ideationConv.selected_idea.title}
              </div>
              <div style={{ marginTop: 3, fontSize: 14.5, fontWeight: 500, color: 'var(--text-2)', lineHeight: 1.65 }}>
                현재 이 아이디어를 중심으로 회의 중입니다.
              </div>
            </div>
            <CheckCircle2 size={21} color="var(--text-2)" style={{ flexShrink: 0 }} />
          </div>
        )}

        {/* "전문가 추천"은 후보 자체를 새로 만드는 게 아니라 다른 답변 경로라 입력창
            아래에 남긴다. "다시 추천"은 후보 카드가 오른쪽 패널로 옮겨가면서 이 자리로
            같이 옮겨왔다(기존 handleSend(REGENERATE_MESSAGE) 로직 그대로). */}
        {phase === 'awaiting_candidate_selection' && (
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button type="button" className="btn-ghost" style={{ fontSize: 13.5 }} disabled={!canReplyOrContinue} onClick={() => handleSend(EXPERT_RECOMMEND_MESSAGE)}>
              전문가 추천
            </button>
            <button
              type="button"
              className="btn-ghost"
              style={{ fontSize: 13.5, display: 'flex', alignItems: 'center', gap: 4 }}
              disabled={!canReplyOrContinue}
              onClick={() => handleSend(REGENERATE_MESSAGE)}
            >
              <RefreshCw size={12} /> 다시 추천
            </button>
          </div>
        )}
      </div>

      {/* 용준/Claude(2026-07-25, 요청: "오른쪽 영역은 참여 AI 휴먼 전용 공간으로 사용") —
          아이디어 제안 목록은 채팅 안으로 옮겨갔으니 오른쪽 패널에는 더 이상 두지 않는다.
          아바타 패널을 이 열의 시각적 핵심으로 둔다. 데스크톱에서는 sticky, 태블릿
          이하에서는 중앙 아래로 내려간다(위 rb-ideation-side 참고).
          용준/Claude(2026-07-26, 요청: "아이디어 기획 캔버스는 참여 위원 오른쪽에") —
          캔버스/합의사항/CTA는 이 열 아래에 이어붙이지 않고 별도의 세 번째 열
          (rb-ideation-canvas-col)로 분리했다. */}
      <div className="rb-ideation-side">
        <div className="card glass">
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-0)', marginBottom: 3 }}>참여 위원</div>
          <div style={{ fontSize: 14.5, fontWeight: 500, color: '#625d72', lineHeight: 1.65, marginBottom: 12 }}>
            진행위원 · 기획 위원 · 개발 위원이 실시간으로 함께 회의해요.
          </div>
          {/* 재인/Claude(2026-07-26, dev #166 병합): 용준님이 만든 이 레이아웃(참여 위원
              카드 안에 아바타 배치)은 그대로 두되, prop만 새 구조로 교체했다. 예전
              onNeedNextSpeaker(페이싱 타이머가 "다음 화자 불러줘"를 알리던 방식)는
              제거됐고(handleAvatarNeedNextSpeaker 함수 자체가 없어서 그대로 두면
              ReferenceError), 지금은 텍스트를 미리 받아두고 아바타가 실제로 재생을
              시작할 때마다 onRevealed로 하나씩 공개하는 방식이다. */}
          {IDEATION_AVATAR_ENABLED ? (
            <IdeationAvatarStage
              playQueue={avatarPlayQueue}
              onRevealed={() => setAvatarRevealedCount((n) => n + 1)}
            />
          ) : (
            // 가은/Claude(2026-07-26): VITE_IDEATION_AVATAR_ENABLED=false일 때는
            // IdeationAvatarStage를 아예 마운트하지 않는다 — 마운트하면 매번 없는
            // 영상 서버에 WebSocket 연결을 시도해 콘솔에 에러만 쌓인다.
            <div style={{ fontSize: 12.5, color: 'var(--text-2)', padding: '10px 0' }}>
              개발 모드: 영상 없이 회의 진행 중 (VITE_IDEATION_AVATAR_ENABLED=false)
            </div>
          )}
        </div>
      </div>

      <div className="rb-ideation-canvas-col">
        <MergeAnalysisPanel
          mergeAnalysis={ideationConv?.merge_analysis}
          sourceCandidates={ideationConv?.source_candidates}
          userSelectionMessage={ideationConv?.user_selection_message}
        />

        {/* 가은/Claude(2026-07-26, dev #131-167 merge 후 재이식) — 신청서 항목 선택
            모달에서 확정한 항목을 회의 중에도 계속 보여준다(items가 비어있으면
            ApplicationFormPanel 자체가 null을 반환해 숨는다). dev 쪽엔
            application_form_draft(항목별 실시간 작성 상태) 데이터 모델이 없어서
            draft는 넘기지 않는다 — 전부 "아직 작성되지 않았어요"로 보이는 대신, 최소
            "이 항목들을 준비해야 한다"는 목록 자체는 다시 보인다. */}
        <ApplicationFormPanel items={applicationFormItems} />

        <IdeaCanvasPanel ideationConv={ideationConv} analysis={announcementAnalysis} />

        {ideationConv && (ideationConv.consensus?.length > 0 || ideationConv.unresolved_issues?.length > 0) && (
          <div className="card glass" style={{ marginBottom: 12, padding: 14 }}>
            {ideationConv.consensus?.length > 0 && (
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 15, fontWeight: 700, color: '#514a61', marginBottom: 4 }}>합의 사항</div>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 15.5, fontWeight: 500, color: 'var(--text-0)', lineHeight: 1.7 }}>
                  {ideationConv.consensus.map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              </div>
            )}
            {ideationConv.unresolved_issues?.length > 0 && (
              <div>
                <div style={{ fontSize: 15, fontWeight: 700, color: '#514a61', marginBottom: 4 }}>미해결 쟁점</div>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 15.5, fontWeight: 500, color: 'var(--text-0)', lineHeight: 1.7 }}>
                  {ideationConv.unresolved_issues.map((u, i) => <li key={i}>{u}</li>)}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* 용준/Claude(2026-07-25, 요청: "버튼은 한 곳에만, 상태별 문구를 명확하게") —
            handleFinalize/canFinalize는 그대로고, 라벨과 비활성 사유만 3단계로 나눈다.
            "회의를 바탕으로 주제 확정하기"는 후보는 골랐지만 위원 논의가 아직 끝나지
            않은 상태를 알려주는 라벨일 뿐, 이 버튼 자체가 그 논의를 진행시키지는
            않는다(논의는 채팅으로 계속된다) — 그래서 이 상태에서도 버튼은 비활성이다. */}
        <div style={{ fontSize: 14.5, fontWeight: 500, color: '#514a61', marginBottom: 8, lineHeight: 1.65 }}>
          지금까지 논의된 내용을 바탕으로 최종 주제를 확정하고 다음 단계로 이동합니다.
        </div>
        <button
          type="button"
          className="btn-primary"
          style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
          disabled={!canFinalize}
          onClick={handleFinalize}
        >
          <Sparkles size={14} />
          {finalizing
            ? '초안 생성 중...'
            : canFinalize
              ? '주제 확정 단계로 이동하기 →'
              : '회의를 바탕으로 주제 확정하기 →'}
        </button>
        {!canFinalize && ideationConv && phase !== 'finalized' && phase !== 'failed' && (
          <p style={{ fontSize: 14.5, fontWeight: 600, color: '#514a61', marginTop: 9, lineHeight: 1.55 }}>
            {!hasSelected && hasCandidates ? '먼저 아이디어 후보를 선택해 주세요.' : nextActionGuideFor(phase)}
          </p>
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
        <h2 style={{ fontSize: 21, fontWeight: 700, marginBottom: 16 }}>주제 발전 회의를 먼저 완료해 주세요</h2>
        <p style={{ fontSize: 14.5, color: 'var(--text-2)' }}>
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
        <button type="button" className="btn-ghost" style={{ marginBottom: 12, padding: '5px 10px', fontSize: 13.5 }} onClick={onBack}>
          ← 이전
        </button>
      )}
      <h2 style={{ fontSize: 23, fontWeight: 700, marginBottom: 20 }}>{proposal.idea_name || '확정된 주제'}</h2>

      <div className="card glass">
        {PROPOSAL_ROWS.map(([key, label], i) => (
          <div key={key} style={{ display: 'grid', gridTemplateColumns: '150px 1fr', gap: 16, padding: '14px 0', borderTop: i > 0 ? '1px solid var(--glass-border)' : 'none' }}>
            <div style={{ fontSize: 13.5, color: 'var(--text-2)', fontFamily: 'var(--mono)' }}>{label}</div>
            <div style={{ fontSize: 14.5, lineHeight: 1.6 }}>{proposalValueDisplay(proposal[key])}</div>
          </div>
        ))}
      </div>

      {hasDiscoveryHistory && (
        <div className="card glass" style={{ marginTop: 16 }}>
          <div style={{ fontSize: 14.5, fontWeight: 700, marginBottom: 12 }}>아이디어 발굴 이력</div>

          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-2)', marginBottom: 6 }}>최초 후보</div>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 14, lineHeight: 1.7 }}>
              {originalCandidates.map((c) => <li key={c.candidate_id}>{c.title}</li>)}
            </ul>
          </div>

          {ideationConv.selected_idea && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-2)', marginBottom: 6 }}>선택하거나 결합한 후보</div>
              <div style={{ fontSize: 14, lineHeight: 1.6 }}>{ideationConv.selected_idea.title}</div>
            </div>
          )}

          {ideationConv.selection_reason && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-2)', marginBottom: 6 }}>선택 이유</div>
              <div style={{ fontSize: 14, lineHeight: 1.6 }}>{ideationConv.selection_reason}</div>
            </div>
          )}

          {ideationConv.user_selection_message && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-2)', marginBottom: 6 }}>사용자 원문 선택 요청</div>
              <div style={{ fontSize: 14, lineHeight: 1.6 }}>“{ideationConv.user_selection_message}”</div>
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
