import { Fragment, useEffect, useRef, useState } from 'react'
import { AlertCircle, ArrowRight, CheckCircle2, ChevronDown, ChevronUp, Circle, ClipboardList, ExternalLink, History, Lightbulb, ListChecks, LoaderCircle, Newspaper, RefreshCw, Send, Sparkles, Users, X } from 'lucide-react'
import {
  cancelIdeationConversation,
  continueIdeationExpertTurnStream,
  finalizeIdeationConversation,
  generateApplicationFormDraft,
  getIdeationConversation,
  getLatestIdeationConversation,
  replyIdeationConversation,
  replyIdeationConversationStream,
  retryFailedIdeationConversationNode,
  startIdeationConversation,
  startIdeationConversationStream,
} from '../../api/ideationConversationApi'
import { getAnnouncementAnalysis, getApplicationFormAnalysis } from '../../api/documentApi'
import IdeaCanvasPanel from './IdeaCanvasPanel'
import IdeationAvatarStage from './IdeationAvatarStage'
import IdeaEvolutionTimeline from './IdeaEvolutionTimeline'
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
  discussionIdeaCountFor,
  humanizeExpertIdentifiers,
  nextActionGuideFor,
  nextStepLabelFor,
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
  // 용준/Claude(2026-07-27, 요청: discovery 모드 구조화 액션 버튼) — 이 세 phase는
  // action_code 계약(select_problem_focus/combine_problem_focus, merge_directions/
  // drop_direction/add_solution_direction/proceed_to_validation/return_to_problem_definition,
  // confirm_concept/revise_candidate/return_to_problem_definition)이 유효한 지점이다.
  // handleSend가 canReplyOrContinue를 통과해야 실제 reply를 보낼 수 있으므로, 새 버튼들도
  // 자유 텍스트 입력창과 동일하게 이 목록에 포함해야 동작한다.
  'awaiting_problem_focus_selection',
  'awaiting_conflict_resolution',
  'awaiting_concept_confirmation',
])

// 재인/Claude(2026-07-23): 아바타 재생 대상 화자 - user는 당연히 제외, 그 외 3명
// (진행자/기획/개발)만 IdeationAvatarStage의 AVATAR_SLOTS에 얼굴·목소리가 등록돼 있다.
const AVATAR_SPEAKER_IDS = new Set(['ideation_facilitator', 'planning_expert', 'dev_expert'])

// 용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") — 사용자
// 입력 없이 "다음 위원 발언 1건만 이어서 요청"하는 eager-fetch 루프가 적용되는 phase
// 목록. expert_discussion(1차 라운드테이블)에 이어 idea_validation(후보 검증, 기획→개발
// 순차 진행)도 같은 패턴을 쓴다 — backend continue_expert_turn_stream이 phase로
// continue_ideation_expert_turn/continue_ideation_validation_turn을 나눠 부른다.
const EAGER_FETCH_TURN_PHASES = ['expert_discussion', 'idea_validation']

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
      @keyframes rb-ideation-spin { to { transform:rotate(360deg); } }
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
        display:grid;
        width:100%; min-height:calc(100vh - 64px);
        grid-template-columns:minmax(460px, 1fr) minmax(500px, 1fr);
        gap:32px; align-items:stretch; max-width:1680px;
      }
      .rb-ideation-main-contents{ min-width:0; display:flex; flex-direction:column; }
      .rb-idea-header{ min-width:0; }
      .rb-idea-topic{ margin:0 0 14px !important; }
      .rb-idea-chat-panel{ min-width:0; display:flex; flex-direction:column; }
      .rb-idea-chat-panel .rb-idea-chat-scroll{
        position:relative; flex:none; min-height:220px; max-height:78vh !important;
        resize:vertical !important; overflow:auto !important; flex-shrink:0;
      }
      .rb-idea-finalize{ min-width:0; margin-top:24px; }
      .rb-idea-right{
        min-width:0; min-height:0; height:calc(100vh - 64px);
        display:grid; grid-template-rows:minmax(0, 1fr) minmax(0, 1fr);
        gap:16px;
      }
      .rb-ideation-side{ position:static; margin:0; min-width:0; min-height:0; display:flex; flex-direction:column; overflow:auto; }
      .rb-avatar-card{ padding:0 !important; border:none !important; background:transparent !important; box-shadow:none !important; backdrop-filter:none !important; }
      .rb-ideation-canvas-col{
        position:static; margin:0; display:flex; flex-direction:column; gap:12px;
        min-height:0; height:100%; overflow-y:auto; padding-right:6px;
        scrollbar-width:thin; scrollbar-color:rgba(124,92,234,.35) transparent;
      }
      .rb-ideation-meta{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }
      .rb-ideation-meta-item{ flex:1 1 120px; border-radius:12px; padding:10px 12px; }
      .rb-ideation-meta-label{ display:flex; align-items:center; gap:5px; margin-bottom:3px; }
      .rb-ideation-meta-value{ font-weight:700; color:var(--text-0); }
      .rb-canvas-tab-shell{ height:100%; min-height:0; display:flex; flex-direction:column; overflow:hidden !important; }
      .rb-canvas-tab-scroll{
        flex:1; min-height:0; overflow-y:auto; overscroll-behavior:contain;
        scrollbar-width:thin; scrollbar-color:rgba(124,92,234,.35) transparent;
      }
      .rb-chat-resize-handle{
        position:static; align-self:flex-end; flex-shrink:0; width:22px; height:22px;
        margin:auto -8px -8px 0; cursor:ns-resize; touch-action:none;
        display:flex; align-items:flex-end; justify-content:flex-end;
        color:var(--text-2); user-select:none; z-index:4;
      }
      @media (max-width: 1180px){
        .rb-ideation-layout{
          grid-template-columns:minmax(0,1fr);
          min-height:0;
        }
        .rb-idea-right{ height:auto; grid-template-rows:auto minmax(380px, 560px); }
        .rb-ideation-side{ overflow:visible; }
        .rb-ideation-canvas-col{ max-height:560px; }
        .rb-idea-chat-panel .rb-idea-chat-scroll{ min-height:420px; }
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
// 용준/Claude(2026-07-29, 요청: target/criteria/외부근거/전문가판단 분리) — 근거 카드에
// 붙는 작은 배지. "검토 대상"(target)은 사용자가 지금 검토받는 아이디어 원문일 뿐 외부
// 근거가 아니므로, "외부 근거"와 같은 스타일로 섞여 보이면 안 된다는 것이 이번 변경의
// 핵심이다 — 색으로도 구분한다(검토 대상은 중립, 공모전 근거/외부 근거는 근거 계열 색).
const EVIDENCE_BADGE_META = {
  target: { label: '검토 대상', color: 'var(--text-2)', bg: 'var(--bg-0)' },
  criteria: { label: '공모전 근거', color: '#5e3ec8', bg: 'rgba(94,62,200,0.08)' },
  external: { label: '외부 근거', color: '#0f766e', bg: 'rgba(15,118,110,0.08)' },
}

function EvidenceBadge({ kind }) {
  const meta = EVIDENCE_BADGE_META[kind]
  if (!meta) return null
  return (
    <span
      style={{
        display: 'inline-block', fontSize: 11, fontWeight: 700, color: meta.color,
        background: meta.bg, borderRadius: 999, padding: '1px 7px', marginBottom: 4,
      }}
    >
      {meta.label}
    </span>
  )
}

// source_type(백엔드가 compose_ideation_evidence_pool에서 매긴 값)별로 화면에 보여줄
// 부가 정보(발행 기관/일자)를 뽑는다 — 값이 없으면 지어내지 않고 그냥 생략한다.
function externalEvidenceMeta(item) {
  const organization = item.organization || item.publisher
  const publishedAt = item.published_at || item.reference_date
  const parts = [organization, publishedAt].filter(Boolean)
  return parts.length > 0 ? parts.join(' · ') : null
}

// 용준/Claude(2026-07-30, 요청: 공고 URL 본문/URL 첨부파일/직접 업로드 구분) —
// origin_type(ai/rag/chunking/schemas.py::SourceType 값을 그대로 노출한 것)을 사람이 읽는
// 라벨로 바꾼다. 값이 없거나 매핑에 없으면 아무것도 표시하지 않는다(지어내지 않음).
const ORIGIN_TYPE_LABEL = {
  url_webpage: '공고 URL 본문',
  url_attachment: 'URL 첨부파일',
  file_upload: '직접 업로드',
}

function EvidenceCard({ item, kind }) {
  const title = item.document_title || item.document_name || item.title
  const page = item.page ?? item.page_number
  const section = item.section || item.page_or_section
  const meta = kind === 'external' ? externalEvidenceMeta(item) : null
  const originLabel = kind === 'criteria' ? ORIGIN_TYPE_LABEL[item.origin_type] : null
  const fileName = kind === 'criteria' ? item.file_name : null
  return (
    <div
      style={{
        fontSize: 13, color: 'var(--text-1)', lineHeight: 1.5,
        background: 'var(--bg-0)', border: '1px solid var(--glass-border)', borderRadius: 8, padding: '6px 8px',
      }}
    >
      <EvidenceBadge kind={kind} />
      {title && (
        <div style={{ fontWeight: 600, color: 'var(--text-2)', marginBottom: 2 }}>
          {title}
          {page != null && ` / ${page}페이지`}
        </div>
      )}
      {(originLabel || fileName) && (
        <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 2 }}>
          {[originLabel, fileName].filter(Boolean).join(' · ')}
        </div>
      )}
      {section && (
        <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 2 }}>
          {kind === 'criteria' ? `섹션: ${section}` : section}
        </div>
      )}
      {meta && <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 2 }}>{meta}</div>}
      {(item.short_summary || item.text || item.quote) && (
        <div style={{ marginBottom: 2 }}>"{item.short_summary || item.text || item.quote}"</div>
      )}
      {item.source_url && (
        <a href={item.source_url} target="_blank" rel="noreferrer" style={{ fontSize: 13.5, fontWeight: 600, color: '#5e3ec8' }}>
          원문 보기
        </a>
      )}
    </div>
  )
}

function EvidenceToggle({
  evidence,
  linkedEvidenceRefs,
  claims,
  reviewedTargetRefs,
  linkedCriteriaRefs,
  linkedExternalEvidenceRefs,
  expertJudgmentWithoutExternalEvidence,
}) {
  const [open, setOpen] = useState(false)
  const evidenceByChunkId = new Map((evidence || []).filter((e) => e && e.chunk_id).map((e) => [e.chunk_id, e]))
  const resolveRefs = (refs) =>
    (refs || []).map((chunkId) => evidenceByChunkId.get(chunkId)).filter(Boolean)

  // 용준/Claude(2026-07-29) — 4개 신규 필드는 _build_message가 항상 채우지만(빈 리스트
  // 기본값), 이 배포 이전에 저장된 세션은 필드 자체가 없다(undefined). 그런 레거시
  // 메시지만 예전 방식(구분 없는 단일 "근거 N건" 목록)으로 폴백한다.
  const bucketsProvided =
    reviewedTargetRefs !== undefined ||
    linkedCriteriaRefs !== undefined ||
    linkedExternalEvidenceRefs !== undefined ||
    expertJudgmentWithoutExternalEvidence !== undefined

  if (!bucketsProvided) {
    const linkedSet = new Set(linkedEvidenceRefs || [])
    const items = (evidence || []).filter((e) => e && e.chunk_id && linkedSet.has(e.chunk_id))
    const unlinkedClaims = (claims || []).filter(
      (c) => c && (c.claim_type === 'expert_judgment' || !linkedSet.size)
    )
    if (items.length === 0 && unlinkedClaims.length === 0) return null
    if (items.length === 0) {
      return <ExpertJudgmentPill />
    }
    return (
      <EvidenceListPanel
        count={items.length}
        open={open}
        onToggle={() => setOpen((v) => !v)}
        renderItems={() => items.map((e, i) => <EvidenceCard key={e.chunk_id || i} item={e} kind="criteria" />)}
      />
    )
  }

  const targetItems = resolveRefs(reviewedTargetRefs)
  const criteriaItems = resolveRefs(linkedCriteriaRefs)
  const externalItems = resolveRefs(linkedExternalEvidenceRefs)
  const totalCount = criteriaItems.length + externalItems.length + targetItems.length

  if (totalCount === 0) {
    if ((expertJudgmentWithoutExternalEvidence || []).length > 0 || (claims || []).length > 0) {
      return <ExpertJudgmentPill />
    }
    return null
  }

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
        근거 {totalCount}건 {open ? '접기' : '보기'}
      </button>
      {open && (
        <div style={{ marginTop: 4, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {/* 표시 순서: 공모전 근거 -> 외부 근거(둘 다 "증거" 계열) -> 검토 대상(참고용,
              증거가 아니라는 것을 시각적으로도 마지막에 배치해 드러낸다). */}
          {criteriaItems.map((e, i) => <EvidenceCard key={`criteria-${e.chunk_id || i}`} item={e} kind="criteria" />)}
          {externalItems.map((e, i) => <EvidenceCard key={`external-${e.chunk_id || i}`} item={e} kind="external" />)}
          {targetItems.map((e, i) => <EvidenceCard key={`target-${e.chunk_id || i}`} item={e} kind="target" />)}
        </div>
      )}
    </div>
  )
}

function ExpertJudgmentPill() {
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
      전문가 판단 · 외부 근거 없음
    </div>
  )
}

function EvidenceListPanel({ count, open, onToggle, renderItems }) {
  return (
    <div style={{ marginTop: 4 }}>
      <button
        type="button"
        onClick={onToggle}
        style={{
          background: 'none', border: 'none', padding: 0, cursor: 'pointer',
          fontSize: 13, color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: 2,
        }}
      >
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        근거 {count}건 {open ? '접기' : '보기'}
      </button>
      {open && <div style={{ marginTop: 4, display: 'flex', flexDirection: 'column', gap: 6 }}>{renderItems()}</div>}
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
            {agreements.map((a, i) => <li key={i}>{humanizeExpertIdentifiers(a)}</li>)}
          </ul>
        </div>
      )}
      {disagreements.length > 0 && (
        <div>
          <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>남은 쟁점</strong>
          <ul style={{ margin: '2px 0 0', paddingLeft: 16, lineHeight: 1.6 }}>
            {disagreements.map((d, i) => <li key={i}>{humanizeExpertIdentifiers(d)}</li>)}
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
  const rawText = streaming ? message.displayedContent ?? '' : message.content
  const text = humanizeExpertIdentifiers(rawText)
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
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: isRight ? 'flex-end' : 'flex-start', gap: 6, marginBottom: 4 }}>
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
            reviewedTargetRefs={message.reviewed_target_refs}
            linkedCriteriaRefs={message.linked_criteria_refs}
            linkedExternalEvidenceRefs={message.linked_external_evidence_refs}
            expertJudgmentWithoutExternalEvidence={message.expert_judgment_without_external_evidence}
          />
        )}
        {isFacilitatorSummary && <FacilitatorSummaryCard structured={message.structured} />}
      </div>
    </div>
  )
}


// 용준/Claude(2026-07-28, 요청: "검증 단계도 진짜 대화하듯이 보였으면 좋겠다") — 검증
// (idea_validation) 단계는 백엔드가 LLM 호출 1번으로 기획·개발 두 위원의 발언을 한꺼번에
// 만들어 finalState로 통째로 던져준다(질문/토론 노드와 달리 message_start/delta로 실시간
// 스트리밍되지 않는다 — backend/app/api/routes/ideation_conversation_streaming.py의
// _PHASE_ONLY_LABELS에 "[검증 규칙]"이 phase 문구로만 잡혀있기 때문). 서버 스트리밍을
// 새로 만드는 대신, 이미 다 받은 텍스트를 화면에서만 한 글자씩 "재생"해 실제 스트리밍과
// 같은 느낌을 낸다 — ideationStreamReducer.js의 charsPerTickFor와 같은 속도 곡선을 쓴다.
function isValidationOpinionMessage(message) {
  return (
    message?.message_type === 'opinion'
    && !!message?.structured?.validation
    && (message.speaker_id === 'planning_expert' || message.speaker_id === 'dev_expert')
  )
}

function TypedMessageBubble({ message, allMessages, isLatest, onDone, onProgress }) {
  const [displayed, setDisplayed] = useState('')
  const onDoneRef = useRef(onDone)
  onDoneRef.current = onDone
  const onProgressRef = useRef(onProgress)
  onProgressRef.current = onProgress

  useEffect(() => {
    let cancelled = false
    let shown = ''
    let rafId
    const full = message.content || ''
    function tick() {
      if (cancelled) return
      if (shown.length >= full.length) {
        onDoneRef.current?.()
        return
      }
      const pending = full.length - shown.length
      shown = full.slice(0, shown.length + charsPerTickFor(pending))
      setDisplayed(shown)
      onProgressRef.current?.()
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)
    return () => {
      cancelled = true
      if (rafId) cancelAnimationFrame(rafId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [message.message_id])

  return (
    <MessageBubble
      message={{ ...message, displayedContent: displayed }}
      streaming
      allMessages={allMessages}
      isLatest={isLatest}
    />
  )
}

const EXPERT_LABELS = {
  planning_expert: '기획 의원',
  dev_expert: '개발 의원',
}
const EXPECTED_DIFFICULTY_LABEL = { high: '낮음', medium: '보통', low: '높음' }

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
          {candidate.innovation_axis && (
            <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>혁신 축 · </strong>
              {candidate.innovation_axis}
            </div>
          )}
          {candidate.novel_mechanism && (
            <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>핵심 참신성 · </strong>
              {candidate.novel_mechanism}
            </div>
          )}
          {candidate.existing_approach && candidate.existing_limitation && (
            <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>기존 방식의 한계 · </strong>
              {candidate.existing_approach} — {candidate.existing_limitation}
            </div>
          )}
          {candidate.novelty_preservation && (
            <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>참신성 검증 MVP · </strong>
              {candidate.novelty_preservation}
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
          {candidate.source_direction_ids?.length > 0 && (
            <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>발전한 해결 방향 · </strong>
              {candidate.source_direction_ids.join(', ')}
            </div>
          )}
          {candidate.reflected_evolution_ids?.length > 0 && (
            <div style={{ fontSize: 14, color: 'var(--text-1)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--text-2)', fontWeight: 600 }}>반영한 반론·수정 · </strong>
              {candidate.reflected_evolution_ids.join(', ')}
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
          현재 예상 난이도 {EXPECTED_DIFFICULTY_LABEL[candidate.feasibility] || '검증 필요'}
        </div>
      )}
      {!selected && !disabled && (
        <div style={{ marginTop: 8, color: 'var(--purple)', fontSize: 13.5, fontWeight: 700 }}>
          검증 후보로 선택
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

function ErrorBanner({ error, onRetry, retrying }) {
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
          <button
            type="button"
            className="btn-ghost"
            style={{ marginTop: 10, padding: '6px 12px', fontSize: 13.5, display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={onRetry}
            disabled={retrying}
          >
            <RefreshCw size={12} /> {retrying ? '다시 시도 중...' : '다시 시도'}
          </button>
        )}
      </div>
    </div>
  )
}

// 용준/Claude(2026-07-27, 요청: discovery 모드 구조화 액션 버튼) — 아래 세 블록은
// awaiting_problem_focus_selection / awaiting_conflict_resolution /
// awaiting_concept_confirmation phase에서만 각각 렌더링된다(IdeationScreen 본문 참고).
// CandidateCard와 같은 시각 언어(card glass, 선택 카드, btn-primary/btn-ghost)를
// 따르지만, 후보 선택과 달리 다중 선택·여러 액션이 필요해 별도 로컬 컴포넌트로 둔다.
// 세 블록 모두 onSend(message, { actionCode, actionPayload })만 호출하고 API를 직접
// 부르지 않는다 — 실제 전송은 IdeationScreen의 handleSend가 담당한다.

function SelectableAreaCard({ area, selected, onToggle, disabled }) {
  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-pressed={selected}
      aria-disabled={disabled}
      onClick={() => !disabled && onToggle()}
      onKeyDown={(e) => {
        if (disabled || (e.key !== 'Enter' && e.key !== ' ')) return
        e.preventDefault()
        onToggle()
      }}
      className="card glass rb-ideation-candidate-card rb-ideation-candidate-card--interactive"
      style={{
        marginBottom: 10,
        padding: 14,
        cursor: disabled ? 'default' : 'pointer',
        border: selected ? '2px solid var(--purple)' : '1px solid var(--glass-border)',
        background: selected ? '#f5f1ff' : 'var(--bg-1)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8, marginBottom: 4 }}>
        <div style={{ fontSize: 16.5, fontWeight: 700, color: 'var(--text-0)' }}>{area.problem_title || area.title}</div>
        {selected ? (
          <CheckCircle2 size={18} color="var(--purple)" style={{ flexShrink: 0 }} />
        ) : (
          <Circle size={16} color="var(--glass-border)" style={{ flexShrink: 0 }} />
        )}
      </div>
      {(area.problem_description || area.summary) && (
        <div style={{ fontSize: 14.5, color: 'var(--text-1)', lineHeight: 1.6, marginBottom: 4 }}>
          <strong style={{ color: '#514a61', fontWeight: 700 }}>문제 상황 · </strong>
          {area.problem_description || area.summary}
        </div>
      )}
      {(area.affected_users || area.who_is_affected) && (
        <div style={{ fontSize: 13.5, color: 'var(--text-2)' }}>
          <strong style={{ fontWeight: 600 }}>영향을 받는 대상 · </strong>{area.affected_users || area.who_is_affected}
        </div>
      )}
      {area.planning_view && (
        <div style={{ fontSize: 13.5, color: 'var(--text-2)', marginTop: 5 }}>
          <strong style={{ fontWeight: 600 }}>기획 관점 · </strong>{area.planning_view}
        </div>
      )}
      {area.technical_view && (
        <div style={{ fontSize: 13.5, color: 'var(--text-2)', marginTop: 3 }}>
          <strong style={{ fontWeight: 600 }}>개발 관점 · </strong>{area.technical_view}
        </div>
      )}
    </div>
  )
}

// 용준/Claude(2026-07-28, 요청: RAG 출처를 시각적으로 보여달라) — 문제 영역 생성에
// 실제로 쓰인 external_evidence(RAG-006 프로젝트 문서 + RAG-007 네이버 뉴스/외부자료)를
// 클릭 가능한 출처 카드로 보여준다. 백엔드가 이미 publisher/source_url이 확인된 자료만
// 내려보내므로(ideation_external_evidence_service.py::_has_confirmed_source) 여기서는
// 추가 필터 없이 있는 그대로 렌더링한다. reference_only=True(확정 근거 아님) 원칙을
// 화면에도 그대로 드러내기 위해 안내 문구를 함께 둔다.
function ExternalEvidenceStrip({ items }) {
  if (!items || items.length === 0) return null
  return (
    <div style={{ marginTop: 10, marginBottom: 10 }}>
      <div style={{ fontSize: 12.5, color: 'var(--text-2)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 4 }}>
        <Newspaper size={13} /> 참고한 외부 자료 (확정 근거 아님, 참고용)
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {items.map((item, idx) => (
          <a
            key={item.source_id || item.chunk_id || idx}
            href={item.source_url || undefined}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 6,
              padding: '8px 10px',
              borderRadius: 10,
              border: '1px solid var(--glass-border)',
              background: 'var(--bg-1)',
              textDecoration: 'none',
              color: 'var(--text-1)',
              fontSize: 13,
              cursor: item.source_url ? 'pointer' : 'default',
            }}
          >
            <ExternalLink size={13} style={{ flexShrink: 0, marginTop: 2, color: 'var(--text-2)' }} />
            <span style={{ flex: 1, minWidth: 0 }}>
              <span
                style={{
                  display: 'block',
                  fontWeight: 600,
                  color: 'var(--text-0)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {item.title || '(제목 없음)'}
              </span>
              <span style={{ fontSize: 11.5, color: 'var(--text-2)' }}>
                {item.publisher}
                {item.reference_date || item.published_at ? ` · ${item.reference_date || item.published_at}` : ''}
              </span>
            </span>
          </a>
        ))}
      </div>
    </div>
  )
}

// phase === 'awaiting_problem_focus_selection'일 때만 쓰인다. 문제 영역(problem_areas)은
// 아직 최종 아이디어가 아니므로 그 취지를 안내 문구로 항상 함께 보여준다(요청 사항 그대로).
function ProblemAreaSelectionBlock({ problemAreas, externalEvidence, onSend, disabled, pendingActionCode }) {
  const [selectedIds, setSelectedIds] = useState([])

  function toggle(areaId) {
    setSelectedIds((prev) => {
      if (prev.includes(areaId)) return prev.filter((id) => id !== areaId)
      if (prev.length >= 2) return [prev[1], areaId] // 최대 2개까지만 유지(결합용).
      return [...prev, areaId]
    })
  }

  const selectedAreas = problemAreas.filter((a) => selectedIds.includes(a.area_id))
  const selectedTitles = selectedAreas.map((a) => a.title).join(' · ')

  function handleSelect() {
    if (selectedIds.length !== 1) return
    onSend(`"${selectedTitles}"를 문제 영역으로 선택합니다.`, {
      actionCode: 'select_problem_focus',
      actionPayload: { area_ids: selectedIds },
    })
  }

  function handleCombine() {
    if (selectedIds.length !== 2) return
    onSend(`"${selectedTitles}" 두 문제 영역을 결합해서 진행합니다.`, {
      actionCode: 'combine_problem_focus',
      actionPayload: { area_ids: selectedIds },
    })
  }

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ fontSize: 13.5, color: 'var(--text-2)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        해결할 문제 선택
      </div>
      <div className="rb-ideation-notice">
        공고문과 최근 이슈를 바탕으로, 이번 공모전에서 해결할 수 있는 문제 후보를 찾았습니다.
        각 후보를 살펴보고 '누가 어떤 상황에서 어떤 불편을 겪는지'를 기준으로 논의할 문제를
        선택해 주세요. 해결 방법과 아이디어는 다음 단계에서 구체화합니다.
      </div>
      <ExternalEvidenceStrip items={externalEvidence} />
      <div style={{ marginTop: 10 }}>
        {problemAreas.map((area) => (
          <SelectableAreaCard
            key={area.area_id}
            area={area}
            selected={selectedIds.includes(area.area_id)}
            onToggle={() => toggle(area.area_id)}
            disabled={disabled}
          />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button
          type="button"
          className="btn-primary"
          disabled={disabled || selectedIds.length !== 1}
          onClick={handleSelect}
        >
          {pendingActionCode === 'select_problem_focus' ? '선택 중...' : '선택한 문제 영역으로 진행'}
        </button>
        <button
          type="button"
          className="btn-ghost"
          disabled={disabled || selectedIds.length !== 2}
          onClick={handleCombine}
        >
          {pendingActionCode === 'combine_problem_focus' ? '결합 중...' : '선택해서 결합'}
        </button>
      </div>
    </div>
  )
}

function SolutionDirectionCard({ direction, selected, onToggle, disabled }) {
  const isActive = direction.status === 'active'
  return (
    <div
      role={isActive ? 'button' : undefined}
      tabIndex={isActive && !disabled ? 0 : -1}
      aria-pressed={isActive ? selected : undefined}
      onClick={() => isActive && !disabled && onToggle()}
      onKeyDown={(e) => {
        if (!isActive || disabled || (e.key !== 'Enter' && e.key !== ' ')) return
        e.preventDefault()
        onToggle()
      }}
      className={`card glass${isActive ? ' rb-ideation-candidate-card rb-ideation-candidate-card--interactive' : ''}`}
      style={{
        marginBottom: 10,
        padding: 12,
        cursor: isActive && !disabled ? 'pointer' : 'default',
        border: selected ? '2px solid var(--purple)' : '1px solid var(--glass-border)',
        background: selected ? '#f5f1ff' : isActive ? 'var(--bg-1)' : '#faf9fc',
        opacity: isActive ? 1 : 0.6,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
        <div
          style={{
            fontSize: 15.5, fontWeight: 700, color: 'var(--text-0)',
            textDecoration: isActive ? 'none' : 'line-through',
          }}
        >
          {direction.title}
        </div>
        {isActive ? (
          selected
            ? <CheckCircle2 size={17} color="var(--purple)" style={{ flexShrink: 0 }} />
            : <Circle size={15} color="var(--glass-border)" style={{ flexShrink: 0 }} />
        ) : (
          <span className="badge grey mono" style={{ fontSize: 11 }}>
            {direction.status === 'dropped'
              ? '폐기됨'
              : direction.status === 'merged'
                ? '결합됨'
                : ['superseded', 'revised'].includes(direction.status)
                  ? '수정됨'
                  : direction.status}
          </span>
        )}
      </div>
      {direction.core_principle && (
        <div style={{ fontSize: 13.5, color: 'var(--text-1)', lineHeight: 1.55, marginTop: 3 }}>
          {direction.core_principle}
        </div>
      )}
    </div>
  )
}

// phase === 'awaiting_conflict_resolution'일 때만 쓰인다.
function ConflictResolutionBlock({ solutionDirections, onSend, disabled, pendingActionCode }) {
  const [selectedIds, setSelectedIds] = useState([])
  const activeDirections = solutionDirections.filter((d) => d.status === 'active')
  const inactiveDirections = solutionDirections.filter((d) => d.status !== 'active')
  const activeDirectionTitles = activeDirections.map((d) => d.title).filter(Boolean)
  const activeDirectionSummary = activeDirectionTitles.slice(0, 3).map((title) => `‘${title}’`).join(', ')
    + (activeDirectionTitles.length > 3 ? ` 외 ${activeDirectionTitles.length - 3}개` : '')
  const activeDirectionContext = activeDirectionSummary
    ? `현재 활성 해결 방향(${activeDirectionSummary})`
    : '현재 해결 방향'

  function toggle(directionId) {
    setSelectedIds((prev) => (prev.includes(directionId) ? prev.filter((id) => id !== directionId) : [...prev, directionId]))
  }

  const selectedTitles = solutionDirections
    .filter((d) => selectedIds.includes(d.direction_id))
    .map((d) => d.title)
    .join(' · ')

  function handleMerge() {
    if (selectedIds.length < 2) return
    onSend(`선택한 방향을 결합합니다: ${selectedTitles}`, {
      actionCode: 'merge_directions',
      actionPayload: { direction_ids: selectedIds },
    })
    setSelectedIds([])
  }

  function handleDrop() {
    if (selectedIds.length < 1) return
    onSend(`선택한 방향을 폐기합니다: ${selectedTitles}`, {
      actionCode: 'drop_direction',
      actionPayload: { direction_ids: selectedIds },
    })
    setSelectedIds([])
  }

  function handleAddDirection() {
    onSend(
      `${activeDirectionContext}을 보완할 새로운 해결 방향을 추가로 제안해 주세요.`,
      { actionCode: 'add_solution_direction', actionPayload: {} },
    )
  }

  function handleProceed() {
    onSend(
      `${activeDirectionContext}으로 검증을 진행합니다.`,
      { actionCode: 'proceed_to_validation', actionPayload: {} },
    )
  }

  function handleReturnToProblem() {
    onSend('문제 정의로 돌아갑니다.', { actionCode: 'return_to_problem_definition', actionPayload: {} })
  }

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ fontSize: 13.5, color: 'var(--text-2)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        해결 방향 조정
      </div>
      {activeDirections.map((d) => (
        <SolutionDirectionCard
          key={d.direction_id}
          direction={d}
          selected={selectedIds.includes(d.direction_id)}
          onToggle={() => toggle(d.direction_id)}
          disabled={disabled}
        />
      ))}
      {inactiveDirections.map((d) => (
        <SolutionDirectionCard key={d.direction_id} direction={d} selected={false} onToggle={() => {}} disabled />
      ))}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
        <button type="button" className="btn-ghost" disabled={disabled || selectedIds.length < 2} onClick={handleMerge}>
          {pendingActionCode === 'merge_directions' ? '결합 중...' : '결합'}
        </button>
        <button type="button" className="btn-ghost" disabled={disabled || selectedIds.length < 1} onClick={handleDrop}>
          {pendingActionCode === 'drop_direction' ? '폐기 중...' : '폐기'}
        </button>
        <button type="button" className="btn-ghost" disabled={disabled} onClick={handleAddDirection}>
          {pendingActionCode === 'add_solution_direction' ? '요청 중...' : '새 방향 추가 요청'}
        </button>
        <button type="button" className="btn-primary" disabled={disabled} onClick={handleProceed}>
          {pendingActionCode === 'proceed_to_validation' ? '검증 요청 중...' : '이대로 검증 진행'}
        </button>
        <button type="button" className="btn-ghost" disabled={disabled} onClick={handleReturnToProblem}>
          {pendingActionCode === 'return_to_problem_definition' ? '이동 중...' : '문제 정의로 돌아가기'}
        </button>
      </div>
    </div>
  )
}

// phase === 'awaiting_concept_confirmation'일 때만 쓰인다. "이 방향으로 최종 확정" 버튼은
// 스펙 요청대로 이 phase에서만 렌더링되며(다른 어떤 phase에서도 노출하지 않음), 다른
// 후보 확정 흐름(CandidateCard/finalize)과는 별개의 confirm_concept 액션이다.
const VALIDATION_STATUS_LABEL = {
  passed: '통과',
  passed_with_caution: '주의 필요',
  needs_revision: '수정 필요',
}

function ValidationReviewCard({ title, result, technical = false }) {
  if (!result) {
    return (
      <div className="card glass" style={{ padding: 14 }}>
        <strong>{title}</strong>
        <div style={{ marginTop: 7, color: 'var(--text-2)' }}>{title}을 진행하고 있습니다.</div>
      </div>
    )
  }
  return (
    <div className="card glass" style={{ padding: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        <strong>{title}</strong>
        <span className={`badge ${result.status === 'needs_revision' ? 'amber' : 'green'} mono`}>
          {VALIDATION_STATUS_LABEL[result.status] || result.status}
        </span>
      </div>
      {result.passed_items?.length > 0 && (
        <div style={{ marginTop: 9 }}>
          <strong style={{ fontSize: 13, color: 'var(--text-2)' }}>통과 항목</strong>
          <ul style={{ margin: '3px 0 0', paddingLeft: 17, lineHeight: 1.6 }}>
            {result.passed_items.map((item, index) => <li key={index}>{item}</li>)}
          </ul>
        </div>
      )}
      {result.issues?.length > 0 && (
        <div style={{ marginTop: 9 }}>
          <strong style={{ fontSize: 13, color: 'var(--text-2)' }}>
            {technical ? '기술 위험' : '수정 필요 항목'}
          </strong>
          <ul style={{ margin: '3px 0 0', paddingLeft: 17, lineHeight: 1.6 }}>
            {result.issues.map((issue, index) => (
              <li key={issue.code || index}>{issue.description}</li>
            ))}
          </ul>
        </div>
      )}
      {result.revision_suggestions?.length > 0 && (
        <div style={{ marginTop: 9 }}>
          <strong style={{ fontSize: 13, color: 'var(--text-2)' }}>수정 제안</strong>
          <ul style={{ margin: '3px 0 0', paddingLeft: 17, lineHeight: 1.6 }}>
            {result.revision_suggestions.map((item, index) => <li key={index}>{item}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

function ValidationPendingBlock() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 10, marginTop: 10 }}>
      <ValidationReviewCard title="기획위원 검증" />
      <ValidationReviewCard title="개발위원 검증" technical />
    </div>
  )
}

function ConceptConfirmationBlock({
  provisionalIdea,
  validationResult,
  ideaEvolution,
  onSend,
  disabled,
  pendingActionCode,
}) {
  function handleConfirm() {
    onSend('이 방향으로 최종 확정합니다.', { actionCode: 'confirm_concept', actionPayload: {} })
  }
  function handleRevise() {
    const target = provisionalIdea?.title ? `‘${provisionalIdea.title}’ 후보를` : '현재 후보를'
    onSend(`${target} 다시 검토하고 수정해 주세요.`, { actionCode: 'revise_candidate', actionPayload: {} })
  }
  function handleReturnToProblem() {
    onSend('문제 정의로 돌아갑니다.', { actionCode: 'return_to_problem_definition', actionPayload: {} })
  }
  function handleChooseAnother() {
    onSend('다른 후보를 선택하겠습니다.', { actionCode: 'choose_another_candidate', actionPayload: {} })
  }
  const revisions = (ideaEvolution || [])
    .filter((item) => item?.action_type === 'revision' || item?.stage === 'idea_validation')
    .slice(-4)
  const cautions = [
    ...(validationResult?.planning?.status === 'passed_with_caution'
      ? validationResult.planning.issues || []
      : []),
    ...(validationResult?.technical?.status === 'passed_with_caution'
      ? validationResult.technical.issues || []
      : []),
  ]
  const aiNecessity = provisionalIdea?.ai_necessity
    || provisionalIdea?.ai_role
    || validationResult?.technical?.passed_items?.find((item) => /AI|인공지능/.test(item))
    || validationResult?.technical?.message

  return (
    <div style={{ marginTop: 4 }}>
      <div className="rb-ideation-notice">
        선택한 후보는 아직 최종 확정되지 않았으며, 기획·개발 관점의 검증 후 변경될 수 있습니다.
      </div>
      {provisionalIdea && (
        <div className="card glass" style={{ marginTop: 10, marginBottom: 10, padding: 14 }}>
          <div style={{ fontSize: 17, fontWeight: 700, color: 'var(--text-0)', marginBottom: 7 }}>
            {provisionalIdea.title}
          </div>
          {provisionalIdea.problem && (
            <div style={{ fontSize: 14.5, color: 'var(--text-0)', lineHeight: 1.6, marginBottom: 4 }}>
              <strong style={{ color: '#514a61', fontWeight: 700 }}>해결할 문제 · </strong>{provisionalIdea.problem}
            </div>
          )}
          {provisionalIdea.target_user && (
            <div style={{ fontSize: 14.5, color: 'var(--text-0)', lineHeight: 1.6, marginBottom: 4 }}>
              <strong style={{ color: '#514a61', fontWeight: 700 }}>목표 사용자 · </strong>{provisionalIdea.target_user}
            </div>
          )}
          {provisionalIdea.solution && (
            <div style={{ fontSize: 14.5, color: 'var(--text-0)', lineHeight: 1.6, marginBottom: 4 }}>
              <strong style={{ color: '#514a61', fontWeight: 700 }}>해결 방안 · </strong>{provisionalIdea.solution}
            </div>
          )}
          {provisionalIdea.differentiation && (
            <div style={{ fontSize: 14.5, color: 'var(--text-0)', lineHeight: 1.6 }}>
              <strong style={{ color: '#514a61', fontWeight: 700 }}>차별점 · </strong>{provisionalIdea.differentiation}
            </div>
          )}
          {provisionalIdea.core_value && (
            <div style={{ fontSize: 14.5, color: 'var(--text-0)', lineHeight: 1.6, marginBottom: 4 }}>
              <strong style={{ color: '#514a61', fontWeight: 700 }}>핵심 가치 · </strong>{provisionalIdea.core_value}
            </div>
          )}
          {Array.isArray(provisionalIdea.required_data) && provisionalIdea.required_data.length > 0 && (
            <div style={{ fontSize: 14.5, color: 'var(--text-0)', lineHeight: 1.6, marginBottom: 4 }}>
              <strong style={{ color: '#514a61', fontWeight: 700 }}>활용 데이터 · </strong>
              {provisionalIdea.required_data.join(', ')}
            </div>
          )}
          {aiNecessity && (
            <div style={{ fontSize: 14.5, color: 'var(--text-0)', lineHeight: 1.6 }}>
              <strong style={{ color: '#514a61', fontWeight: 700 }}>AI가 필요한 이유 · </strong>
              {aiNecessity}
            </div>
          )}
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 10, marginBottom: 10 }}>
        <ValidationReviewCard title="기획위원 검증" result={validationResult?.planning} />
        <ValidationReviewCard title="개발위원 검증" result={validationResult?.technical} technical />
      </div>
      {revisions.length > 0 && (
        <div className="card glass" style={{ padding: 14, marginBottom: 10 }}>
          <strong>검증 과정에서 수정된 내용</strong>
          <ul style={{ margin: '5px 0 0', paddingLeft: 17, lineHeight: 1.6 }}>
            {revisions.map((item) => <li key={item.record_id}>{item.content || item.title}</li>)}
          </ul>
        </div>
      )}
      {cautions.length > 0 && (
        <div className="rb-ideation-notice" style={{ marginBottom: 10 }}>
          남아 있는 주의사항 · {cautions.map((item) => item.description).filter(Boolean).join(' · ')}
        </div>
      )}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button type="button" className="btn-primary" disabled={disabled} onClick={handleConfirm}>
          {pendingActionCode === 'confirm_concept' ? '확정 중...' : '이 아이디어로 최종 확정'}
        </button>
        <button type="button" className="btn-ghost" disabled={disabled} onClick={handleRevise}>
          {pendingActionCode === 'revise_candidate' ? '요청 중...' : '검증 결과를 반영해 다시 수정'}
        </button>
        <button type="button" className="btn-ghost" disabled={disabled} onClick={handleChooseAnother}>
          {pendingActionCode === 'choose_another_candidate' ? '이동 중...' : '다른 후보 선택'}
        </button>
        <button type="button" className="btn-ghost" disabled={disabled} onClick={handleReturnToProblem}>
          {pendingActionCode === 'return_to_problem_definition' ? '이동 중...' : '문제 정의로 돌아가기'}
        </button>
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
  onGoFormDraft,
  onBack,
  saving = false,
  saveError = '',
}) {
  const [starting, setStarting] = useState(!ideationConv)
  const [sending, setSending] = useState(false)
  // 용준/Claude(2026-07-27, 요청: "처리 중인 액션에 상태 표시") — 구조화 액션 버튼 클릭
  // 시점의 action_code를 기억해뒀다가, 그 요청이 처리되는 동안(sending=true) 같은 버튼에
  // "선택 중"/"결합 중" 같은 라벨을 보여준다. sending이 false로 돌아가면(성공/실패/취소
  // 어느 경로든) 아래 useEffect가 자동으로 비운다 — setSending(false) 호출부가 여러 곳
  // (스트리밍/비스트리밍/폴백)이라 매 지점마다 따로 리셋하지 않아도 항상 정확하다.
  const [pendingActionCode, setPendingActionCode] = useState(null)
  const [finalizing, setFinalizing] = useState(false)
  const [showFinalizedModal, setShowFinalizedModal] = useState(false)
  const [stayInMeetingAfterFinalize, setStayInMeetingAfterFinalize] = useState(false)
  const [previewProposal, setPreviewProposal] = useState(null)
  const [generatingModalDraft, setGeneratingModalDraft] = useState(false)
  const [modalDraftError, setModalDraftError] = useState(null)
  // 용준/Claude(2026-07-28, 요청: "다시 시도"가 전체 회의를 처음부터 다시 실행하지 않게)
  // — phase="failed"일 때 ErrorBanner의 "다시 시도" 버튼이 이 상태를 쓴다(handleRestart와
  // 별개 — 전체 재시작이 아니라 failed_node만 재실행).
  const [retryingFailedNode, setRetryingFailedNode] = useState(false)
  const [error, setError] = useState(null)
  const [draft, setDraft] = useState('')
  const [chatHeight, setChatHeight] = useState(null)
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
  // pge/Claude(2026-07-28, 요청: "진행 상황 패널은 아예 안 불러오게 — 그 탭엔 진행
  // 상황이 아니라 아이디어 변화 과정이 들어가야 함") — IdeationProgressPanel은 이 화면
  // import·렌더링에서 완전히 뺐다. IdeaCanvasPanel과 IdeaEvolutionTimeline을 탭으로
  // 전환하는 하나의 자리로 합친다. 기본 탭은 "canvas"다.
  const [canvasColumnTab, setCanvasColumnTab] = useState('evolution')
  // 용준/Claude(2026-07-28, 요청: 검증 단계 타이핑 재현) — 검증(idea_validation) 발언은
  // 아바타가 꺼져 있을 때(IDEATION_AVATAR_ENABLED=false, 개발 모드) 통째로 한 번에
  // 도착하므로, "이미 화면에 다 드러난 발언 id"를 여기 담아둔다. 세션을 처음 불러올
  // 때(마운트 시점) 이미 있던 검증 발언은 lazy 초기값으로 전부 넣어 애니메이션을
  // 건너뛰고(재방문 시 다시 타이핑되면 어색하다), 마운트 이후 새로 도착한 발언만
  // TypedMessageBubble로 타이핑된다.
  const [revealedValidationIds, setRevealedValidationIds] = useState(() => {
    const ids = new Set()
    for (const m of ideationConv?.messages || []) {
      if (isValidationOpinionMessage(m)) ids.add(m.message_id)
    }
    return ids
  })
  const markValidationRevealed = (messageId) => {
    setRevealedValidationIds((prev) => {
      if (prev.has(messageId)) return prev
      const next = new Set(prev)
      next.add(messageId)
      return next
    })
  }
  // 재인/Claude(2026-07-23, 2026-07-24 갱신): 아래 eager fetch effect가 지금 진행 중인
  // continue-turn fetch를 추적한다 — "잠시만"이 그 사이에 눌리면 abort()로 끊어서, 이미
  // 중단한 뒤에 뒤늦게 도착하는 응답이 canonical state를 다시 덮어쓰지 않게 막는다
  // (handleInterject 참고). 그 effect 자신이 sending/interrupting이 걸려 있으면 애초에
  // 새로 시작하지 않으므로, 세션 락 409 자체는 이 ref 없이도 이미 피한다 — 이 ref는 그
  // 이후(이미 시작된 호출)에 대한 정리용이다.
  const avatarTurnAbortRef = useRef(null)

  const startedRef = useRef(false)
  const chatScrollRef = useRef(null)
  const chatResizeDragRef = useRef(null)
  // 사용자가 채팅 맨 아래를 보고 있을 때만 새 메시지를 따라간다. 과거 메시지를 읽으려고
  // 위로 스크롤하면 자동 이동을 멈추고, 다시 아래로 내리면 자동 추적을 재개한다.
  const shouldFollowChatRef = useRef(true)
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
  // 회의 대화 메시지가 처음 채워질 때(마운트 직후 fetch 완료 시점)는 auto-follow
  // effect가 곧바로 맨 아래로 스크롤해버려 마운트 시 scrollTop = 0 초기화를 덮어썼다.
  // 이 ref로 "메시지가 처음 채워진 순간"을 구분해 그때만 맨 위 고정을 유지한다.
  const initialMessagesLoadedRef = useRef(false)

  useEffect(() => {
    return () => {
      // 요청: "컴포넌트 unmount 시 요청 취소".
      streamAbortRef.current?.abort()
      pendingFinalRef.current = null
    }
  }, [])

  // 버그 수정: 최초 렌더 시 스크롤 컨테이너가 맨 아래에서 시작하던 문제 —
  // 마운트 시점에 명시적으로 맨 위로 초기화한다.
  useEffect(() => {
    const container = chatScrollRef.current
    if (container) container.scrollTop = 0
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
          setStarting(false)
        })
        .catch((err) => {
          setError(classifyIdeationConvError(err))
          setStarting(false)
        })
      return
    }
    setStarting(true)
    if (projectId) {
      getAnnouncementAnalysis(projectId).then(setAnnouncementAnalysis).catch(() => {})
    }
    getIdeationConversation(savedSessionId)
      .then((data) => {
        setIdeationConv(data)
        setStarting(false)
      })
      .catch((err) => {
        if (err?.status !== 404) {
          setError(classifyIdeationConvError(err))
          setStarting(false)
          return
        }
        if (key) sessionStorage.removeItem(key)
        return runStart()
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const container = chatScrollRef.current
    if (!container) return

    const messageCount = ideationConv?.messages?.length || 0
    if (messageCount > 0 && !initialMessagesLoadedRef.current) {
      // 메시지가 처음 채워지는 시점 — 맨 아래로 따라가지 않고 맨 위에 고정한 채로 시작한다.
      initialMessagesLoadedRef.current = true
      container.scrollTop = 0
      return
    }

    if (!shouldFollowChatRef.current) return

    // scrollIntoView()는 채팅 박스의 모든 스크롤 가능한 조상(바깥 페이지 포함)을 함께
    // 움직인다. 컨테이너의 scrollTop만 바꿔 새 위원 발언이 와도 페이지 위치는 고정한다.
    container.scrollTop = container.scrollHeight
    // 화면에 실제로 드러난 글자 수(displayedContent)가 늘어날 때(타이핑 진행)마다도
    // 스크롤해야 하므로, 배열 참조 자체가 아니라 지금까지 누적된 총 글자 수를 의존값으로
    // 쓴다(요청: "delta가 들어올 때 자동 스크롤" — content가 아니라 실제 화면 표시 기준).
  }, [
    ideationConv?.messages?.length,
    optimisticUserMessage?.message_id,
    streamState.messages.reduce((n, m) => n + (m.displayedContent?.length || 0), 0),
  ])

  // 용준/Claude(2026-07-28) — TypedMessageBubble(검증 발언 타이핑 재현)은 streamState
  // 바깥에서 독립적으로 글자를 늘리므로 위 effect의 의존값에 안 잡힌다. 매 프레임 직접
  // 호출해 같은 규칙(사용자가 바닥 근처에 있을 때만 따라 내려감)으로 스크롤을 맞춘다.
  function followChatScrollIfNeeded() {
    const container = chatScrollRef.current
    if (!container || !shouldFollowChatRef.current) return
    container.scrollTop = container.scrollHeight
  }

  function handleChatScroll(event) {
    const container = event.currentTarget
    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight
    shouldFollowChatRef.current = distanceFromBottom <= 72
  }

  function handleChatResizePointerDown(event) {
    const container = chatScrollRef.current
    if (!container) return
    event.preventDefault()
    event.currentTarget.setPointerCapture?.(event.pointerId)
    chatResizeDragRef.current = {
      startY: event.clientY,
      startHeight: container.getBoundingClientRect().height,
    }
  }

  function handleChatResizePointerMove(event) {
    const drag = chatResizeDragRef.current
    if (!drag) return
    const maxHeight = window.innerHeight * 0.78
    setChatHeight(Math.max(220, Math.min(maxHeight, drag.startHeight + event.clientY - drag.startY)))
  }

  function handleChatResizePointerUp(event) {
    chatResizeDragRef.current = null
    event.currentTarget.releasePointerCapture?.(event.pointerId)
  }

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
  // sending/interrupting 중 하나라도 걸려 있으면 부르지 않는다 — 사용자가
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
  // "진짜로" 바뀌거나 sending/interrupting이 바뀔 때만 재평가한다.
  useEffect(() => {
    if (!ideationConv?.session_id) return
    // 용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") —
    // idea_validation도 expert_discussion과 같은 "위원 발언 1건씩 이어서 요청" 패턴을 쓴다
    // (backend/app/api/routes/ideation_conversation_preview.py::continue_expert_turn_stream이
    // phase로 continue_ideation_expert_turn/continue_ideation_validation_turn을 나눠
    // 부른다). 그래서 이 루프는 phase 이름만 넓히면 그대로 재사용된다.
    if (!EAGER_FETCH_TURN_PHASES.includes(ideationConv.phase)) return
    if (sending || interrupting) return
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
          if (cancelled || !nextState || !EAGER_FETCH_TURN_PHASES.includes(nextState.phase)) break
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
  }, [ideationConv?.session_id, ideationConv?.phase, sending, interrupting])

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
  const hasVisibleConversation = [
    ...canonicalMessages,
    ...(optimisticUserMessage ? [optimisticUserMessage] : []),
    ...streamState.messages,
  ].some((message) => (message?.displayedContent ?? message?.content ?? '').trim())
  // 첫 API 응답을 기다리는 동안뿐 아니라, canonical 메시지는 도착했지만 아바타 재생
  // 순서 때문에 아직 첫 발언이 공개되지 않은 순간에도 빈 흰 박스를 보여주지 않는다.
  const showMeetingPreparing = !hasVisibleConversation
    && !phaseFailure
    && !error
    && (
      starting
      || sending
      || !ideationConv
      || rawMessages.length > canonicalMessages.length
      || (!!ideationConv && phase !== 'finalized' && phase !== 'failed')
    )
  const meetingPreparingDetail = startPhaseLabel
    || streamState.phaseLabel
    || '공모전 자료와 최신 근거를 검토하고 있어요'
  const latestVisibleMessageId = [
    ...canonicalMessages,
    ...(optimisticUserMessage ? [optimisticUserMessage] : []),
    ...streamState.messages,
  ]
    .filter((message) => (message?.displayedContent ?? message?.content ?? '').trim())
    .at(-1)?.message_id
  // sending이 false로 돌아가는 모든 경로(성공/실패/취소/폴백)를 한곳에서 감지해
  // pendingActionCode를 비운다 — setSending(false) 호출부마다 따로 리셋할 필요가 없다.
  useEffect(() => {
    if (!sending) setPendingActionCode(null)
  }, [sending])
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
  // 2026-07-26 라운드테이블 재설계 — 진행자가 needs_user_decision=true와 함께 반환하는
  // structured.choices(백엔드는 이미 보내고 있었지만 프론트가 안 쓰고 있었다)를 후보 카드와
  // 같은 자리·같은 클릭→자유텍스트 패턴으로 노출한다. phase가 awaiting_user_decision일
  // 때만 의미가 있고(그 외엔 이미 답변된 지난 선택지라 다시 눌러도 무효), canonicalMessages의
  // 마지막 진행자 메시지만 본다 — 그보다 이전 메시지의 choices는 이미 지난 사이클이다.
  const latestFacilitatorMessage = [...canonicalMessages]
    .reverse()
    .find((m) => m.speaker_id === 'ideation_facilitator')
  const latestFacilitatorChoices =
    phase === 'awaiting_user_decision' && latestFacilitatorMessage?.structured?.needs_user_decision
      ? latestFacilitatorMessage.structured.choices || []
      : []
  // 용준/Claude(2026-07-22, 요청: "잠시만" 버튼) — 실제로 기획/개발 위원이 발언을
  // 스트리밍하는 동안에만(말풍선이 하나 이상 생겨야) 활성화한다. 이미 취소 확인을 기다리는
  // 중이면(interrupting) 다시 누를 수 없다.
  const canInterject = sending && !interrupting && streamState.messages.length > 0

  // 용준/Claude(2026-07-27, 요청: discovery 모드 구조화 액션 버튼) — actionCode/actionPayload는
  // 순수 추가 파라미터다. 생략하면(기존 모든 호출부) undefined로 넘어가 지금까지와 동일하게
  // 자유 텍스트 message만으로 reply가 해석된다.
  async function sendNonStreaming(text, actionCode, actionPayload) {
    try {
      const data = await replyIdeationConversation(ideationConv.session_id, text, undefined, actionCode, actionPayload)
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
  // 용준/Claude(2026-07-27, 요청: discovery 모드 구조화 액션 버튼) — 세 번째 인자는 순수
  // 추가 파라미터(action_code/action_payload)다. 기존 호출부(candidate 선택, 일반 채팅
  // 전송, 진행자 선택지 버튼, interject 등)는 이 인자를 전혀 넘기지 않으므로
  // actionCode/actionPayload가 항상 undefined로 남아 지금까지와 동작이 완전히 동일하다.
  async function handleSend(overrideText, { actionCode, actionPayload } = {}) {
    const text = (overrideText ?? draft).trim()
    if (!text || !ideationConv) return
    if (!canReplyOrContinue) return
    setSending(true)
    if (actionCode) setPendingActionCode(actionCode)
    setError(null)
    // 보내는 즉시 화면에 반영 — 서버 왕복(위원 응답 생성)이 끝나기를 기다리지 않는다.
    setOptimisticUserMessage({ message_id: 'LOCAL-OPTIMISTIC-USER', speaker_id: 'user', message_type: 'answer', content: text })
    setDraft('')

    if (!streamingSupportedRef.current) {
      await sendNonStreaming(text, actionCode, actionPayload)
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
        // 재인/Claude(2026-07-23, 2026-07-24 갱신): 이 화면은 항상 아바타를 재생하므로,
        // 라운드를 새로 여는 reply라도 위원 발언이 한 번에 다 몰려오지 않고 딱 1건만
        // 오게 매번 true로 보낸다(위쪽 eager fetch effect가 나머지 턴을 이어서 요청함 —
        // 텍스트는 이제 아바타 재생과 무관하게 미리 다 뽑히지만, 채팅/아바타 노출은
        // avatarRevealedCount로 여전히 하나씩 순서대로 공개된다).
        singleTurn: true,
        actionCode,
        actionPayload,
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
        await sendNonStreaming(text, actionCode, actionPayload)
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
  }

  // 용준/Claude(2026-07-22, 요청: "잠시만" 버튼) — 위원이 실제로 발언을 스트리밍하는 동안만
  // 호출된다. 순서: ① 화면 타이핑/실제 스트리밍 중단 요청 → ② "회의를 잠시 멈추고
  // 있어요..." 표시 → ③ 백엔드 취소 확인(세션 lock이 실제로 풀릴 때까지 대기) → 중단된
  // 발언까지만 canonical에 반영하고 라운드는 다음 정지 지점(사용자 결정 대기 등)까지
  // 그대로 이어간다.
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
    }
  }

  async function handleFinalize() {
    if (!canFinalize) return
    setError(null)
    const currentIdea = ideationConv.idea_canvas || ideationConv.selected_idea || ideationConv.provisional_idea || {}
    setPreviewProposal({
      idea_name: currentIdea.title || currentIdea.idea_name || ideationConv.selected_idea?.title || '확정 예정 아이디어',
      problem_definition: currentIdea.problem || currentIdea.problem_definition,
      target_user: currentIdea.target_user,
      core_user_value: currentIdea.core_value || currentIdea.core_user_value,
      key_features: currentIdea.key_features || currentIdea.features,
      required_data: currentIdea.required_data,
      tech_direction: currentIdea.tech_direction || currentIdea.technical_approach,
      mvp_scope: currentIdea.mvp_scope,
      differentiation: currentIdea.differentiation,
      risks_and_mitigations: currentIdea.risks_and_mitigations || currentIdea.risks,
      success_metrics: currentIdea.success_metrics,
      expert_final_opinions: ideationConv.consensus,
      unverified_assumptions: ideationConv.unresolved_issues,
      final_recommendation: currentIdea.contest_fit || currentIdea.final_recommendation,
    })
    setShowFinalizedModal(true)
  }

  async function handleModalGoToFormDraft() {
    if (!ideationConv?.session_id || generatingModalDraft) return
    setGeneratingModalDraft(true)
    setModalDraftError(null)
    try {
      setFinalizing(true)
      const finalizedData = await finalizeIdeationConversation(ideationConv.session_id)
      setIdeationConv(finalizedData)
      const draftData = await generateApplicationFormDraft(finalizedData.session_id)
      setIdeationConv(draftData)
      await onGoFormDraft?.(draftData)
    } catch (err) {
      setModalDraftError(classifyIdeationConvError(err))
    } finally {
      setFinalizing(false)
      setGeneratingModalDraft(false)
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

  // 용준/Claude(2026-07-28, 요청: "다시 시도"가 전체 회의를 처음부터 다시 실행하지 않게)
  // — session_id가 있는 phase="failed" 세션은 failed_node만 재실행한다(messages/
  // problem_definition/idea_evolution 등 기존 회의 내용을 그대로 유지). session_id가
  // 없으면(세션 시작 자체가 실패한 경우) 재실행할 노드가 없으므로 기존 전체 재시작으로
  // 폴백한다.
  async function handleRetryFailedNode() {
    if (!ideationConv?.session_id) {
      handleRestart()
      return
    }
    setRetryingFailedNode(true)
    setError(null)
    try {
      const data = await retryFailedIdeationConversationNode(ideationConv.session_id)
      setIdeationConv(data)
    } catch (err) {
      setError(classifyIdeationConvError(err))
    } finally {
      setRetryingFailedNode(false)
    }
  }

  // 이미 확정까지 끝난 세션으로 이 화면에 돌아온 경우(사이드바 재진입) — 다시 채팅하지
  // 않고 바로 결과로 넘어갈 수 있게만 안내한다.
  // 신청서 초안을 생성하는 동안 finalize 응답이 먼저 도착해도 기존 주제 확정
  // 결과 화면으로 분기하지 않는다. 모달의 로딩 상태를 유지한 뒤 초안 생성이
  // 완료되면 부모가 곧바로 form_draft 단계로 이동한다.
  if (
    ideationConv?.phase === 'finalized'
    && !stayInMeetingAfterFinalize
    && !generatingModalDraft
  ) {
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
  const nextStepLabel = nextStepLabelFor(ideationConv)
  const metaItems = ideationConv
    ? [
        // 용준/Claude(2026-07-26, 요청: "/3은 필요없고 실시간 라운드 숫자만, 4라운드
        // 넘어가면 4로") — max_rounds는 백엔드가 무한 루프를 막는 안전 상한일 뿐
        // "정확히 이 라운드까지"라는 목표치가 아니므로 더 이상 분모로 보여주지 않는다.
        // ideationConv.round 값 자체를 그대로 보여주면 라운드가 늘어날 때마다 자동으로
        // 최신 숫자가 표시된다.
        { icon: ListChecks, label: '진행 라운드', value: `${ideationConv.round ?? 0}` },
        { icon: Users, label: '참여 위원', value: '3명' },
        { icon: Lightbulb, label: '논의 아이디어', value: `${discussionIdeaCountFor(ideationConv)}개` },
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
      {showFinalizedModal && previewProposal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="finalized-idea-title"
          style={{
            position: 'fixed', inset: 0, zIndex: 100,
            display: 'grid', placeItems: 'center', padding: 24,
            background: 'rgba(28,26,46,.42)', backdropFilter: 'blur(5px)',
          }}
        >
          <div
            className="card glass"
            aria-busy={generatingModalDraft}
            style={{
              position: 'relative', width: 'min(820px, 94vw)', maxHeight: '88vh',
              overflowY: 'auto', padding: '30px 32px 26px', background: '#fff',
              boxShadow: '0 24px 70px rgba(28,26,46,.24)',
            }}
          >
            <button
              type="button"
              className="btn-ghost"
              aria-label="주제 확정 결과 닫기"
              onClick={() => setShowFinalizedModal(false)}
              disabled={generatingModalDraft}
              style={{
                position: 'absolute', top: 16, right: 16, width: 38, height: 38,
                display: 'grid', placeItems: 'center', padding: 0, borderRadius: 10,
              }}
            >
              <X size={19} />
            </button>

            <div className="badge green mono" style={{ marginBottom: 12 }}>주제 확정 완료</div>
            <h2
              id="finalized-idea-title"
              style={{ margin: '0 48px 24px 0', fontSize: 28, fontWeight: 800, lineHeight: 1.35 }}
            >
              {previewProposal.idea_name || '확정된 아이디어 주제'}
            </h2>

            <div style={{ borderTop: '1px solid var(--glass-border)' }}>
              {PROPOSAL_ROWS.map(([key, label]) => (
                <div
                  key={key}
                  style={{
                    display: 'grid', gridTemplateColumns: '160px 1fr', gap: 18,
                    padding: '13px 0', borderBottom: '1px solid var(--glass-border)',
                  }}
                >
                  <div style={{ fontSize: 13.5, fontWeight: 700, color: 'var(--text-2)' }}>{label}</div>
                  <div style={{ fontSize: 14.5, lineHeight: 1.65 }}>
                    {proposalValueDisplay(previewProposal[key])}
                  </div>
                </div>
              ))}
            </div>

            {modalDraftError && (
              <p style={{ color: 'var(--coral)', fontSize: 14, textAlign: 'center', margin: '16px 0 0' }}>
                {modalDraftError.message}
              </p>
            )}
            {generatingModalDraft && (
              <div
                role="status"
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 9,
                  marginTop: 18, padding: '12px 14px', borderRadius: 11,
                  color: 'var(--purple)', background: 'var(--purple-dim)',
                  fontSize: 14.5, fontWeight: 700,
                }}
              >
                <LoaderCircle size={18} style={{ animation: 'rb-ideation-spin .9s linear infinite' }} />
                신청서 초안 생성 중입니다. 잠시만 기다려 주세요.
              </div>
            )}
            <div style={{ display: 'flex', justifyContent: 'center', gap: 10, marginTop: 24, flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn-ghost"
                onClick={() => setShowFinalizedModal(false)}
                disabled={generatingModalDraft}
              >
                닫기
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={handleModalGoToFormDraft}
                disabled={generatingModalDraft}
              >
                {generatingModalDraft ? '신청서 초안 작성 중...' : '신청서 초안 작성하러 가기'}
              </button>
            </div>
          </div>
        </div>
      )}
      <div className="rb-ideation-main-contents">
        <section className="rb-idea-header">
        <div className="rb-page-eyebrow-row">
          {onBack && (
            <button type="button" className="rb-back-button" onClick={onBack} disabled={busy} aria-label="이전 화면으로 이동">
              {'←'}
            </button>
          )}
          <div className="rb-page-eyebrow">AI REVIEW BOARD</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 2 }}>
          <h2 style={{ fontSize: 27, fontWeight: 700, color: 'var(--text-0)', letterSpacing: '-0.02em' }}>AI 아이디어 회의</h2>
          <span className="badge amber mono">
            {statusLabelFor({ phase, starting, sending, finalizing, interrupting })}
          </span>
        </div>
        <div style={{ fontSize: 16, fontWeight: 500, color: '#625d72', marginBottom: 14, lineHeight: 1.5 }}>
          공모전 분석 결과를 바탕으로 진행자, 기획 의원, 개발 의원이 함께 아이디어를 논의하고 있습니다.
        </div>
        </section>
        {ideationConv?.competition_name && (
          <div
            className="rb-idea-topic"
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

        {(phaseFailure || error) && <div className="rb-idea-error">
          <ErrorBanner
            error={phaseFailure || error}
            onRetry={phaseFailure ? handleRetryFailedNode : handleRestart}
            retrying={retryingFailedNode}
          />
        </div>}

        {/* 용준/Claude(2026-07-25, 요청: "회의 대화 영역 상단에 가로형 상태 요약 카드") —
            round/idea_candidates.length/phase는 전부 ideationConv에 이미 있는 실제 값이다. */}
        <section className="rb-idea-chat-panel">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-0)' }}>회의 대화</div>
          {ideationConv && phase !== 'finalized' && phase !== 'failed' && (
            <span className="badge green mono" style={{ fontSize: 12 }}>실시간</span>
          )}
          <span style={{ marginLeft: 'auto', fontSize: 12.5, color: 'var(--text-2)' }}>
            ↕ 오른쪽 아래 모서리를 드래그해 높이 조절
          </span>
        </div>
        <div
          ref={chatScrollRef}
          className="card glass rb-idea-chat-scroll"
          onScroll={handleChatScroll}
          aria-busy={showMeetingPreparing}
          style={{
            height: chatHeight ? `${chatHeight}px` : '60vh',
            minHeight: 240,
            maxHeight: '78vh',
            resize: 'vertical',
            overflow: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            padding: 16,
          }}
        >
          {starting && !ideationConv && (
            <p className="rb-ideation-notice">
              {startPhaseLabel ? `${startPhaseLabel}...` : '공모전 분석을 바탕으로 아이디어 후보를 만들고 있어요...'}
            </p>
          )}
          {interruptionMarkers
            .filter((marker) => marker.afterMessageId === null)
            .map((marker) => <InterruptionMarker key={marker.markerId} speakerId={marker.speakerId} />)}
          {/* 검증(idea_validation) 발언은 기획→개발 순으로 하나씩만 타이핑되게, 아래
              validationTurnTaken 플래그로 순서를 막는다(TypedMessageBubble 참고). */}
          {(() => {
            let validationTurnTaken = false
            return canonicalMessages.map((m) => {
              const isLatest = m.message_id === latestVisibleMessageId
              const needsTyping = !IDEATION_AVATAR_ENABLED
                && isValidationOpinionMessage(m)
                && !revealedValidationIds.has(m.message_id)
              let bubble
              if (needsTyping && validationTurnTaken) {
                // 앞선 검증 발언이 아직 타이핑 중 — 이 발언은 그 차례가 올 때까지 대기한다.
                bubble = null
              } else if (needsTyping) {
                validationTurnTaken = true
                bubble = (
                  <TypedMessageBubble
                    message={m}
                    allMessages={visibleMessages}
                    isLatest={isLatest}
                    onDone={() => markValidationRevealed(m.message_id)}
                    onProgress={followChatScrollIfNeeded}
                  />
                )
              } else {
                bubble = <MessageBubble message={m} allMessages={visibleMessages} isLatest={isLatest} />
              }
              return (
                <Fragment key={m.message_id}>
                  {bubble}
                  {interruptionMarkers
                    .filter((marker) => marker.afterMessageId === m.message_id)
                    .map((marker) => <InterruptionMarker key={marker.markerId} speakerId={marker.speakerId} />)}
                </Fragment>
              )
            })
          })()}
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
                발전된 아이디어 후보
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
              <p style={{ margin: '8px 0 0', color: 'var(--text-2)', fontSize: 13.5 }}>
                후보를 선택하면 전문가 검증을 자동으로 진행한 뒤 주제 확정 단계로 이동합니다.
              </p>
            </div>
          )}
          {sending && phase === 'awaiting_candidate_selection' && <ValidationPendingBlock />}
          {/* 2026-07-26 라운드테이블 재설계 — 진행자가 매 사이클 던지는 선택지를 후보
              카드와 같은 자리에 버튼으로 노출한다(요청: "사용자가 주어진 선택지로 참여").
              클릭하면 선택지 라벨을 그대로 자유텍스트로 보낸다 — 후보 카드와 동일한 패턴이라
              백엔드 응답 계약을 바꾸지 않는다. 직접 입력은 아래 항상 떠 있는 입력창으로도
              가능하므로 별도 "직접 입력" 버튼을 강제하지 않는다(진행자가 필요하면 선택지
              안에 "직접 입력" 항목을 이미 포함해서 준다). */}
          {/* 가은/Claude(2026-07-27, 요청: "선택 버블을 눌렀는데 너무 늦게 사라짐") — 예전엔
              disabled={!canReplyOrContinue}만으로 막아서, 클릭 즉시 버튼이 비활성화(회색)될
              뿐 다음 진행자 응답이 도착할 때까지(수 초) 선택지 묶음 자체는 화면에 그대로
              남아 있었다(latestFacilitatorChoices는 phase/canonicalMessages가 실제로 바뀌어야
              갱신되는데, 그건 왕복 응답이 끝나야 일어난다). sending은 handleSend가 네트워크
              요청 전에 동기적으로 true가 되므로, 클릭 즉시 선택지 묶음 자체를 감춘다. */}
          {latestFacilitatorChoices.length > 0 && !sending && (
            <div
              role="group"
              aria-label="검증 방향 선택"
              style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 8, width: '100%' }}
            >
              <p style={{ margin: '0 0 2px', color: 'var(--text-2)', fontSize: 13.5, lineHeight: 1.5 }}>
                아래 선택지는 최종 주제 확정이 아니라, 다음 검증에서 적용할 방향입니다.
              </p>
              {latestFacilitatorChoices.map((choice, i) => {
                const detail = typeof choice.detail === 'string' ? choice.detail.trim() : ''
                return (
                  <button
                    key={choice.id || i}
                    type="button"
                    className="btn-ghost"
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      gap: detail && detail !== choice.label ? 5 : 0,
                      width: '100%',
                      padding: '11px 14px',
                      fontSize: 14.5,
                      borderRadius: 12,
                      textAlign: 'left',
                      whiteSpace: 'normal',
                      overflowWrap: 'anywhere',
                      lineHeight: 1.5,
                    }}
                    onClick={() => handleSend(choice.label)}
                    disabled={!canReplyOrContinue}
                  >
                    <strong>{choice.label}</strong>
                    {detail && detail !== choice.label && (
                      <span style={{ color: 'var(--text-2)', fontSize: 13.5, fontWeight: 400 }}>
                        {detail}
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
          )}
          {/* 용준/Claude(2026-07-27, 요청: discovery 모드 구조화 액션 버튼) — 위
              latestFacilitatorChoices/후보 카드와 같은 자리(대화 흐름 맨 아래)에, 현재
              phase에 맞는 블록 하나만 조건부로 렌더링한다. handleSend가 세 번째 인자로
              actionCode/actionPayload를 받아 그대로 API에 실어 보낸다. */}
          {phase === 'awaiting_problem_focus_selection' && (ideationConv?.problem_areas?.length || 0) > 0 && (
            <ProblemAreaSelectionBlock
              problemAreas={ideationConv.problem_areas}
              externalEvidence={ideationConv.external_evidence}
              onSend={(msg, opts) => handleSend(msg, opts)}
              disabled={!canReplyOrContinue}
              pendingActionCode={pendingActionCode}
            />
          )}
          {phase === 'awaiting_conflict_resolution' && (ideationConv?.solution_directions?.length || 0) > 0 && (
            <ConflictResolutionBlock
              solutionDirections={ideationConv.solution_directions}
              onSend={(msg, opts) => handleSend(msg, opts)}
              disabled={!canReplyOrContinue}
              pendingActionCode={pendingActionCode}
            />
          )}
          {phase === 'awaiting_concept_confirmation' && (
            <ConceptConfirmationBlock
              provisionalIdea={ideationConv?.provisional_idea}
              validationResult={ideationConv?.validation_result}
              ideaEvolution={ideationConv?.idea_evolution}
              onSend={(msg, opts) => handleSend(msg, opts)}
              disabled={!canReplyOrContinue}
              pendingActionCode={pendingActionCode}
            />
          )}
          <span
            className="rb-chat-resize-handle"
            role="separator"
            aria-label="대화창 높이 조절"
            aria-orientation="horizontal"
            onPointerDown={handleChatResizePointerDown}
            onPointerMove={handleChatResizePointerMove}
            onPointerUp={handleChatResizePointerUp}
            onPointerCancel={handleChatResizePointerUp}
          >
            ↘
          </span>
        </div>

        {ideationConv && phase !== 'finalized' && (
          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key !== 'Enter') return
                if (canReplyOrContinue) handleSend()
              }}
              placeholder={
                phase === 'failed'
                  ? '회의 처리 오류가 발생했습니다. 위의 다시 시도 버튼을 눌러주세요.'
                  : !canReplyOrContinue
                    ? '전문가 응답을 기다리는 중입니다'
                    : phase === 'awaiting_candidate_selection'
                      ? '답변을 입력하거나 후보를 선택해 주세요.'
                      : phase === 'awaiting_user_decision'
                        ? '필요하면 의견을 남겨주세요 (선택 사항)'
                        : hasSelected
                          ? '선택한 아이디어에 대해 추가 의견을 입력해 주세요.'
                          : '답변을 입력하세요'
              }
              disabled={!canReplyOrContinue}
              style={{ flex: 1, background: 'var(--bg-1)', border: '1px solid var(--glass-border)', borderRadius: 10, padding: '10px 14px', color: 'var(--text-0)', fontSize: 15.5 }}
            />
            <button
              type="button"
              aria-label="메시지 보내기"
              className="btn-primary"
              style={{ padding: '10px 14px' }}
              disabled={!canReplyOrContinue || !draft.trim()}
              onClick={() => handleSend()}
            >
              <Send size={14} />
            </button>
          </div>
        )}
        </section>

        <section className="rb-idea-finalize">
        {/* 요청: "주제 확정하기" 버튼을 선택한 아이디어 카드 옆(바로 아래)으로 이동 —
            handleFinalize/canFinalize 로직은 그대로고, 위치만 rb-ideation-canvas-col에서
            이 카드 바로 아래로 옮겼다. */}
        {ideationConv && (
          <>
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
                : '주제 확정하기 →'}
            </button>
            {!canFinalize && ideationConv && phase !== 'finalized' && phase !== 'failed' && (
              <p style={{ fontSize: 14.5, fontWeight: 600, color: '#514a61', marginTop: 9, lineHeight: 1.55 }}>
                {!hasSelected && hasCandidates ? '먼저 아이디어 후보를 선택해 주세요.' : nextActionGuideFor(phase)}
              </p>
            )}
          </>
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
        </section>
      </div>

      {/* 용준/Claude(2026-07-25, 요청: "오른쪽 영역은 참여 AI 휴먼 전용 공간으로 사용") —
          아이디어 제안 목록은 채팅 안으로 옮겨갔으니 오른쪽 패널에는 더 이상 두지 않는다.
          아바타 패널을 이 열의 시각적 핵심으로 둔다. 데스크톱에서는 sticky, 태블릿
          이하에서는 중앙 아래로 내려간다(위 rb-ideation-side 참고).
          용준/Claude(2026-07-26, 요청: "아이디어 기획 캔버스는 참여 위원 오른쪽에") —
          캔버스/합의사항/CTA는 이 열 아래에 이어붙이지 않고 별도의 세 번째 열
          (rb-ideation-canvas-col)로 분리했다. */}
      <div className="rb-idea-right">
      <div className="rb-ideation-side">
        <div className="card glass rb-avatar-card">
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
              layout="row"
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
        {/* 가은/Claude(2026-07-27, 요청: "AI 아이디어 회의 페이지에 신청서 작성항목 패널
            없애줘") — 회의 중 오른쪽 패널에 노출하던 ApplicationFormPanel을 제거했다.
            신청서 초안은 이제 주제 확정 후 별도 페이지(ApplicationFormDraftScreen,
            stage="form_draft")에서만 보여준다 — ApplicationFormPanel 자체는 그 화면에서
            계속 재사용하므로 컴포넌트/import는 그대로 둔다.
            pge/Claude(2026-07-28, 요청: "아이디어 진행 상황 패널은 아예 안 불러오게 —
            그 자리엔 진행 상황이 아니라 아이디어 변화 과정(IdeaEvolutionTimeline)이
            들어가야 함") — IdeationProgressPanel은 이 화면에서 import·렌더링을 완전히
            뺐다(파일 자체는 다른 곳에서 쓸 수 있으니 지우지 않는다). 탭 전환 자리에는
            기존에 아래쪽에 항상 떠 있던 IdeaEvolutionTimeline을 그대로 옮겨왔다 —
            idea_evolution 데이터가 있을 때만 두 번째 탭 버튼이 나타난다(예전 단독
            렌더링과 같은 게이팅 조건, 요청: "idea_evolution?.length > 0일 때만"). */}
        {/* pge/Claude(2026-07-28, 요청: "탭이랑 패널이 붙어있어야 해 — 떠 있으면 안 됨") —
            탭 줄과 내용을 별도 박스 두 개로 겹쳐 붙이려던 첫 시도(그림자·둥근 모서리를 각자
            그리고 margin으로 겹치기)는 살짝의 오차에도 틈이 보였다. 그 대신 탭 줄과 내용을
            하나의 card glass 박스 안에 함께 넣는다 — 물리적으로 같은 상자이므로 절대
            떨어져 보일 수 없다. IdeaCanvasPanel/IdeaEvolutionTimeline에는 bare=true를 줘서
            자기 자신의 card glass 외곽(테두리·그림자·둥근 모서리)을 그리지 않고 내용만
            채우게 한다. */}
        {ideationConv && (() => {
          const hasEvolution = (ideationConv.idea_evolution?.length || 0) > 0
          return (
            <div className="card glass rb-canvas-tab-shell" style={{ marginBottom: 12 }}>
              {/* pge/Claude(2026-07-28, 요청: "회의 시작 전엔 탭 자체가 안 뜨더라 — 내용은
                  안 떠도 되는데 탭 두 개는 항상 고정") — 탭 줄 자체는 hasEvolution과 무관하게
                  항상 렌더링한다. idea_evolution 데이터가 아직 없을 때만 "변화 과정" 탭의
                  내용 영역이 안내 문구로 대체된다(아래 참고). */}
              <div style={{ display: 'flex', borderBottom: '1px solid var(--glass-border)' }}>
                {[
                  { key: 'canvas', label: '기획 캔버스', Icon: ClipboardList },
                  { key: 'evolution', label: '아이디어 변화 과정', Icon: History },
                ].map(({ key, label, Icon }) => {
                  const active = canvasColumnTab === key
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setCanvasColumnTab(key)}
                      style={{
                        flex: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 4,
                        whiteSpace: 'nowrap',
                        padding: '10px 8px',
                        fontSize: 12.5,
                        fontWeight: 700,
                        cursor: 'pointer',
                        border: 'none',
                        borderBottom: active ? '2px solid var(--purple)' : '2px solid transparent',
                        background: active ? 'var(--bg-1)' : 'transparent',
                        color: active ? 'var(--text-0)' : 'var(--text-2)',
                        marginBottom: -1,
                        transition: 'background 0.12s ease, color 0.12s ease',
                      }}
                    >
                      <Icon size={14} color={active ? 'var(--purple)' : 'currentColor'} style={{ flexShrink: 0 }} />
                      {label}
                    </button>
                  )
                })}
              </div>
              <div className="rb-canvas-tab-scroll">
                {canvasColumnTab === 'evolution' ? (
                  hasEvolution ? (
                    <IdeaEvolutionTimeline
                      idea_evolution={ideationConv.idea_evolution}
                      solution_directions={ideationConv.solution_directions}
                      problem_areas={ideationConv.problem_areas}
                      bare
                    />
                  ) : (
                    <div style={{ padding: 18, fontSize: 13.5, color: 'var(--text-2)' }}>
                      아직 표시할 변화 이력이 없어요. 회의가 진행되면 여기에 쌓여요.
                    </div>
                  )
                ) : (
                  <IdeaCanvasPanel ideationConv={ideationConv} analysis={announcementAnalysis} bare />
                )}
              </div>
            </div>
          )
        })()}

      </div>
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
              ? humanizeExpertIdentifiers([item.risk, item.mitigation].filter(Boolean).join(' → ') || JSON.stringify(item))
              : humanizeExpertIdentifiers(String(item))}
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
            <strong style={{ fontWeight: 600 }}>{humanizeExpertIdentifiers(k)}</strong> · {humanizeExpertIdentifiers(String(v))}
          </li>
        ))}
      </ul>
    )
  }
  return humanizeExpertIdentifiers(String(value))
}

// 용준/Claude(2026-07-28, 요청: "후보 선택부터 주제 확정까지 추가 입력 없이 한 번의
// 흐름으로") — awaiting_concept_confirmation phase 전용 요약 화면. 기존
// ConceptConfirmationBlock(대화 화면 안에 인라인으로 뜨던 카드, 그대로 둠)과 달리 여기는
// "주제 확정" 단계(IdeationResultScreen) 전용으로, 요청 스펙 그대로 딱 6개 정보 +
// 3개 버튼만 보여준다. onSend 시그니처는 handleSend와 동일(msg, {actionCode,
// actionPayload})해서 IdeationResultScreen이 그대로 replyIdeationConversation에 넘길 수 있다.
function ConceptConfirmationSummary({ provisionalIdea, validationResult, onSend, disabled, pendingActionCode }) {
  function handleConfirm() {
    onSend('이 방향으로 최종 확정합니다.', { actionCode: 'confirm_concept', actionPayload: {} })
  }
  function handleRevise() {
    const target = provisionalIdea?.title ? `‘${provisionalIdea.title}’ 후보를` : '현재 후보를'
    onSend(`${target} 다시 검토하고 수정해 주세요.`, { actionCode: 'revise_candidate', actionPayload: {} })
  }
  function handleChooseAnother() {
    onSend('다른 후보를 선택하겠습니다.', { actionCode: 'choose_another_candidate', actionPayload: {} })
  }
  const cautions = [
    ...(validationResult?.planning?.status === 'passed_with_caution' ? validationResult.planning.issues || [] : []),
    ...(validationResult?.technical?.status === 'passed_with_caution' ? validationResult.technical.issues || [] : []),
  ]

  const rows = [
    ['최종 문제', provisionalIdea?.problem],
    ['대상 사용자', provisionalIdea?.target_user],
    ['핵심 해결 방식', provisionalIdea?.solution],
    ['기획 검증 요약', validationResult?.planning?.message],
    ['개발 검증 요약', validationResult?.technical?.message],
  ].filter(([, value]) => !!value)

  return (
    <div className="card glass" style={{ padding: 20 }}>
      {rows.map(([label, value], i) => (
        <div key={label} style={{ paddingTop: i > 0 ? 14 : 0, marginTop: i > 0 ? 14 : 0, borderTop: i > 0 ? '1px solid var(--glass-border)' : 'none' }}>
          <div style={{ fontSize: 13, color: 'var(--text-2)', fontFamily: 'var(--mono)', marginBottom: 4 }}>{label}</div>
          <div style={{ fontSize: 15, lineHeight: 1.65 }}>{value}</div>
        </div>
      ))}
      {cautions.length > 0 && (
        <div className="rb-ideation-notice" style={{ marginTop: 16 }}>
          남은 주의사항 · {cautions.map((item) => item.description).filter(Boolean).join(' · ')}
        </div>
      )}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 18 }}>
        <button type="button" className="btn-primary" disabled={disabled} onClick={handleConfirm}>
          {pendingActionCode === 'confirm_concept' ? '확정 중...' : '이 아이디어로 확정'}
        </button>
        <button type="button" className="btn-ghost" disabled={disabled} onClick={handleRevise}>
          {pendingActionCode === 'revise_candidate' ? '요청 중...' : '한 번 수정하기'}
        </button>
        <button type="button" className="btn-ghost" disabled={disabled} onClick={handleChooseAnother}>
          {pendingActionCode === 'choose_another_candidate' ? '이동 중...' : '다른 후보 선택'}
        </button>
      </div>
    </div>
  )
}

export function IdeationResultScreen({ ideationConv, setIdeationConv, onBack, onNext, onReturnToConversation }) {
  // 용준/Claude(2026-07-28) — 후보 선택 직후 자동으로 넘어온 "주제 확정 대기"
  // (awaiting_concept_confirmation) 상태 전용 분기. 확정/수정/다른 후보 액션은 채팅
  // 스트리밍이 필요 없는 단발 요청이라 handleSend의 스트리밍 로직 없이
  // replyIdeationConversation을 직접 호출한다 — 응답이 오면 대화가 다시 진행되어야
  // 하므로(확정→refinement 라운드, 수정→conflict_and_merge, 다른 후보→candidate_selection)
  // onReturnToConversation으로 "주제 아이디어 회의" 화면으로 돌려보낸다.
  const [confirmSending, setConfirmSending] = useState(false)
  const [confirmError, setConfirmError] = useState(null)
  const [pendingActionCode, setPendingActionCode] = useState(null)
  // 가은/Claude(2026-07-27, 요청: "주제 확정하고 아래에 신청서 초안 버튼 하나 만들어서
  // 페이지로 하나 띄워주자") — 신청서 항목을 선택한 세션에서만 버튼을 보여준다(선택 안 한
  // 세션엔 채울 필드 자체가 없다).
  const [generatingFormDraft, setGeneratingFormDraft] = useState(false)
  const [formDraftError, setFormDraftError] = useState(null)
  // 용준/Claude(2026-07-28) — 이 화면이 awaiting_concept_confirmation과 finalized 두
  // phase를 조건부로 분기해서 그리다 보니, 아래 모든 useState는 Hooks 규칙(매 렌더 동일한
  // 순서/개수로 호출)을 지키기 위해 두 분기의 조건문보다 항상 먼저(무조건) 선언한다 —
  // 그중 하나라도 조건부 return 아래에 있으면, phase가 바뀌어 분기가 달라지는 순간
  // "Rendered fewer/more hooks than expected" 에러로 화면이 통째로 날아간다.

  async function handleConfirmAction(message, { actionCode, actionPayload } = {}) {
    if (!ideationConv?.session_id || confirmSending) return
    setConfirmSending(true)
    setConfirmError(null)
    setPendingActionCode(actionCode || null)
    try {
      const data = await replyIdeationConversation(ideationConv.session_id, message, undefined, actionCode, actionPayload)
      setIdeationConv(data)
      onReturnToConversation?.()
    } catch (err) {
      setConfirmError(classifyIdeationConvError(err))
    } finally {
      setConfirmSending(false)
      setPendingActionCode(null)
    }
  }

  async function handleGenerateFormDraft() {
    if (generatingFormDraft) return
    setFormDraftError(null)
    setGeneratingFormDraft(true)
    try {
      const data = await generateApplicationFormDraft(ideationConv.session_id)
      setIdeationConv?.(data)
      onNext?.()
    } catch (err) {
      setFormDraftError(classifyIdeationConvError(err))
    } finally {
      setGeneratingFormDraft(false)
    }
  }

  if (ideationConv?.phase === 'awaiting_concept_confirmation') {
    return (
      <div style={{ maxWidth: 780 }}>
        <div className="rb-page-eyebrow-row">
          {onBack && (
            <button type="button" className="rb-back-button" onClick={onBack} aria-label="이전 화면으로 이동">
              {'←'}
            </button>
          )}
          <div className="rb-page-eyebrow">AI REVIEW BOARD</div>
        </div>
        <div className="badge purple mono" style={{ marginBottom: 12 }}>주제 확정 대기 · 전문가 검증 완료</div>
        {confirmError && <p style={{ color: 'var(--coral)', fontSize: 14.5, marginBottom: 12 }}>{confirmError.message}</p>}
        <ConceptConfirmationSummary
          provisionalIdea={ideationConv?.provisional_idea}
          validationResult={ideationConv?.validation_result}
          onSend={handleConfirmAction}
          disabled={confirmSending}
          pendingActionCode={pendingActionCode}
        />
      </div>
    )
  }

  if (!ideationConv || ideationConv.phase !== 'finalized' || !ideationConv.idea_proposal) {
    return (
      <div style={{ maxWidth: 760 }}>
        <div className="rb-page-title-row">
          <h2 style={{ fontSize: 21, fontWeight: 700 }}>주제 발전 회의를 먼저 완료해 주세요</h2>
          <span className="badge amber mono">아직 확정되지 않음</span>
        </div>
        <p style={{ fontSize: 14.5, color: 'var(--text-2)' }}>
          "주제 아이디어 회의" 단계에서 후보를 선택하고 전문가 질문에 답한 뒤, 확정 버튼을 눌러야 결과가 만들어져요.
        </p>
      </div>
    )
  }

  const proposal = ideationConv.idea_proposal

  return (
    <div style={{ maxWidth: 780 }}>
      <div className="rb-page-eyebrow-row">
        {onBack && (
          <button type="button" className="rb-back-button" onClick={onBack} aria-label="이전 화면으로 이동">
            {'←'}
          </button>
        )}
        <div className="rb-page-eyebrow">AI REVIEW BOARD</div>
      </div>
      <div className="card glass">
        <div className="rb-page-title-row" style={{ paddingBottom: 18, borderBottom: '1px solid var(--glass-border)' }}>
          <h2 style={{ fontSize: 27, fontWeight: 800 }}>
            {proposal.idea_name || '확정된 아이디어 주제'}
          </h2>
          <span className="badge green mono">주제 확정 · 기획서 작성 출발점</span>
        </div>
        {PROPOSAL_ROWS.map(([key, label], i) => (
          <div key={key} style={{ display: 'grid', gridTemplateColumns: '150px 1fr', gap: 16, padding: '14px 0', borderTop: i > 0 ? '1px solid var(--glass-border)' : 'none' }}>
            <div style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-2)' }}>{label}</div>
            <div style={{ fontSize: 14.5, lineHeight: 1.6 }}>{proposalValueDisplay(proposal[key])}</div>
          </div>
        ))}
      </div>

      {(ideationConv.application_form_items || []).length > 0 && (
        <div style={{ marginTop: 20 }}>
          {formDraftError && (
            <p style={{ color: 'var(--coral)', fontSize: 14.5, marginBottom: 10 }}>{formDraftError.message}</p>
          )}
          <button
            type="button"
            className="btn-primary"
            style={{ display: 'flex', alignItems: 'center', gap: 8 }}
            onClick={handleGenerateFormDraft}
            disabled={generatingFormDraft}
          >
            {generatingFormDraft ? '신청서 초안 작성 중...' : '신청서 초안 만들기'}
          </button>
        </div>
      )}
    </div>
  )
}

// 가은/Claude(2026-07-27, 요청: "주제 확정하고 아래에 신청서 초안 버튼 하나 만들어서 페이지로
// 하나 띄워주자") — IdeationResultScreen의 "신청서 초안 만들기" 버튼으로만 진입한다(goNext).
// 필드 렌더링 자체는 새로 만들지 않고 기존 ApplicationFormPanel(회의 화면 오른쪽 패널에서
// 이미 쓰던 컴포넌트)을 그대로 재사용한다 — items/draft 스키마가 동일하기 때문.
export function ApplicationFormDraftScreen({ ideationConv, onBack, onGoMain, onGoFeedback }) {
  const items = ideationConv?.application_form_items || []
  const draft = ideationConv?.application_form_draft || []

  if (!ideationConv || items.length === 0) {
    return (
      <div style={{ maxWidth: 760 }}>
        <div className="rb-page-eyebrow-row">
          {onBack && (
            <button type="button" className="rb-back-button" onClick={onBack} aria-label="이전 화면으로 이동">
              {'←'}
            </button>
          )}
          <div className="rb-page-eyebrow">AI REVIEW BOARD</div>
        </div>
        <div className="rb-page-title-row">
          <h2 style={{ fontSize: 21, fontWeight: 700 }}>이 세션에는 선택된 신청서 항목이 없어요</h2>
          <span className="badge amber mono">신청서 항목 없음</span>
        </div>
        <p style={{ fontSize: 14.5, color: 'var(--text-2)' }}>
          회의를 시작하기 전 신청서 양식 항목을 선택해야 초안을 만들 수 있어요.
        </p>
      </div>
    )
  }

  const draftById = new Map(draft.map((row) => [row.field_id, row]))
  const unfinishedCount = items.filter((item, i) => {
    const row = draftById.get(item.field_id || `form_field_${i + 1}`)
    return !row?.value
  }).length
  // 가은/Claude(2026-07-27, 요청: "보완이 필요한 정보 섹션") — 본문에 넣지 못한(입력에
  // 없어서 지어낼 수 없었던) 항목을 별도 카드로 보여준다. 본문 값과 섞이지 않게 항상
  // 분리해서 보여준다(프롬프트의 [출력 규칙] 4번과 대응).
  const supplementNotes = ideationConv.application_form_supplement_notes || []

  return (
    <div style={{ maxWidth: 780 }}>
      <div className="rb-page-eyebrow-row">
        {onBack && (
          <button type="button" className="rb-back-button" onClick={onBack} aria-label="이전 화면으로 이동">
            {'←'}
          </button>
        )}
        <div className="rb-page-eyebrow">AI REVIEW BOARD</div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
        <h2 style={{ fontSize: 23, fontWeight: 700, margin: 0 }}>
          {ideationConv.idea_proposal?.idea_name || '확정된 아이디어'}
        </h2>
        <span className="badge green mono">신청서 초안</span>
      </div>
      <p style={{ fontSize: 14, color: 'var(--text-2)', marginBottom: 16, lineHeight: 1.6 }}>
        회의에서 확정한 아이디어를 근거로 채운 초안입니다. 제출 전 내용을 꼭 검토해 주세요.
        {unfinishedCount > 0 && ` (아직 채워지지 않은 항목 ${unfinishedCount}개)`}
      </p>
      <ApplicationFormPanel items={items} draft={draft} />
      {supplementNotes.length > 0 && (
        <div className="card glass" style={{ marginTop: 16, borderColor: 'var(--amber, var(--coral))' }}>
          <div style={{ fontSize: 14.5, fontWeight: 700, marginBottom: 10 }}>보완이 필요한 정보</div>
          <p style={{ fontSize: 13, color: 'var(--text-2)', marginBottom: 10, lineHeight: 1.6 }}>
            회의·확정 결과에 없어서 초안에 넣지 못한 내용이에요. 제출 전 직접 확인해서 채워주세요.
          </p>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 14, lineHeight: 1.8 }}>
            {supplementNotes.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
        </div>
      )}
      {/* pge/Claude(2026-07-29, 요청: "패널 오른쪽 아래에 메인으로 돌아가기 버튼, 새 분석
          시작으로") — 신청서 초안까지 끝난 뒤 이 세션을 마무리하고 EntryScreen("새 분석
          시작")으로 완전히 새로 시작할 수 있는 출구. */}
      {(onGoMain || onGoFeedback) && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: 10, marginTop: 24, flexWrap: 'wrap' }}>
          {onGoMain && (
            <button type="button" className="btn-ghost" onClick={onGoMain}>
              메인으로 가기
            </button>
          )}
          {onGoFeedback && (
            <button type="button" className="btn-primary" onClick={onGoFeedback}>
              피드백 받으러 가기
            </button>
          )}
        </div>
      )}
    </div>
  )
}
