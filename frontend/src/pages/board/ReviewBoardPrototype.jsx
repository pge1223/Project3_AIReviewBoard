import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Link2, Upload, FileText, Sparkles,
  CheckCircle2, Circle, AlertCircle, AlertTriangle, Award, Target, ShieldCheck,
  ArrowRight, TrendingUp, ChevronDown, ChevronUp, ChevronRight, Calendar, FolderOpen, X, Trash2,
  Menu, User, LogOut, ExternalLink, Gift, AlertOctagon, Quote, FileStack,
} from "lucide-react";
import { createProject, getProject, updateProject, getLatestMeeting } from "../../api/projectApi";
import {
  fetchUrl as fetchCriteriaUrl,
  uploadDocument,
  getAnnouncementAnalysis,
  getApplicationFormAnalysis,
  getContestWorksByTitle,
  deleteDocument,
  getDocuments,
} from "../../api/documentApi";
import { analyzeProject, getAnalyzeProgress, getMentorCandidates } from "../../api/projectApi";
import { isAcceptedDocument, formatFileSize, ACCEPTED_DOCUMENT_EXTENSIONS } from "../../utils/file";
import { assessCriteriaContent } from "../../utils/criteriaAssessment";
import { pollDocumentIndexing } from "../../utils/documentIndexingPoll";
import { parseEvaluationCriteria, summarizeCriteria, CONFIDENCE_LABEL } from "../../utils/contestAnalysisDisplay";
import ProgressBar from "../../components/common/ProgressBar";
import WorkbenchScreen from "./WorkbenchScreen";
// 경이/Claude(2026-07-22): "AI 피드백" 다음 "완성 리포트" 단계 — 경이가 설계한 버전 추적형
// 리포트(VersionTrackerTestPage, 애니메이션/버전추적/프로필 토글/이전·현재 비교 포함)를
// 흐름 안에 embedded 모드로 끼워 넣는다. 이 파일(가은님 소유)의 변경은 워크벤치와 같은
// 방식으로 라벨/흐름/렌더 3곳만 최소화했다.
import VersionTrackerTestPage from "../VersionTrackerTestPage";
import { IdeationScreen, IdeationResultScreen } from "./IdeationConversationScreen";

/* 가은/Claude(2026-07-20): "작성 전(주제 발굴)/작성 후(문서 피드백)" 2-모드
 * 신규 플로우. docs/REVIEW_BOARD_서비스_방향성_정리_20260720.md의 방향을
 * 구현한 화면으로, "작성 전(주제 발굴)" 경로(ideation/ideation_result)는 아직
 * 더미 데이터다 — 백엔드에 주제 발굴 회의 API가 없어서다.
 * "작성 후(문서 피드백)" 경로는 기존 /projects/new(평가 대상 문서 업로드)와
 * MentorSelectionPage(분석 시작 → analyzeProject)의 실제 로직을 그대로
 * 재사용한다 — 다만 멘토 선택 화면 없이 추천 멘토를 자동으로 전부 쓴다.
 * 배경·색 팔레트는 웜 화이트(#faf8f4) 고정 디자인이다(가은 확정) — OS
 * 다크모드를 따라 자동 전환하지 않는다.
 */

const STAGE_LABELS = {
  entry: "공모전 입력",
  analysis: "공모전 분석",
  // 가은/Claude(2026-07-24, 요청: 공모전 분석 결과 화면 개편) — 내부 stage 키("ideation")와
  // 라우팅/데이터는 그대로 두고, 화면에 노출되는 문구만 팀 UX 레퍼런스에 맞춰 "AI 아이디어
  // 회의"로 바꾼다. IdeationConversationScreen.jsx 쪽 배지 등 이 상수를 안 쓰는 곳은 영향 없음.
  ideation: "AI 아이디어 회의",
  ideation_result: "주제 확정",
  upload: "기획서 업로드 · 분석",
  // 재인/Claude(2026-07-21): docs/REVIEW_BOARD_서비스_방향성_정리_20260720.md의
  // "5. 핵심 UI: 시각적 인터랙티브 워크벤치" 구현 — 실제 화면은
  // WorkbenchScreen.jsx(신규 파일)에 분리해서, 이 파일(가은님 소유)의 변경은
  // 이 라벨/흐름 추가 정도로 최소화했다.
  workbench: "AI 피드백",
  report: "완성 리포트",
};

// 가은/Claude(2026-07-24, 요청: 공모전 분석 결과 화면 개편) — 왼쪽 단계 메뉴와 오른쪽
// 분석 진행 상태 패널이 같은 문구를 공유한다. 실제 분석 데이터가 아니라 "이 단계에서 뭘
// 하는가"를 설명하는 고정 안내문이라 API 값과 무관하게 둬도 값을 지어내는 게 아니다.
const STAGE_DESCRIPTIONS = {
  entry: "공모전 자료 등록이 완료되었습니다.",
  analysis: "공모전의 핵심 내용과 평가 기준을 분석합니다.",
  ideation: "AI 전문가들이 아이디어를 논의합니다.",
  ideation_result: "최종 아이디어를 선택하고 확정합니다.",
  upload: "평가받을 기획서를 업로드하고 분석을 시작합니다.",
  workbench: "AI 위원들의 피드백을 확인합니다.",
  report: "버전별 개선 결과를 확인합니다.",
};

const FLOW_BY_MODE = {
  pre: ["entry", "analysis", "ideation", "ideation_result"],
  // 가은/Claude(2026-07-21): 실측 요청 — "작성 후(문서 피드백)"로 들어오면 공모전 분석
  // 화면 없이 바로 기획서 업로드·분석으로 간다. 공모전 분석은 주제를 정하기 전(작성 전)
  // 에나 필요한 단계라서다. entry에서 등록한 공고문(criteria)은 화면만 안 거칠 뿐,
  // 색인은 그대로 되어 피드백 때 심사기준 근거로 쓰인다.
  post: ["entry", "upload", "workbench", "report"],
};

// 가은/Claude(2026-07-21): "작성 전" 흐름에서 확정한 아이디어 프로젝트를 표시하는 마커.
// 아직 실제 주제 발굴 회의 API가 없어(더미) 확정 주제도 IdeationResultScreen과 동일한
// 고정 값을 쓴다 — 실제 회의 API가 붙으면 이 상수 대신 회의 결과의 주제명을 저장한다.
export const IDEA_PROJECT_MARKER = "작성 전 주제 발굴 흐름에서 확정한 아이디어 프로젝트입니다.";
const IDEA_PROJECT_TOPIC = "예비창업인 재고관리 AI 비서";

const MIN_MENTORS = 2
const MAX_MENTORS = 4
const ANALYZE_POLL_INTERVAL_MS = 1000

// 가은/Claude(2026-07-21): 실측 요청 — "내 프로젝트"에서 이전 프로젝트를 불러오면
// board에서 올렸던 공고문·기획서가 안 보이던 문제. DocumentUploadPage.jsx의
// toRowStatus()와 같은 취지(백엔드 status -> 화면 배지)지만, 재조회 시점엔 색인이
// 이미 끝났을 확률이 높아 "진행 중" 상태를 굳이 추정하지 않고 done으로 본다.
function _resumedDocStatus(backendStatus) {
  if (backendStatus === 'indexed_empty') return 'warning'
  if (backendStatus === 'indexing_failed' || backendStatus === 'conversion_failed' || backendStatus === 'indexing_timeout') return 'error'
  return 'done'
}

function Shell({ children, active, mode, onNavigate, showNav }) {
  const flow = mode ? FLOW_BY_MODE[mode] : ["entry"];
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);

  function handleLogout() {
    localStorage.removeItem('auth_token');
    setMenuOpen(false);
    navigate('/login');
  }

  return (
    <div className="rb-root">
      <style>{`
        .rb-root{
          --bg-0:#faf8f4; --bg-1:#ffffff; --bg-2:#f1eee5;
          --glass: rgba(255,255,255,0.72); --glass-border: rgba(28,26,46,0.10);
          --purple:#7c5cea; --purple-dim: rgba(124,92,234,0.12);
          --coral:#e0603d; --coral-dim: rgba(224,96,61,0.12);
          --green:#16a37a; --green-dim: rgba(22,163,122,0.12);
          --amber:#b8830b; --amber-dim: rgba(184,131,11,0.14);
          --rose:#c23a6b; --rose-dim: rgba(194,58,107,0.12);
          --text-0:#1c1a2e; --text-1:#5b5770; --text-2:#918d9f;
          --cream:#f4efe2; --cream-line:#e7ddc4;
          --mono: 'JetBrains Mono', ui-monospace, monospace;
        }
        .rb-root{
          position:relative; min-height:100vh; width:100%;
          background:
            radial-gradient(1100px 600px at 12% -10%, rgba(124,92,234,0.10), transparent 60%),
            radial-gradient(900px 500px at 100% 10%, rgba(22,163,122,0.07), transparent 55%),
            radial-gradient(800px 500px at 50% 110%, rgba(224,96,61,0.06), transparent 55%),
            var(--bg-0);
          color:var(--text-0);
          font-family:'Pretendard', -apple-system, sans-serif;
          font-size:15px;
          display:flex;
        }
        .rb-root .glass{ background:var(--glass); border:1px solid var(--glass-border); backdrop-filter: blur(14px); box-shadow: 0 2px 14px rgba(28,26,46,0.05); }
        .rb-root .mono{ font-family:var(--mono); letter-spacing:0.02em; }
        .rb-root .navrail{ width:220px; flex-shrink:0; border-right:1px solid var(--glass-border); padding:24px 14px; }
        .rb-root .navstep{ position:relative; padding-bottom:20px; }
        .rb-root .navstep:last-child{ padding-bottom:0; }
        .rb-root .navstep-line{ position:absolute; left:17px; top:28px; bottom:0; width:1.5px; background:var(--glass-border); }
        .rb-root .navstep-row{ display:flex; align-items:flex-start; gap:10px; padding:6px; margin:-6px; border-radius:10px; text-align:left; background:none; border:none; width:100%; font:inherit; color:inherit; }
        .rb-root .navstep-done .navstep-row, .rb-root .navstep-current .navstep-row{ cursor:pointer; }
        .rb-root .navstep-done .navstep-row:hover, .rb-root .navstep-current .navstep-row:hover{ background:var(--bg-2); }
        .rb-root .navstep-upcoming .navstep-row{ cursor:default; }
        .rb-root .navstep-dot{ width:22px; height:22px; border-radius:999px; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0; margin-top:1px; }
        .rb-root .navstep-done .navstep-dot{ background:var(--green-dim); color:var(--green); }
        .rb-root .navstep-current .navstep-dot{ background:var(--purple); color:#fff; }
        .rb-root .navstep-upcoming .navstep-dot{ background:var(--bg-2); color:var(--text-2); }
        .rb-root .navstep-label{ font-size:14px; font-weight:600; color:var(--text-1); }
        .rb-root .navstep-current .navstep-label{ color:var(--text-0); font-weight:700; }
        .rb-root .navstep-upcoming .navstep-label{ color:var(--text-2); }
        .rb-root .navstep-desc{ font-size:12px; color:var(--text-2); margin-top:2px; line-height:1.5; }
        .rb-root .main{ flex:1; min-width:0; padding:32px 40px; overflow-y:auto; }
        .rb-root .badge{ display:inline-flex; align-items:center; gap:6px; font-size:12px; padding:3px 9px; border-radius:99px; font-family:var(--mono); }
        .rb-root .badge.purple{ background:var(--purple-dim); color:var(--purple); }
        .rb-root .badge.coral{ background:var(--coral-dim); color:var(--coral); }
        .rb-root .badge.green{ background:var(--green-dim); color:var(--green); }
        .rb-root .badge.amber{ background:var(--amber-dim); color:var(--amber); }
        .rb-root .badge.grey{ background:var(--bg-2); color:var(--text-2); }
        .rb-root .btn-primary{ background:linear-gradient(135deg, var(--purple), #8b6ef0); color:#0b0a16; font-weight:600; border:none; border-radius:12px; padding:11px 20px; cursor:pointer; font-size:15px; }
        .rb-root .btn-primary:disabled{ opacity:0.4; cursor:not-allowed; }
        .rb-root .btn-ghost{ background:transparent; border:1px solid var(--glass-border); color:var(--text-1); border-radius:12px; padding:10px 18px; cursor:pointer; font-size:15px; }
        .rb-root .btn-ghost:hover{ background:var(--bg-2); }
        .rb-root .card{ border-radius:16px; padding:20px; }
        .rb-root .progress-track{ height:8px; border-radius:999px; background:var(--bg-2); overflow:hidden; }
        .rb-root .progress-fill{ height:100%; border-radius:999px; background:linear-gradient(135deg, var(--purple), #8b6ef0); transition:width .5s ease; }
        .rb-root .rb-top-menu{ position:absolute; top:20px; right:24px; z-index:10; }
        .rb-root .rb-icon-btn{ width:40px; height:40px; display:inline-flex; align-items:center; justify-content:center; padding:0; border:none; background:transparent; color:var(--text-1); border-radius:10px; cursor:pointer; }
        .rb-root .rb-icon-btn:hover{ background:var(--bg-2); color:var(--text-0); }
        .rb-root .rb-menu-panel{ position:absolute; top:48px; right:0; width:168px; border-radius:12px; padding:6px; }
        .rb-root .rb-menu-item{ width:100%; display:flex; align-items:center; gap:8px; padding:10px 11px; border:none; border-radius:9px; background:transparent; color:var(--text-1); font-size:14px; cursor:pointer; text-align:left; }
        .rb-root .rb-menu-item:hover{ background:var(--bg-2); color:var(--text-0); }
        .rb-root .rb-entry-actions{ display:flex; justify-content:flex-end; margin-bottom:10px; }
        .rb-root .rb-inline-projects{ display:inline-flex; align-items:center; gap:6px; padding:8px 12px; font-size:15px; font-family:inherit; font-weight:400; letter-spacing:0; }
        .rb-root .rb-back-button{ width:26px; height:26px; display:inline-flex; align-items:center; justify-content:center; border:none; background:transparent; color:#000; border-radius:8px; cursor:pointer; font-size:13px; font-weight:300; line-height:1; padding:0; flex-shrink:0; }
        .rb-root .rb-back-button:hover{ background:var(--bg-2); color:var(--text-0); }
        .rb-root .rb-typing-cursor{ display:inline-block; margin-left:1px; animation: rb-blink 0.9s steps(1) infinite; }
        @keyframes rb-blink{ 0%,49%{ opacity:1; } 50%,100%{ opacity:0; } }
        @media (max-width: 780px){
          .rb-root .rb-top-menu{ top:10px; right:12px; }
          .rb-root{ flex-direction:column; }
          .rb-root .navrail{ display:none; }
          .rb-root .main{ padding:20px; }
          .rb-grid-2{ grid-template-columns: 1fr !important; }
        }
      `}</style>

      {/* 가은/Claude(2026-07-21): 기존 /projects 페이지(레거시 스타일 그대로 유지)로
          돌아갈 수 있는 진입점 — 버튼 자체만 board 톤(glass/badge 스타일)으로 맞춘다. */}
      <div className="rb-top-menu">
        <button
          type="button"
          className="rb-icon-btn"
          aria-label="메뉴 열기"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          <Menu size={20} />
        </button>
        {menuOpen && (
          <div className="rb-menu-panel glass">
            <button type="button" className="rb-menu-item" onClick={() => { setMenuOpen(false); navigate('/mypage'); }}>
              <User size={15} /> 마이페이지
            </button>
            <button type="button" className="rb-menu-item" onClick={handleLogout}>
              <LogOut size={15} /> 로그아웃
            </button>
          </div>
        )}
      </div>

      {showNav && (
        <div className="navrail glass">
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 20, letterSpacing: "0.02em" }}>
            AI Review Board
          </div>
          {flow.map((k, idx) => {
            const activeIdx = flow.indexOf(active);
            const state = idx < activeIdx ? "done" : idx === activeIdx ? "current" : "upcoming";
            const reachable = state !== "upcoming";
            return (
              <div key={k} className={`navstep navstep-${state}`}>
                <button
                  type="button"
                  className="navstep-row"
                  onClick={reachable ? () => onNavigate(k) : undefined}
                  disabled={!reachable}
                  aria-current={state === "current" ? "step" : undefined}
                >
                  <span className="navstep-dot">
                    {state === "done" ? <CheckCircle2 size={13} /> : idx + 1}
                  </span>
                  <span>
                    <div className="navstep-label">{STAGE_LABELS[k]}</div>
                    {STAGE_DESCRIPTIONS[k] && <div className="navstep-desc">{STAGE_DESCRIPTIONS[k]}</div>}
                  </span>
                </button>
                {idx < flow.length - 1 && <div className="navstep-line" />}
              </div>
            );
          })}
        </div>
      )}
      <div className="main">{children}</div>
    </div>
  );
}

/* ---------------- 1. 진입화면 : 모드 선택 + 공고 입력 ---------------- */
/* 가은/Claude(2026-07-20): 실측 버그 — "공모전 공고 URL 입력하면 안 먹어" 제보.
 * 원인은 이 화면 URL 입력창이 그냥 텍스트 필드였고, "분석 시작"을 눌러야만(그것도
 * 조용히 실패를 삼키며) 한 번 fetchUrl을 시도했기 때문 — 실패해도 아무 표시가
 * 없었다. DocumentUploadPage.jsx(/projects/new)의 "기준 문서·공고문" 로직(URL 탭 +
 * 파일 업로드 탭, 전용 "가져오기" 버튼, 진행 상태·에러 표시, 색인 폴링)을 그대로
 * 옮겨왔다.
 *
 * 가은/Claude(2026-07-24, 요청: 첫 화면 개편) — 콘텐츠가 세로로만 길게 쌓이던 걸
 * "왼쪽 진행 단계 / 가운데 방식 선택·자료 등록 / 오른쪽 현재 설정 요약" 3열
 * 대시보드로 재구성했다. 상태·핸들러(모드 선택, URL 가져오기, 파일 업로드, 삭제,
 * 시작 조건)는 전부 그대로이고 레이아웃·표시만 바뀐다. 신규로 늘어난 것은
 * unsupportedLinks를 문서 데이터에 실어 보내(assessCriteriaContent가 이미 계산해
 * 주던 값인데 기존엔 meta 문장에만 녹여서 버렸다) "추가 파일 확인 필요" 경고를
 * 별도 영역으로 분리한 것뿐이다.
 */
const MODE_META = {
  pre: {
    title: "아이디어가 없어요",
    badge: "아이디어 발굴",
    description: "공모전을 분석하고 아이디어를 발굴하며 전문가 회의를 진행해요.",
    outcomes: ["공모전 핵심 분석", "평가 기준 및 배점 정리", "아이디어 후보와 전문가 회의"],
    icon: Sparkles,
    accent: "purple",
  },
  post: {
    title: "작성한 문서가 있어요",
    badge: "문서 피드백",
    description: "기획서나 제안서를 평가하고 개선 우선순위를 확인해요.",
    outcomes: ["항목별 평가 및 점수", "개선 우선순위 제안", "피드백 및 수정 가이드"],
    icon: FileText,
    accent: "coral",
  },
};

// 가은/Claude(2026-07-24): "분석 후 제공되는 결과"는 지어낸 기능 설명 대신, 이 모드가
// 실제로 거쳐가는 다음 단계 이름(STAGE_LABELS·FLOW_BY_MODE, 이 파일 상단에 이미 정의됨)
// 을 그대로 보여준다 — 사용자가 다음에 뭘 하게 되는지 정확히 예고한다.
function expectedResultLabels(mode) {
  if (!mode) return [];
  return FLOW_BY_MODE[mode].slice(1).map((k) => STAGE_LABELS[k]);
}

const ENTRY_PROGRESS_STEPS = [
  { key: "select", label: "새 분석 시작", desc: "분석 방식과 자료를 등록해 주세요." },
  { key: "analyzing", label: "분석 진행 중", desc: "AI가 공모전과 문서를 분석합니다." },
  { key: "result", label: "결과 확인", desc: "분석 결과와 인사이트를 확인합니다." },
];

function ModeCard({ meta, selected, onSelect }) {
  const Icon = meta.icon;
  function handleKeyDown(e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect();
    }
  }
  return (
    <div
      role="radio"
      aria-checked={selected}
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={handleKeyDown}
      className={`card glass es-mode-card ${selected ? "es-mode-card-selected" : ""}`}
    >
      <div className="es-mode-card-top">
        <Icon size={20} color={`var(--${meta.accent})`} />
        {selected ? <CheckCircle2 size={18} color="var(--purple)" /> : <Circle size={16} color="var(--glass-border)" />}
      </div>
      <span className={`badge ${meta.accent} mono`} style={{ marginTop: 10, width: "fit-content" }}>{meta.badge}</span>
      <div style={{ fontWeight: 700, fontSize: 15, margin: "10px 0 6px" }}>{meta.title}</div>
      <div style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.6, marginBottom: 12 }}>{meta.description}</div>
      <ul className="es-mode-outcomes">
        {meta.outcomes.map((o) => <li key={o}>{o}</li>)}
      </ul>
    </div>
  );
}

function EntryScreen({ onEnter, onModeSelect, loading, error, projectId, ensureProject, documents, setDocuments }) {
  const navigate = useNavigate();
  const [mode, setMode] = useState(null);
  const [dismissedAlerts, setDismissedAlerts] = useState([]);

  // 가은/Claude(2026-07-21): 실측 요청 — 여기서 모드 카드를 고른 시점과 "분석 시작"을
  // 누르는 시점 사이에 공고문을 먼저 등록하면(ensureProject가 여기서 바로 호출됨) 그
  // 시점엔 아직 부모의 mode가 안 알려져 있어 flow_mode 없이 프로젝트가 생겼다. 카드를
  // 고르자마자 부모에도 즉시 알려서 ensureProject()가 항상 올바른 flow_mode로 만들게 한다.
  function selectMode(m) {
    setMode(m);
    onModeSelect?.(m);
  }
  const [criteriaTab, setCriteriaTab] = useState('url');
  const [criteriaUrl, setCriteriaUrl] = useState('');
  const [criteriaLoading, setCriteriaLoading] = useState(false);
  const [criteriaError, setCriteriaError] = useState('');
  const [isCriteriaDragging, setIsCriteriaDragging] = useState(false);
  const [deletingIds, setDeletingIds] = useState([]);
  const criteriaFileInputRef = useRef(null);

  function updateDoc(id, patch) {
    setDocuments((prev) => prev.map((doc) => (doc.id === id ? { ...doc, ...patch } : doc)));
  }

  // 가은/Claude(2026-07-21): 실측 요청 — 잘못 올린 공고문·평가기준 문서를 지울 수 있게.
  // backendId(실제 DB document_id)가 아직 없으면(업로드/색인 중) 서버 삭제는 건너뛰고
  // 목록에서만 뺀다 — 곧 도착할 폴링/응답이 사라진 행을 되살리지 않도록 documents에서
  // 완전히 지운다(updateDoc과 달리 필터로 제거).
  async function handleDeleteDoc(doc) {
    if (deletingIds.includes(doc.id)) return
    setDeletingIds((prev) => [...prev, doc.id])
    try {
      if (doc.backendId && projectId) {
        await deleteDocument(projectId, doc.backendId)
      }
      setDocuments((prev) => prev.filter((d) => d.id !== doc.id))
    } catch (err) {
      setCriteriaError(err.message)
      setDeletingIds((prev) => prev.filter((id) => id !== doc.id))
    }
  }

  async function handleFetchCriteriaUrl() {
    if (!criteriaUrl.trim()) {
      setCriteriaError('URL을 입력해주세요.');
      return;
    }
    setCriteriaError('');
    setCriteriaLoading(true);
    try {
      const pid = await ensureProject();
      const result = await fetchCriteriaUrl(criteriaUrl.trim(), pid);
      const id = `url-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const title = result.page_content?.title || criteriaUrl.trim();
      const { status: contentStatus, meta: contentMeta, unsupportedLinks } = assessCriteriaContent(result);

      // 가은/Claude(2026-07-21): 실측 지적 — 여기서 원문 앞부분 300자를 그대로 미리보기로
      // 보여줬더니 메뉴/내비게이션 텍스트("공지사항 - 공지사항 - 뉴포커스 - ...")가 그대로
      // 나왔다. 원문 요약은 "공모전 분석" 화면(AnalysisScreen)이 LLM으로 뽑은 요약을 대신
      // 보여주므로, 여기서는 원문 슬라이스를 더 이상 안 만든다.
      if (result.document_status === 'indexing' && result.document_id) {
        setDocuments((prev) => [...prev, { id, backendId: result.document_id, type: 'url', name: title, meta: '공고문을 색인하는 중...', status: 'embedding', progress: 50, unsupportedLinks }]);
        pollDocumentIndexing(pid, result.document_id, id, contentStatus, contentMeta, updateDoc);
      } else {
        setDocuments((prev) => [...prev, { id, backendId: result.document_id, type: 'url', name: title, meta: contentMeta, status: contentStatus, unsupportedLinks }]);
      }
      setCriteriaUrl('');
    } catch (err) {
      setCriteriaError(err.message);
    } finally {
      setCriteriaLoading(false);
    }
  }

  async function uploadCriteriaFile(file) {
    const id = `${file.name}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setDocuments((prev) => [...prev, { id, type: 'file', name: file.name, meta: formatFileSize(file.size), status: 'embedding', progress: 50 }]);
    try {
      const pid = await ensureProject();
      const doc = await uploadDocument(pid, file, 'pdf', 'criteria');
      if (doc.status === 'conversion_failed') {
        updateDoc(id, { backendId: doc.id, status: 'error', meta: doc.conversion_metadata?.conversion_error || '문서를 변환하지 못했습니다.' });
        return;
      }
      // 가은/Claude(2026-07-21): 색인이 백그라운드로 바뀌어 업로드 응답이 "indexing"으로
      // 즉시 돌아온다 — 행은 "색인 중" 그대로 두고 폴링으로 완료를 확인한다.
      if (doc.status === 'indexing') {
        updateDoc(id, { backendId: doc.id, progress: 75 });
        pollDocumentIndexing(pid, doc.id, id, 'done', formatFileSize(file.size), updateDoc);
        return;
      }
      updateDoc(id, { backendId: doc.id, status: 'done', meta: formatFileSize(file.size) });
    } catch (err) {
      updateDoc(id, { status: 'error', meta: err.message });
    }
  }

  function addCriteriaFiles(fileList) {
    const files = Array.from(fileList);
    const accepted = files.filter(isAcceptedDocument);
    const rejected = files.length - accepted.length;
    setCriteriaError(rejected > 0 ? 'PDF, DOCX, PPTX, HWP, HWPX 파일만 업로드할 수 있습니다.' : '');
    accepted.forEach(uploadCriteriaFile);
  }

  const criteriaDropHandlers = {
    onDragOver: (e) => { e.preventDefault(); setIsCriteriaDragging(true); },
    onDragLeave: (e) => { e.preventDefault(); setIsCriteriaDragging(false); },
    onDrop: (e) => { e.preventDefault(); setIsCriteriaDragging(false); addCriteriaFiles(e.dataTransfer.files); },
  };

  // 가은/Claude(2026-07-22)의 정책(공고문·평가기준 등록 필수) — 그대로 유지: ① 문서가
  // 하나라도 done/warning이어야 하고 ② 색인 중인 문서가 없어야 "분석 시작"이 눌린다.
  const hasReadyCriteriaDoc = documents.some((doc) => doc.status === 'done' || doc.status === 'warning');
  const isIndexingCriteriaDoc = documents.some((doc) => doc.status === 'embedding');
  const isMaterialProcessing = isIndexingCriteriaDoc || criteriaLoading;
  const canStart = !!mode && hasReadyCriteriaDoc && !isIndexingCriteriaDoc && !criteriaLoading;
  const guide = !mode
    ? '분석 방식을 먼저 선택해 주세요.'
    : isMaterialProcessing
      ? '자료를 처리하는 중이에요 — 완료되면 시작할 수 있어요.'
      : !hasReadyCriteriaDoc
        ? '필수 공모전 자료를 등록해 주세요.'
        : '';

  const materialStatusCounts = documents.reduce((acc, doc) => {
    acc[doc.status] = (acc[doc.status] || 0) + 1;
    return acc;
  }, {});
  const materialStatusEntries = [
    ['분석 준비 완료', materialStatusCounts.done, 'green'],
    ['확인 필요', materialStatusCounts.warning, 'amber'],
    ['처리 중', materialStatusCounts.embedding, 'purple'],
    ['처리 실패', materialStatusCounts.error, 'coral'],
  ].filter(([, count]) => count > 0);

  const expectedResults = expectedResultLabels(mode);
  const checklist = [
    { done: !!mode, label: '분석 방식 선택' },
    { done: hasReadyCriteriaDoc, label: '공모전 자료 등록' },
    { done: !isMaterialProcessing, label: '자료 처리 완료' },
  ];

  return (
    <div style={{ maxWidth: 1500, margin: "0 auto" }}>
      <style>{`
        .es-header-row{ display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:28px; flex-wrap:wrap; }
        .es-eyebrow{ font-size:13px; font-weight:700; color:var(--purple); letter-spacing:.03em; margin-bottom:6px; }
        .es-title{ font-size:28px; font-weight:700; margin:0 0 6px; }
        .es-subtitle{ font-size:14.5px; color:var(--text-2); margin:0; }

        .es-layout{ display:grid; grid-template-columns:220px minmax(0,1fr) 320px; gap:28px; align-items:start; }
        @media (max-width:1180px){
          .es-layout{ grid-template-columns:minmax(0,1fr); }
          .es-progress-list{ flex-direction:row; overflow-x:auto; gap:20px; }
          .es-progress-step{ flex-direction:column; padding-bottom:0; min-width:150px; }
          .es-progress-step::before{ display:none; }
          .es-side{ position:static !important; }
        }

        .es-progress-list{ display:flex; flex-direction:column; }
        .es-progress-step{ display:flex; gap:12px; padding-bottom:22px; position:relative; }
        .es-progress-step:last-child{ padding-bottom:0; }
        .es-progress-step::before{ content:''; position:absolute; left:11px; top:26px; bottom:0; width:1.5px; background:var(--glass-border); }
        .es-progress-step:last-child::before{ display:none; }
        .es-progress-dot{ width:24px; height:24px; border-radius:999px; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0; }
        .es-progress-step.active .es-progress-dot{ background:var(--purple); color:#fff; }
        .es-progress-step.upcoming .es-progress-dot{ background:var(--bg-2); color:var(--text-2); }
        .es-progress-label{ font-size:14px; font-weight:700; }
        .es-progress-step.upcoming .es-progress-label{ color:var(--text-2); font-weight:600; }
        .es-progress-desc{ font-size:12.5px; color:var(--text-2); margin-top:2px; line-height:1.5; }

        .es-mode-grid{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px; }
        @media (max-width:640px){ .es-mode-grid{ grid-template-columns:1fr; } }
        .es-mode-card{ cursor:pointer; border:1.5px solid var(--glass-border); display:flex; flex-direction:column; transition:border-color .18s ease, box-shadow .18s ease; }
        .es-mode-card:hover{ border-color:var(--purple); }
        .es-mode-card:focus-visible{ outline:2px solid var(--purple); outline-offset:2px; }
        .es-mode-card-selected{ border:2px solid var(--purple); background:var(--purple-dim); box-shadow:0 4px 14px rgba(124,92,234,0.14); }
        .es-mode-card-top{ display:flex; justify-content:space-between; align-items:flex-start; }
        .es-mode-outcomes{ margin:0; padding-left:16px; font-size:13px; color:var(--text-1); line-height:1.8; }

        .es-material-section.disabled{ opacity:.55; }
        .es-tabs{ display:flex; gap:4px; background:var(--bg-2); border-radius:999px; padding:4px; margin-bottom:14px; width:fit-content; }
        .es-tab{ padding:7px 16px; border-radius:999px; border:none; cursor:pointer; font-size:14px; font-weight:600; font-family:inherit; background:transparent; color:var(--text-2); }
        .es-tab.active{ background:var(--bg-1); color:var(--purple); box-shadow:0 1px 4px rgba(28,26,46,0.1); }

        .es-doc-row{ display:flex; justify-content:space-between; align-items:center; padding:12px 0; gap:12px; }
        .es-doc-row + .es-doc-row{ border-top:1px solid var(--glass-border); }
        .es-doc-name{ font-size:14.5px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:320px; }
        .es-doc-meta{ font-size:12.5px; color:var(--text-2); margin-top:2px; }

        .es-alert{ display:flex; gap:10px; padding:14px 16px; border-radius:12px; background:var(--amber-dim); border:1px solid rgba(184,131,11,0.25); margin-bottom:12px; }
        .es-alert-title{ font-size:14px; font-weight:700; color:var(--amber); margin-bottom:4px; }
        .es-alert-body{ font-size:13px; color:var(--text-1); line-height:1.6; }
        .es-alert-actions{ display:flex; gap:8px; margin-top:10px; }

        .es-side{ position:sticky; top:24px; }
        .es-side-title{ font-size:15px; font-weight:700; margin-bottom:14px; }
        .es-side-row{ padding:12px 0; border-top:1px solid var(--glass-border); }
        .es-side-row:first-child{ border-top:none; padding-top:0; }
        .es-side-label{ font-size:12.5px; color:var(--text-2); margin-bottom:6px; }
        .es-checklist{ list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:6px; font-size:13.5px; }
        .es-checklist li{ display:flex; align-items:center; gap:8px; color:var(--text-1); }
        .es-checklist li.done{ color:var(--green); }
      `}</style>

      <div className="es-header-row">
        <div>
          <div className="es-eyebrow mono">AI REVIEW BOARD</div>
          <h1 className="es-title">새 분석 시작</h1>
          <p className="es-subtitle">현재 준비 상태에 맞는 분석 방식을 선택하고 필요한 자료를 등록해 주세요.</p>
        </div>
        <button type="button" className="btn-ghost rb-inline-projects" onClick={() => navigate('/projects')}>
          <FolderOpen size={14} /> 내 프로젝트
        </button>
      </div>

      <div className="es-layout">
        <div className="es-progress-list">
          {ENTRY_PROGRESS_STEPS.map((step, i) => (
            <div key={step.key} className={`es-progress-step ${i === 0 ? "active" : "upcoming"}`}>
              <div className="es-progress-dot">{i + 1}</div>
              <div>
                <div className="es-progress-label">{step.label}</div>
                <div className="es-progress-desc">{step.desc}</div>
              </div>
            </div>
          ))}
        </div>

        <div style={{ minWidth: 0 }}>
          <div role="radiogroup" aria-label="분석 방식 선택" className="es-mode-grid">
            {['pre', 'post'].map((key) => (
              <ModeCard key={key} meta={MODE_META[key]} selected={mode === key} onSelect={() => selectMode(key)} />
            ))}
          </div>

          <div className={`card glass es-material-section ${!mode ? "disabled" : ""}`}>
            {!mode ? (
              <div style={{ fontSize: 14, color: "var(--text-2)" }}>분석 방식을 먼저 선택해 주세요.</div>
            ) : (
              <>
                <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 4, display: "flex", alignItems: "center", gap: 6 }}>
                  <Link2 size={14} color="var(--text-2)" /> 공모전 자료 등록
                  <span className="badge purple mono" style={{ marginLeft: 6 }}>필수 입력</span>
                </div>
                <div style={{ fontSize: 13, color: "var(--text-2)", marginBottom: 14 }}>
                  공고문, 평가 기준, 신청서 양식 등 공모전 관련 자료를 등록해 주세요.
                </div>

                {mode === 'post' && (
                  <div style={{ fontSize: 12.5, color: "var(--text-1)", background: "var(--bg-2)", borderRadius: 10, padding: "10px 12px", marginBottom: 14, lineHeight: 1.6 }}>
                    평가받을 기획서·제안서·사업계획서 같은 문서는 다음 단계(기획서 업로드)에서 따로 등록해요. 여기서는 공모전 공고문·평가기준·신청서 양식만 등록하면 돼요.
                  </div>
                )}

                <div className="es-tabs" role="tablist" aria-label="자료 등록 방식">
                  {[['url', 'URL로 가져오기'], ['file', '파일 업로드']].map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      role="tab"
                      aria-selected={criteriaTab === key}
                      tabIndex={criteriaTab === key ? 0 : -1}
                      className={`es-tab ${criteriaTab === key ? "active" : ""}`}
                      onClick={() => setCriteriaTab(key)}
                      onKeyDown={(e) => {
                        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
                          e.preventDefault();
                          setCriteriaTab((prev) => (prev === 'url' ? 'file' : 'url'));
                        }
                      }}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {criteriaTab === 'url' ? (
                  <div style={{ display: "flex", gap: 10 }}>
                    <input
                      value={criteriaUrl}
                      onChange={(e) => setCriteriaUrl(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleFetchCriteriaUrl()}
                      placeholder="공모전 페이지 URL을 입력하세요."
                      disabled={criteriaLoading}
                      style={{ flex: 1, background: "var(--bg-1)", border: "1px solid var(--glass-border)", borderRadius: 10, padding: "11px 14px", color: "var(--text-0)", fontSize: 14 }}
                    />
                    <button type="button" className="btn-ghost" onClick={handleFetchCriteriaUrl} disabled={criteriaLoading || !criteriaUrl.trim()}>
                      {criteriaLoading ? '가져오는 중...' : '가져오기'}
                    </button>
                  </div>
                ) : (
                  <div
                    style={{
                      border: `1.5px dashed ${isCriteriaDragging ? 'var(--purple)' : 'var(--glass-border)'}`,
                      borderRadius: 12, padding: '18px 16px', display: 'flex', alignItems: 'center',
                      justifyContent: 'space-between', background: isCriteriaDragging ? 'var(--purple-dim)' : 'var(--bg-1)',
                    }}
                    {...criteriaDropHandlers}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <Upload size={20} color="var(--purple)" />
                      <div>
                        <div style={{ fontSize: 15, fontWeight: 700 }}>{isCriteriaDragging ? '여기에 놓으세요' : '공고문 · 평가기준 · 신청서 양식'}</div>
                        <div style={{ fontSize: 13, color: 'var(--text-2)' }}>PDF, DOCX, PPTX, HWP, HWPX · 파일당 최대 50MB · 여러 개 선택 가능</div>
                      </div>
                    </div>
                    <button type="button" className="btn-ghost" onClick={() => criteriaFileInputRef.current?.click()}>파일 선택</button>
                    <input
                      ref={criteriaFileInputRef}
                      type="file"
                      accept={ACCEPTED_DOCUMENT_EXTENSIONS.join(',')}
                      multiple
                      style={{ display: 'none' }}
                      onChange={(e) => { addCriteriaFiles(e.target.files); e.target.value = ''; }}
                    />
                  </div>
                )}

                {criteriaError && <p style={{ color: "var(--coral)", fontSize: 13, marginTop: 12 }}>{criteriaError}</p>}

                {documents.filter((d) => d.unsupportedLinks?.length > 0 && !dismissedAlerts.includes(d.id)).map((doc) => (
                  <div key={`alert-${doc.id}`} className="es-alert" style={{ marginTop: 14 }}>
                    <AlertTriangle size={18} color="var(--amber)" style={{ flexShrink: 0, marginTop: 1 }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="es-alert-title">추가 파일 확인이 필요합니다</div>
                      <div className="es-alert-body">
                        공모전 페이지에서 첨부파일을 발견했지만 자동으로 내용을 가져오지 못했습니다.
                      </div>
                      <ul style={{ margin: '8px 0 0', paddingLeft: 16, fontSize: 12.5, color: "var(--text-1)" }}>
                        {doc.unsupportedLinks.map((link, i) => (
                          <li key={i}>{link.file_name || link.url}</li>
                        ))}
                      </ul>
                      <div className="es-alert-actions">
                        <button type="button" className="btn-ghost" onClick={() => { setCriteriaTab('file'); criteriaFileInputRef.current?.click(); }}>
                          파일 업로드
                        </button>
                        <button type="button" className="btn-ghost" onClick={() => setDismissedAlerts((prev) => [...prev, doc.id])}>
                          제외하기
                        </button>
                      </div>
                    </div>
                  </div>
                ))}

                {documents.length > 0 && (
                  <div style={{ marginTop: 14 }}>
                    {documents.map((doc) => (
                      <div key={doc.id} className="es-doc-row">
                        <div style={{ display: 'flex', gap: 10, minWidth: 0, alignItems: 'flex-start' }}>
                          {doc.type === 'url'
                            ? <Link2 size={16} color="var(--text-2)" style={{ marginTop: 2, flexShrink: 0 }} />
                            : <FileText size={16} color="var(--text-2)" style={{ marginTop: 2, flexShrink: 0 }} />}
                          <div style={{ minWidth: 0 }}>
                            <div className="es-doc-name" title={doc.name}>{doc.name}</div>
                            <div className="es-doc-meta">{doc.meta}</div>
                          </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                          {doc.status === 'embedding' && (
                            <ProgressBar percent={doc.progress} label="처리 중" color="linear-gradient(135deg, #7c5cea, #8b6ef0)" trackColor="var(--bg-2)" />
                          )}
                          {doc.status === 'done' && <span className="badge green mono">분석 준비 완료</span>}
                          {doc.status === 'warning' && <span className="badge amber mono">확인 필요</span>}
                          {doc.status === 'error' && <span className="badge coral mono">처리 실패</span>}
                          <button
                            type="button"
                            onClick={() => handleDeleteDoc(doc)}
                            disabled={deletingIds.includes(doc.id)}
                            aria-label={`${doc.name} 삭제`}
                            style={{ background: 'none', border: 'none', padding: 6, cursor: deletingIds.includes(doc.id) ? 'default' : 'pointer', color: 'var(--text-2)', display: 'flex', opacity: deletingIds.includes(doc.id) ? 0.4 : 1 }}
                          >
                            <Trash2 size={15} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>

          {error && <p style={{ color: "var(--coral)", fontSize: 13 }}>{error}</p>}
        </div>

        <aside className="es-side">
          <div className="card glass">
            <div className="es-side-title">현재 분석 설정</div>

            <div className="es-side-row">
              <div className="es-side-label">분석 방식</div>
              <div style={{ fontSize: 14.5, fontWeight: 600 }}>
                {mode ? MODE_META[mode].badge : '분석 방식을 선택해 주세요.'}
              </div>
            </div>

            {materialStatusEntries.length > 0 && (
              <div className="es-side-row">
                <div className="es-side-label">등록된 자료 상태</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {materialStatusEntries.map(([label, count, color]) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13.5 }}>
                      <span style={{ color: 'var(--text-1)' }}>{label}</span>
                      <span className={`badge ${color} mono`}>{count}개</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {mode && expectedResults.length > 0 && (
              <div className="es-side-row">
                <div className="es-side-label">분석 후 다음 순서</div>
                <ul style={{ margin: 0, paddingLeft: 16, fontSize: 13.5, color: "var(--text-1)", lineHeight: 1.8 }}>
                  {expectedResults.map((r) => <li key={r}>{r}</li>)}
                </ul>
              </div>
            )}

            <div className="es-side-row">
              <ul className="es-checklist">
                {checklist.map((item) => (
                  <li key={item.label} className={item.done ? 'done' : ''}>
                    {item.done ? <CheckCircle2 size={14} /> : <Circle size={13} />} {item.label}
                  </li>
                ))}
              </ul>

              <button
                type="button"
                className="btn-primary"
                style={{ marginTop: 14, width: "100%", opacity: canStart ? 1 : 0.4, display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}
                disabled={!canStart || loading}
                onClick={() => onEnter(mode)}
              >
                {loading
                  ? "분석을 시작하고 있습니다..."
                  : mode === 'post'
                    ? "문서 평가 시작하기"
                    : "공모전 분석 시작하기"} <ArrowRight size={15} />
              </button>
              {guide && !loading && (
                <p style={{ fontSize: 12.5, color: "var(--text-2)", marginTop: 8, textAlign: "center" }}>{guide}</p>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

const DETAIL_TABS = [
  { key: "strategy", label: "지원 전략" },
  { key: "schedule", label: "일정·신청" },
  { key: "eligibility", label: "신청 조건" },
  { key: "benefits", label: "선정 혜택" },
  { key: "caution", label: "유의사항" },
];

// 가은/Claude(2026-07-24, 요청: 공모전 분석 결과 화면 개편) — 상단 가로형 메타 카드.
// 실제 API가 내려주지 않는 값(분석 완료 시각·소요 시간·모델명)은 절대 지어내지 않고,
// 실제 데이터로 계산 가능한 항목만 items로 넘겨 받는다 — items가 비면 행 전체를 숨긴다.
function MetaSummaryRow({ items }) {
  if (items.length === 0) return null;
  return (
    <div className="cas-meta-row">
      {items.map((item) => (
        <div key={item.label} className="cas-meta-item">
          <item.icon size={14} color="var(--text-2)" />
          <div>
            <div className="cas-meta-label">{item.label}</div>
            <div className="cas-meta-value">{item.value}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function AnalysisSummaryCard({ summary, onExpand }) {
  return (
    <div className="card glass cas-section">
      <div className="cas-section-head">
        <div className="cas-card-title"><Target size={15} color="var(--purple)" /> AI 분석 요약</div>
        {summary && (
          <button type="button" className="btn-ghost cas-detail-btn" onClick={onExpand}>
            자세히 보기 <ChevronRight size={13} />
          </button>
        )}
      </div>
      {summary ? <p className="cas-summary-text">{summary}</p> : <div className="cas-empty">핵심 과제를 판단할 근거가 부족해요.</div>}
    </div>
  );
}

function EvaluationCriteriaSummary({ groups, expanded, onToggle }) {
  const allItems = groups.flatMap((g) => g.items);
  const { itemCount, totalScore } = summarizeCriteria(groups);

  return (
    <div className="card glass cas-section">
      <div className="cas-section-head">
        <div>
          <div className="cas-card-title"><Award size={15} color="var(--coral)" /> 평가 기준 요약</div>
          {itemCount > 0 && (
            <div className="cas-card-subtitle">
              {itemCount}개 항목{totalScore != null ? `, 총 ${totalScore}점 만점으로 평가됩니다.` : "으로 평가됩니다."}
            </div>
          )}
        </div>
        {itemCount > 0 && (
          <button type="button" className="btn-ghost cas-detail-btn" onClick={onToggle}>
            {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />} {expanded ? "접기" : "전체 평가 기준 보기"}
          </button>
        )}
      </div>

      {itemCount === 0 && <div className="cas-empty">등록한 공고문에서 확인된 평가 기준이 없어요.</div>}

      {itemCount > 0 && !expanded && (
        <div className="cas-criteria-chips">
          {allItems.slice(0, 4).map((item, i) => (
            <div key={i} className="cas-criteria-chip">
              <span>{item.name}</span>
              {item.score != null && <span className="mono cas-criteria-score">{item.score}점</span>}
            </div>
          ))}
          {allItems.length > 4 && <div className="cas-criteria-more">+{allItems.length - 4}개 항목</div>}
        </div>
      )}

      {itemCount > 0 && expanded && (
        <div className="cas-criteria-groups">
          {groups.map((group) => (
            <div key={group.groupName} className="cas-criteria-group">
              <div className="cas-criteria-group-name">
                {group.groupName}
                {group.totalScore != null && <span className="mono cas-criteria-score"> · {group.totalScore}점</span>}
              </div>
              <div className="cas-criteria-chips">
                {group.items.map((item, i) => (
                  <div key={i} className="cas-criteria-chip">
                    <span>{item.name}</span>
                    {item.score != null && <span className="mono cas-criteria-score">{item.score}점</span>}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SimilarCaseSection({ hasData, works, expanded, onToggle, onWorkClick, loadingWork }) {
  if (!hasData || works.length === 0) {
    return (
      <div className="card glass cas-section">
        <div className="cas-card-title"><TrendingUp size={15} color="var(--amber)" /> 수상작·유사 사례 분석</div>
        <div className="cas-empty">
          {hasData ? "일치하는 유사 사례를 찾지 못했어요." : "수상작 자료 미확보 — 유사 공모전 사례 기반 분석은 아직 지원하지 않아요."}
        </div>
      </div>
    );
  }

  const visible = expanded ? works : works.slice(0, 3);
  const restCount = works.length - visible.length;

  return (
    <div className="card glass cas-section">
      <div className="cas-section-head">
        <div>
          <div className="cas-card-title"><TrendingUp size={15} color="var(--amber)" /> 수상작·유사 사례 분석</div>
          <div className="cas-card-subtitle">최근 수상작 및 유사 사례 {works.length}건을 분석했습니다.</div>
        </div>
        <button type="button" className="btn-ghost cas-detail-btn" onClick={onToggle}>
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />} {expanded ? "접기" : "자세히 보기"}
        </button>
      </div>

      <div className="cas-case-grid">
        {visible.map((work, i) => {
          const clickable = !!work.contest_title && !loadingWork;
          const Tag = clickable ? "button" : "div";
          return (
            <Tag
              key={i}
              type={clickable ? "button" : undefined}
              className="card glass cas-case-card"
              onClick={clickable ? () => onWorkClick(work) : undefined}
            >
              {work.selection_status && (
                <span className={`badge ${work.selection_status === "winner" ? "green" : "amber"} mono`}>
                  {work.selection_status === "winner" ? (work.award_grade || "수상") : "후보"}
                </span>
              )}
              <div className="cas-case-title">{work.title}</div>
              {(work.contest_title || work.source_org) && (
                <div className="cas-case-source">{[work.contest_title, work.source_org].filter(Boolean).join(" · ")}</div>
              )}
            </Tag>
          );
        })}
        {!expanded && restCount > 0 && (
          <button type="button" className="card glass cas-case-card cas-case-more" onClick={onToggle}>
            +{restCount}건 더 보기
          </button>
        )}
      </div>
    </div>
  );
}

function ContestDetailTabs({ activeTab, onTabChange, facts, strategy, formAnalysis }) {
  function handleTabKeyDown(e) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const idx = DETAIL_TABS.findIndex((t) => t.key === activeTab);
    const dir = e.key === "ArrowRight" ? 1 : -1;
    onTabChange(DETAIL_TABS[(idx + dir + DETAIL_TABS.length) % DETAIL_TABS.length].key);
  }

  const eligibilityItems = [
    ...(facts?.eligibility || []),
    ...(facts?.submission_requirements || []),
    ...(facts?.application_review_conditions || []),
  ];
  const disqualificationRules = facts?.disqualification_rules || [];
  const riskFlags = strategy?.risk_flags || [];
  const formatScheduleDate = (value, weekday, includeYear = true) => {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value || "");
    if (!match) return value || "";
    const [, year, month, day] = match;
    return `${includeYear ? `${Number(year)}년 ` : ""}${Number(month)}월 ${Number(day)}일${weekday ? `(${weekday})` : ""}`;
  };
  const formatScheduleItem = (item) => {
    const start = formatScheduleDate(item.start_date, item.start_weekday);
    const sameYear = item.end_date?.slice(0, 4) === item.start_date?.slice(0, 4);
    const end = item.end_date
      ? ` ~ ${formatScheduleDate(item.end_date, item.end_weekday, !sameYear)}`
      : "";
    return [item.event_label, `${start}${end}`, item.method].filter(Boolean).join(" · ");
  };

  return (
    <div className="cas-tabs-wrap">
      <div className="es-tabs cas-detail-tabs" role="tablist" aria-label="핵심 요구사항 상세">
        {DETAIL_TABS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.key}
            tabIndex={activeTab === tab.key ? 0 : -1}
            className={`es-tab ${activeTab === tab.key ? "active" : ""}`}
            onClick={() => onTabChange(tab.key)}
            onKeyDown={handleTabKeyDown}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="cas-tab-panel" role="tabpanel">
        {activeTab === "strategy" && (
          (strategy?.winning_points?.length || 0) + (strategy?.recommended_direction?.length || 0) > 0 ? (
            <div className="cas-numbered-list">
              {strategy?.winning_points?.length > 0 && <div className="cas-subheading">차별화 포인트</div>}
              {strategy?.winning_points?.map((v, i) => (
                <div key={`w-${i}`} className="cas-numbered-item"><span className="cas-num">{i + 1}</span><span>{v}</span></div>
              ))}
              {strategy?.recommended_direction?.length > 0 && <div className="cas-subheading">추천 방향</div>}
              {strategy?.recommended_direction?.map((v, i) => (
                <div key={`r-${i}`} className="cas-numbered-item"><span className="cas-num">{i + 1}</span><span>{v}</span></div>
              ))}
            </div>
          ) : <div className="cas-empty">제안할 전략을 판단할 근거가 부족해요.</div>
        )}

        {activeTab === "schedule" && (
          <>
            <div className="cas-subheading" style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Calendar size={13} color="var(--purple)" /> 주요 일정
            </div>
            {(facts?.schedule_items?.length || 0) > 0 ? (
              <div className="cas-timeline">
                {facts.schedule_items.map((item, i) => (
                  <div key={`${item.event_label}-${item.start_date}-${i}`} className="cas-timeline-row">
                    <span className="cas-timeline-dot" />
                    <span>{formatScheduleItem(item)}</span>
                  </div>
                ))}
              </div>
            ) : (facts?.key_dates?.length || 0) > 0 ? (
              <div className="cas-timeline">
                {facts.key_dates.map((date, i) => (
                  <div key={i} className="cas-timeline-row">
                    <span className="cas-timeline-dot" />
                    <span>{date}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className={facts?.deadline && facts.deadline !== "미공개" ? "cas-plain-text" : "cas-empty"}>
                제출 마감: {facts?.deadline || "미공개"}
              </div>
            )}
          </>
        )}

        {activeTab === "eligibility" && (
          <>
            {eligibilityItems.length > 0 ? (
              <ul className="cas-plain-list">{eligibilityItems.map((v, i) => <li key={i}>{v}</li>)}</ul>
            ) : <div className="cas-empty">공고문에서 확인된 신청·심사 조건이 없어요.</div>}

            {formAnalysis?.has_application_form && formAnalysis.items.length > 0 && (
              <div className="cas-form-items">
                <div className="cas-subheading" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <FileText size={13} color="var(--purple)" /> 신청양식에서 써야 할 항목
                </div>
                <div className="cas-form-note">
                  등록한 신청양식에서 찾은 기입란이에요. 여러 부문(도시/기업 등)이 한 문서에 있으면 관련 없는 항목도 섞여 보일 수 있어요.
                </div>
                <ul className="cas-plain-list">
                  {formAnalysis.items.map((item, i) => (
                    <li key={i}>
                      <strong>{item.field_name}</strong>
                      {item.char_limit != null && <span className="mono cas-form-limit"> ({item.char_limit}자 이내)</span>}
                      {item.description && <div className="cas-form-desc">{item.description}</div>}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {activeTab === "benefits" && (
          (facts?.selection_benefits?.length || 0) > 0 ? (
            <div className="cas-benefit-grid">
              {facts.selection_benefits.map((v, i) => (
                <div key={i} className="card glass cas-benefit-card">
                  <Gift size={15} color="var(--purple)" />
                  <div>{v}</div>
                </div>
              ))}
            </div>
          ) : <div className="cas-empty">공고문에서 확인된 선정 혜택이 없어요.</div>
        )}

        {activeTab === "caution" && (
          <>
            {disqualificationRules.length > 0 ? (
              <div className="cas-warning-box">
                <AlertOctagon size={16} color="var(--amber)" style={{ flexShrink: 0, marginTop: 1 }} />
                <div>
                  <div className="cas-warning-title">수상 취소 조건</div>
                  <ul className="cas-plain-list">{disqualificationRules.map((v, i) => <li key={i}>{v}</li>)}</ul>
                </div>
              </div>
            ) : <div className="cas-empty">공고문에서 확인된 수상 취소 조건이 없어요.</div>}

            {riskFlags.length > 0 && (
              <div className="cas-risk-list">
                <div className="cas-subheading">주의 리스크</div>
                <ul className="cas-plain-list">{riskFlags.map((v, i) => <li key={i}>{v}</li>)}</ul>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function CoreRequirementsSection({ open, onToggle, ...tabProps }) {
  return (
    <div className="card glass cas-section">
      <div className="cas-section-head">
        <div>
          <div className="cas-card-title"><ShieldCheck size={15} color="var(--green)" /> 핵심 요구사항 정리</div>
          <div className="cas-card-subtitle">공모전에서 요구하는 핵심 조건과 제출물, 일정, 신청 대상 및 유의사항을 정리했습니다.</div>
        </div>
        <button type="button" className="btn-ghost cas-detail-btn" onClick={onToggle}>
          {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />} {open ? "접기" : "자세히 보기"}
        </button>
      </div>
      {open && <ContestDetailTabs {...tabProps} />}
    </div>
  );
}

function EvidenceSection({ evidence, sourceDocs, expanded, onToggle }) {
  if (evidence.length === 0) {
    return (
      <div className="card glass cas-section">
        <div className="cas-card-title"><Quote size={15} color="var(--purple)" /> 분석 근거</div>
        <div className="cas-empty">현재 연결된 분석 근거가 없습니다.</div>
      </div>
    );
  }

  const visible = expanded ? evidence : evidence.slice(0, 3);

  return (
    <div className="card glass cas-section">
      <div className="cas-section-head">
        <div className="cas-card-title"><Quote size={15} color="var(--purple)" /> 분석 근거</div>
        {evidence.length > 3 && (
          <button type="button" className="btn-ghost cas-detail-btn" onClick={onToggle}>
            {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />} {expanded ? "접기" : "전체 근거 보기"}
          </button>
        )}
      </div>
      <div className="cas-evidence-list">
        {visible.map((e, i) => (
          <div key={i} className="cas-evidence-row">
            <div className="cas-evidence-head">
              <span className={`badge ${e.source_type === "announcement" ? "green" : "purple"} mono`}>
                {e.source_type === "announcement" ? "공고문 근거" : "AI 추론"}
              </span>
            </div>
            <div className="cas-evidence-claim">{e.claim}</div>
            <div className="cas-evidence-meta">
              {e.location || (e.source_type === "announcement" ? "위치 미상" : "추론")} · {CONFIDENCE_LABEL[e.confidence] || e.confidence}
            </div>
          </div>
        ))}
      </div>
      {sourceDocs.length > 0 && <div className="cas-evidence-sources">참고 문서: {sourceDocs.join(", ")}</div>}
    </div>
  );
}

function NextStepCard({ nextLabel, nextDesc, ctaLabel, onAdvance, advancing }) {
  return (
    <div className="card glass">
      {nextLabel && <div className="cas-next-title">{nextLabel}</div>}
      {nextDesc && <div className="cas-next-desc">{nextDesc}</div>}
      <button type="button" className="btn-primary cas-next-btn" onClick={onAdvance} disabled={advancing}>
        {advancing ? "이동하는 중..." : ctaLabel} <ArrowRight size={14} />
      </button>
    </div>
  );
}

// 가은/Claude(2026-07-24, 요청: 공모전 분석 결과 화면 개편) — 오른쪽 "분석 진행 상태"
// 패널. 왼쪽 Shell navrail과 같은 FLOW_BY_MODE/STAGE_LABELS를 그대로 참조해 두 표시가
// 어긋나지 않게 한다. 여기 진행 단계 목록은 정보용이라 클릭 이동은 왼쪽 단계 메뉴에만 둔다.
function AnalysisStatusPanel({ mode, active, statRows, nextLabel, nextDesc, ctaLabel, onAdvance, advancing }) {
  const flow = mode ? FLOW_BY_MODE[mode] : ["entry"];
  const activeIdx = flow.indexOf(active);

  return (
    <aside className="cas-aside">
      <div className="card glass">
        <div className="cas-aside-title">분석 진행 상태</div>
        <div className="cas-status-list">
          {flow.map((k, idx) => {
            const state = idx < activeIdx ? "done" : idx === activeIdx ? "current" : "upcoming";
            return (
              <div key={k} className="cas-status-row">
                <span>{STAGE_LABELS[k]}</span>
                <span className={`badge ${state === "done" ? "green" : state === "current" ? "purple" : "grey"} mono`}>
                  {state === "done" ? "완료" : state === "current" ? "진행 중" : "대기"}
                </span>
              </div>
            );
          })}
        </div>

        {statRows.length > 0 && (
          <>
            <div className="cas-aside-title" style={{ marginTop: 18 }}>현재 분석 요약</div>
            <div className="cas-stat-list">
              {statRows.map((row) => (
                <div key={row.label} className="cas-stat-row">
                  <div className="cas-stat-label">{row.label}</div>
                  <div className="cas-stat-value">{row.value}</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <NextStepCard nextLabel={nextLabel} nextDesc={nextDesc} ctaLabel={ctaLabel} onAdvance={onAdvance} advancing={advancing} />
    </aside>
  );
}

/* ---------------- 2. 공모전 분석 결과 (공통) ----------------
 * 가은/Claude(2026-07-21): 실측 제보 두 건에 대한 대응.
 * 1) "URL 넣고 분석 시작 눌렀는데 예시 카드만 나왔다" — 이 화면이 늘 고정 예시만
 *    보여줬던 문제. 이제 실제 수집한 공고문을 근거로 백엔드가 LLM 1회 호출로 뽑은
 *    official_facts(공고문에 실제 있는 사실)/strategic_analysis(AI 추론)를 보여준다.
 * 2) 팀 UX 스펙(2026-07-21) 그대로: 첫 화면은 4개 카드(핵심 과제/평가 갈리는 지점/
 *    반드시 지킬 조건/수상작 경향)만, 아래에 접을 수 있는 상세 분석(추천 전략/주의
 *    리스크/출처)을 둔다. 수상작·유사사례 경향은 kyh님의 소통혁신24 수상작 아카이브
 *    (contest_works)가 생기기 전까지는 데이터 소스가 없어 "자료 미확보"로 고정
 *    표시했었다 — 이제 실제 매칭 결과가 있으면 보여주고, 없을 때만 그 문구를 쓴다.
 *
 * 가은/Claude(2026-07-24, 요청: 공모전 분석 결과 화면 개편) — 팀 레퍼런스(왼쪽 단계/
 * 가운데 분석 결과/오른쪽 진행 상태) 3열 대시보드로 재구성. 데이터 소스(useState·useEffect·
 * API 호출·handleSimilarWorkClick 등)는 전혀 바꾸지 않았고, 그 아래 표시 방식과 위 보조
 * 컴포넌트들만 새로 짰다. 레퍼런스 화면 속 문구·숫자·모델명은 목업이라 그대로 옮기지
 * 않고, 실제 API 응답에 있는 값만 채운다 — 값이 없으면 해당 UI를 숨긴다.
 */
function AnalysisScreen({ mode, onNext, onBack, projectId }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [analysis, setAnalysis] = useState(null);
  // 가은/Claude(2026-07-22, 요청: 신청양식 항목 패널 분리) — "평가에서 갈리는 지점"
  // (심사 기준·배점)과 "신청양식에서 써야 할 항목"(실제 빈칸)은 성격이 다른 정보라 섞이면
  // 헷갈린다(실측: 도시 부문·기업 부문 두 신청서가 한 파일에 있어 평가기준 10개가 한
  // 리스트에 뒤섞여 나온 사례). getApplicationFormAnalysis는 이미 아이디어 회의 시작 시
  // 내부적으로 쓰던 것과 같은 함수 — 여기서는 화면에 직접 보여주는 용도로 별도 호출한다.
  const [formAnalysis, setFormAnalysis] = useState(null);
  // 가은/Claude(2026-07-23, 요청: 로딩 프로그레스바) — getAnnouncementAnalysis는 단발
  // LLM 호출(+누락 감지 시 최대 1회 재검증)이라 실제 진행률을 알 방법이 없다. 대기
  // 시간에 막대가 멈춰 보이지 않도록 서서히 채우다가, 완료되는 순간에만 100%로 마무리한다.
  const [progressPercent, setProgressPercent] = useState(8);
  const [detailOpen, setDetailOpen] = useState(false);
  const [ideaTopic, setIdeaTopic] = useState(null);
  const [workDetail, setWorkDetail] = useState(null); // { contestTitle, works } | null
  const [workDetailLoading, setWorkDetailLoading] = useState(false);
  const [panelWidth, setPanelWidth] = useState(460);
  const resizingRef = useRef(false);
  // 가은/Claude(2026-07-24, 요청: 공모전 분석 결과 화면 개편) — 새 카드형 레이아웃의
  // 펼침/탭 상태. 전부 표시용 UI 상태라 API 재호출이나 데이터 변형을 일으키지 않는다.
  const [detailTab, setDetailTab] = useState("strategy");
  const [criteriaExpanded, setCriteriaExpanded] = useState(false);
  const [similarExpanded, setSimilarExpanded] = useState(false);
  const [evidenceExpanded, setEvidenceExpanded] = useState(false);
  const [advancing, setAdvancing] = useState(false);

  function openDetail(tab) {
    setDetailTab(tab);
    setDetailOpen(true);
  }

  function handleAdvance() {
    if (advancing) return;
    setAdvancing(true);
    onNext();
  }

  // 가은/Claude(2026-07-21): 실측 요청 — 수상작 상세 패널을 마우스로 드래그해 너비를
  // 조절할 수 있게. 패널이 화면 오른쪽에 고정돼 있어 왼쪽 가장자리를 끌면 그 지점부터
  // 화면 끝까지가 새 너비가 된다.
  useEffect(() => {
    function handleMouseMove(e) {
      if (!resizingRef.current) return;
      const next = window.innerWidth - e.clientX;
      setPanelWidth(Math.min(Math.max(next, 340), Math.min(760, window.innerWidth - 80)));
    }
    function handleMouseUp() {
      if (!resizingRef.current) return;
      resizingRef.current = false;
      document.body.style.userSelect = '';
    }
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  useEffect(() => {
    if (!projectId) {
      setLoading(false);
      setAnalysis({ has_announcement: false });
      setFormAnalysis(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError('');
    setProgressPercent(8);
    const progressTimer = setInterval(() => {
      setProgressPercent((prev) => Math.min(92, prev + 2));
    }, 500);
    getAnnouncementAnalysis(projectId)
      .then((data) => { if (!cancelled) setAnalysis(data); })
      .catch((err) => { if (!cancelled) setError(err.message); })
      .finally(() => {
        clearInterval(progressTimer);
        if (!cancelled) { setProgressPercent(100); setLoading(false); }
      });
    // 신청양식 항목은 "선택 정보"라 실패해도 공고문 분석 자체를 막지 않는다 — 조용히
    // null로 두면 아래 카드가 그냥 안 나타난다.
    getApplicationFormAnalysis(projectId)
      .then((data) => { if (!cancelled) setFormAnalysis(data); })
      .catch(() => { if (!cancelled) setFormAnalysis(null); });
    return () => { cancelled = true; clearInterval(progressTimer); };
  }, [projectId]);

  // 가은/Claude(2026-07-21): 실측 요청 — "내 프로젝트"에서 아이디어 프로젝트를 불러오면
  // 확정한 주제가 어디에도 안 보였다. 프로젝트 설명에 IDEA_PROJECT_MARKER가 있으면
  // "작성 전" 흐름에서 주제를 확정한 프로젝트로 보고 제목을 주제로 표시, 없으면 "미정".
  useEffect(() => {
    if (!projectId) {
      setIdeaTopic(null);
      return;
    }
    let cancelled = false;
    getProject(projectId)
      .then((project) => {
        if (cancelled) return;
        setIdeaTopic(project.description === IDEA_PROJECT_MARKER ? project.title : null);
      })
      .catch(() => { if (!cancelled) setIdeaTopic(null); });
    return () => { cancelled = true; };
  }, [projectId]);

  // 가은/Claude(2026-07-21): 실측 요청 — "수상작·유사사례 경향" 항목을 클릭하면 같은
  // 공모전의 다른 수상작/후보작을 옆 패널로 보여준다. "볼 게 있을 때만" 열려야 하므로
  // (자기 자신 하나뿐이고 이미지도 없으면 클릭해도 반응 없음) 먼저 조회하고 나서 연다.
  async function handleSimilarWorkClick(work) {
    if (!work.contest_title || workDetailLoading) return;
    setWorkDetailLoading(true);
    try {
      const data = await getContestWorksByTitle(work.contest_title);
      const hasMore = data.works.length > 1 || data.works.some((w) => w.images.length > 0 || w.ocr_text);
      if (hasMore) setWorkDetail({ contestTitle: data.contest_title, works: data.works });
    } catch {
      // 실측 지침: 데이터가 없거나(조회 실패 포함) 볼 게 없으면 그냥 아무 반응 없이 둔다.
    } finally {
      setWorkDetailLoading(false);
    }
  }

  if (loading) {
    return (
      <div style={{ maxWidth: 820 }}>
        <div className="badge purple mono">공고문 분석 중</div>
        <h1 style={{ fontSize: 26, fontWeight: 700, margin: "12px 0 24px" }}>등록한 공고문을 읽고 있어요...</h1>
        <div className="card glass" style={{ padding: 20 }}>
          <ProgressBar
            percent={progressPercent}
            label={`${Math.round(progressPercent)}%`}
            color="linear-gradient(135deg, #7c5cea, #8b6ef0)"
            trackColor="var(--bg-2)"
            fill
            height={8}
          />
          <div style={{ color: "var(--text-2)", fontSize: 13, marginTop: 10 }}>잠시만 기다려주세요.</div>
        </div>
      </div>
    );
  }

  const hasAnnouncement = !!analysis?.has_announcement;
  const facts = analysis?.official_facts;
  const strategy = analysis?.strategic_analysis;
  const evidence = analysis?.evidence || [];
  // 가은/Claude(2026-07-21): kyh님이 크롤링+분류한 소통혁신24 수상작 아카이브
  // (contest_works)가 생겨서 실제 유사사례를 보여줄 수 있게 됐다 — 백엔드가 매칭을
  // 못 찾으면(팀 공유 DB에 아직 데이터가 안 올라왔거나 이 카테고리에 사례가 없으면)
  // has_similar_case_data가 false로 오므로 그때는 기존 "미확보" 문구를 그대로 보여준다.
  const similarWorks = analysis?.similar_works || [];
  const criteriaGroups = parseEvaluationCriteria(facts?.evaluation_criteria);
  const criteriaSummary = summarizeCriteria(criteriaGroups);
  const sourceDocCount = analysis?.source_document_names?.length || 0;

  const nextStageKey = (() => {
    const seq = (mode && FLOW_BY_MODE[mode]) || [];
    const i = seq.indexOf("analysis");
    return i >= 0 && i < seq.length - 1 ? seq[i + 1] : null;
  })();
  const nextLabel = nextStageKey ? STAGE_LABELS[nextStageKey] : null;
  const nextDesc = nextStageKey ? STAGE_DESCRIPTIONS[nextStageKey] : null;
  const ctaLabel = mode === "pre" ? "AI 위원과 주제 확정 시작" : "기획서 업로드하기";

  const metaItems = [{ icon: Sparkles, label: "아이디어 주제", value: ideaTopic || "미정" }];
  if (sourceDocCount > 0) metaItems.push({ icon: FileStack, label: "분석에 사용된 자료", value: `${sourceDocCount}개` });
  if (hasAnnouncement) metaItems.push({ icon: CheckCircle2, label: "분석 상태", value: "완료" });

  const statRows = [];
  if (hasAnnouncement) statRows.push({ label: "분석 범위", value: "공모전 공고 및 평가 기준" });
  if (criteriaSummary.itemCount > 0) {
    statRows.push({
      label: "평가 항목",
      value: criteriaSummary.totalScore != null
        ? `${criteriaSummary.itemCount}개 항목 · ${criteriaSummary.totalScore}점`
        : `${criteriaSummary.itemCount}개 항목`,
    });
  }
  if (sourceDocCount > 0) statRows.push({ label: "분석 근거 자료", value: `${sourceDocCount}개 문서` });
  if (analysis?.has_similar_case_data && similarWorks.length > 0) {
    statRows.push({ label: "유사 사례", value: `${similarWorks.length}개 분석` });
  }

  return (
    <>
    <div className="cas-wrap">
      <style>{`
        .cas-wrap{ max-width:1320px; }
        .cas-title-row{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:10px 0 6px; }
        .cas-title{ font-size:28px; font-weight:700; margin:0; }
        .cas-subtitle{ font-size:14.5px; color:var(--text-2); margin:0 0 18px; }

        .cas-meta-row{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:18px; }
        .cas-meta-item{ display:flex; align-items:center; gap:8px; background:var(--bg-1); border:1px solid var(--glass-border); border-radius:12px; padding:10px 14px; flex:0 1 auto; }
        .cas-meta-label{ font-size:12px; color:var(--text-2); }
        .cas-meta-value{ font-size:14.5px; font-weight:700; }
        @media (max-width:640px){ .cas-meta-item{ flex:1 1 45%; } }

        .cas-layout{ display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:24px; align-items:start; }
        @media (max-width:1180px){ .cas-layout{ grid-template-columns:minmax(0,1fr); } .cas-aside{ position:static !important; } }

        .cas-section{ display:flex; flex-direction:column; gap:10px; }
        .cas-section-head{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
        .cas-card-title{ display:flex; align-items:center; gap:7px; font-size:15.5px; font-weight:700; }
        .cas-card-subtitle{ font-size:13px; color:var(--text-2); margin-top:2px; line-height:1.5; }
        .cas-detail-btn{ display:flex; align-items:center; gap:4px; padding:7px 14px; font-size:13.5px; flex-shrink:0; white-space:nowrap; }
        .cas-empty{ font-size:13.5px; color:var(--text-2); }
        .cas-summary-text{ font-size:14px; color:var(--text-1); line-height:1.7; margin:0; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }

        .cas-criteria-chips{ display:flex; flex-wrap:wrap; gap:8px; }
        .cas-criteria-chip{ display:flex; align-items:center; gap:8px; background:var(--bg-2); border-radius:10px; padding:8px 12px; font-size:13.5px; }
        .cas-criteria-score{ color:var(--purple); font-size:12.5px; }
        .cas-criteria-more{ display:flex; align-items:center; font-size:13.5px; color:var(--text-2); padding:8px 4px; }
        .cas-criteria-groups{ display:flex; flex-direction:column; gap:14px; }
        .cas-criteria-group-name{ font-size:13.5px; font-weight:700; margin-bottom:6px; }

        .cas-case-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
        @media (max-width:900px){ .cas-case-grid{ grid-template-columns:repeat(2,1fr); } }
        @media (max-width:560px){ .cas-case-grid{ grid-template-columns:1fr; } }
        .cas-case-card{ min-height:96px; display:flex; flex-direction:column; gap:8px; text-align:left; font:inherit; color:inherit; }
        .cas-case-title{ font-size:14px; font-weight:600; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; line-height:1.5; }
        .cas-case-source{ font-size:12px; color:var(--text-2); }
        .cas-case-more{ align-items:center; justify-content:center; color:var(--purple); font-size:13.5px; font-weight:600; cursor:pointer; border:1px dashed var(--glass-border); }

        .cas-tabs-wrap{ margin-top:6px; }
        .cas-detail-tabs{ margin-bottom:14px; flex-wrap:wrap; }
        .cas-tab-panel{ font-size:13.5px; color:var(--text-1); line-height:1.8; }
        .cas-subheading{ font-size:13px; font-weight:700; color:var(--text-1); margin:10px 0 6px; }
        .cas-subheading:first-child{ margin-top:0; }
        .cas-numbered-item{ display:flex; gap:8px; margin-bottom:6px; }
        .cas-num{ flex-shrink:0; width:18px; height:18px; margin-top:1px; border-radius:999px; background:var(--purple-dim); color:var(--purple); font-size:10.5px; font-weight:700; display:flex; align-items:center; justify-content:center; }
        .cas-plain-list{ margin:0; padding-left:16px; }
        .cas-plain-list li{ margin-bottom:4px; }
        .cas-plain-text{ color:var(--text-1); }
        .cas-timeline{ display:flex; flex-direction:column; gap:10px; }
        .cas-timeline-row{ display:flex; align-items:center; gap:10px; }
        .cas-timeline-dot{ width:8px; height:8px; border-radius:999px; background:var(--purple); flex-shrink:0; }
        .cas-benefit-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }
        @media (max-width:560px){ .cas-benefit-grid{ grid-template-columns:1fr; } }
        .cas-benefit-card{ display:flex; align-items:flex-start; gap:8px; font-size:13.5px; }
        .cas-warning-box{ display:flex; gap:10px; background:var(--amber-dim); border:1px solid rgba(184,131,11,0.25); border-radius:12px; padding:14px 16px; }
        .cas-warning-title{ font-weight:700; color:var(--amber); margin-bottom:6px; font-size:13.5px; }
        .cas-risk-list{ margin-top:14px; }
        .cas-form-items{ margin-top:14px; padding-top:14px; border-top:1px solid var(--glass-border); }
        .cas-form-note{ font-size:12.5px; color:var(--text-2); margin-bottom:8px; }
        .cas-form-limit{ color:var(--text-2); font-size:12px; }
        .cas-form-desc{ font-size:12.5px; color:var(--text-2); }

        .cas-evidence-list{ display:flex; flex-direction:column; }
        .cas-evidence-row{ padding:10px 0; border-top:1px solid var(--glass-border); }
        .cas-evidence-row:first-child{ border-top:none; padding-top:0; }
        .cas-evidence-head{ display:flex; align-items:center; gap:6px; margin-bottom:4px; }
        .cas-evidence-claim{ font-size:13.5px; color:var(--text-1); line-height:1.6; }
        .cas-evidence-meta{ font-size:12px; color:var(--text-2); margin-top:2px; }
        .cas-evidence-sources{ font-size:12.5px; color:var(--text-2); margin-top:10px; padding-top:10px; border-top:1px solid var(--glass-border); }

        .cas-aside{ position:sticky; top:24px; display:flex; flex-direction:column; gap:16px; }
        .cas-aside-title{ font-size:14.5px; font-weight:700; margin-bottom:10px; }
        .cas-status-list{ display:flex; flex-direction:column; gap:8px; }
        .cas-status-row{ display:flex; justify-content:space-between; align-items:center; font-size:13.5px; }
        .cas-stat-list{ display:flex; flex-direction:column; gap:8px; }
        .cas-stat-row{ display:flex; justify-content:space-between; gap:10px; font-size:13px; padding-top:8px; border-top:1px solid var(--glass-border); }
        .cas-stat-row:first-child{ border-top:none; padding-top:0; }
        .cas-stat-label{ color:var(--text-2); }
        .cas-stat-value{ font-weight:600; color:var(--text-1); text-align:right; }
        .cas-next-title{ font-size:14.5px; font-weight:700; }
        .cas-next-desc{ font-size:13px; color:var(--text-2); line-height:1.6; margin:6px 0 14px; }
        .cas-next-btn{ width:100%; display:flex; align-items:center; justify-content:center; gap:8px; }
      `}</style>

      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <button type="button" className="rb-back-button" onClick={onBack} aria-label="이전 화면으로 이동">
          {'←'}
        </button>
      </div>
      <div className="cas-title-row">
        <h1 className="cas-title">
          {hasAnnouncement ? (analysis?.announcement_title || "등록한 공고문을 확인하세요") : "등록된 공고문이 없어요"}
        </h1>
        <span className="badge purple mono">{(mode && MODE_META[mode]?.badge) || "공모전 분석"}</span>
        {!hasAnnouncement && <span className="badge amber mono">공고문 미등록</span>}
      </div>
      {STAGE_DESCRIPTIONS.analysis && <p className="cas-subtitle">{STAGE_DESCRIPTIONS.analysis}</p>}

      {error && <p style={{ color: "var(--coral)", fontSize: 13, marginBottom: 16 }}>{error}</p>}

      <MetaSummaryRow items={metaItems} />

      <div className="cas-layout">
        <div style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
          {hasAnnouncement ? (
            <>
              <AnalysisSummaryCard summary={strategy?.core_intent} onExpand={() => openDetail("strategy")} />
              <EvaluationCriteriaSummary
                groups={criteriaGroups}
                expanded={criteriaExpanded}
                onToggle={() => setCriteriaExpanded((v) => !v)}
              />
              <SimilarCaseSection
                hasData={!!analysis?.has_similar_case_data}
                works={similarWorks}
                expanded={similarExpanded}
                onToggle={() => setSimilarExpanded((v) => !v)}
                onWorkClick={handleSimilarWorkClick}
                loadingWork={workDetailLoading}
              />
              <CoreRequirementsSection
                open={detailOpen}
                onToggle={() => setDetailOpen((v) => !v)}
                activeTab={detailTab}
                onTabChange={setDetailTab}
                facts={facts}
                strategy={strategy}
                formAnalysis={formAnalysis}
              />
              <EvidenceSection
                evidence={evidence}
                sourceDocs={analysis?.source_document_names || []}
                expanded={evidenceExpanded}
                onToggle={() => setEvidenceExpanded((v) => !v)}
              />
            </>
          ) : (
            <div className="card glass" style={{ borderColor: "var(--amber-dim)" }}>
              <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                <AlertCircle size={15} color="var(--amber)" style={{ marginTop: 2, flexShrink: 0 }} />
                <div style={{ fontSize: 12.5, color: "var(--text-1)", lineHeight: 1.6 }}>
                  이전 화면에서 공고 URL이나 파일을 등록하지 않았어요 — 공식 심사기준 없이 일반적인 평가 관점으로 진행합니다.
                </div>
              </div>
            </div>
          )}
        </div>

        <AnalysisStatusPanel
          mode={mode}
          active="analysis"
          statRows={statRows}
          nextLabel={nextLabel}
          nextDesc={nextDesc}
          ctaLabel={ctaLabel}
          onAdvance={handleAdvance}
          advancing={advancing}
        />
      </div>
    </div>

    {workDetail && (
      <div
        className="glass"
        style={{
          position: "fixed", top: 0, right: 0, bottom: 0, width: panelWidth, maxWidth: "90vw",
          padding: 24, overflowY: "auto", zIndex: 40, borderLeft: "1px solid var(--glass-border)",
          boxShadow: "-8px 0 24px rgba(28,26,46,0.10)",
        }}
      >
        <div
          onMouseDown={() => { resizingRef.current = true; document.body.style.userSelect = 'none'; }}
          style={{
            position: "absolute", left: 0, top: 0, bottom: 0, width: 8,
            cursor: "col-resize", zIndex: 1,
          }}
        />
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, gap: 8 }}>
          <div style={{ fontWeight: 700, fontSize: 14, lineHeight: 1.5 }}>{workDetail.contestTitle}</div>
          <button
            onClick={() => setWorkDetail(null)}
            aria-label="닫기"
            style={{ background: "none", border: "none", padding: 4, cursor: "pointer", color: "var(--text-2)", flexShrink: 0 }}
          >
            <X size={16} />
          </button>
        </div>
        {workDetail.works.map((w, i) => (
          <div key={i} style={{ padding: "14px 0", borderTop: i > 0 ? "1px solid var(--glass-border)" : "none" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <span className={`badge ${w.selection_status === "winner" ? "green" : "amber"} mono`}>
                {w.selection_status === "winner" ? "수상" : "후보"}
              </span>
              {w.award_grade && <span style={{ fontSize: 11.5, color: "var(--text-2)" }}>{w.award_grade}</span>}
            </div>
            {w.source_url ? (
              <a
                href={w.source_url}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: "inline-flex", alignItems: "center", gap: 4,
                  fontSize: 13.5, fontWeight: 600, marginBottom: 6, color: "var(--purple)",
                }}
              >
                {w.work_title} <ExternalLink size={12} />
              </a>
            ) : (
              <div style={{ fontSize: 13.5, fontWeight: 600, marginBottom: 6 }}>{w.work_title}</div>
            )}
            {w.images.length > 0 && (
              <img src={w.images[0]} alt={w.work_title} style={{ width: "100%", borderRadius: 8, marginBottom: 6, display: "block" }} />
            )}
            {w.ocr_text && <div style={{ fontSize: 12.5, color: "var(--text-1)", lineHeight: 1.6 }}>{w.ocr_text}</div>}
          </div>
        ))}
      </div>
    )}
    </>
  );
}

function overallPercent(snapshot) {
  if (!snapshot) return 0
  if (snapshot.chair_done) return 100
  if (snapshot.score_done) return 85
  if (snapshot.reviews_total) return 10 + (snapshot.reviews_done / snapshot.reviews_total) * 60
  return 5
}

/* ---------------- 5. 작성 후: 기획서 업로드 → 분석 시작 → 피드백 확인 (실제 API) ----------------
 * 가은/Claude(2026-07-20): 평가 대상 문서 업로드는 DocumentUploadPage.jsx(/projects/new)의
 * uploadDocument(pid, file, 'pdf', 'target') 로직을 그대로 재사용한다. "분석 시작"은
 * MentorSelectionPage.jsx가 하던 getMentorCandidates → analyzeProject → getAnalyzeProgress
 * 폴링을 그대로 쓰되, 멘토 선택 화면 없이 추천 후보를 전부(최대 4명) 자동 선택한다.
 */
function UploadAndAnalyzeScreen({ projectId, onFeedbackReady, onBack, initialDocuments }) {
  const [documents, setDocuments] = useState(() => initialDocuments || [])
  const [isDragging, setIsDragging] = useState(false)
  const [fileError, setFileError] = useState('')
  const fileInputRef = useRef(null)

  const [candidatesError, setCandidatesError] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [snapshot, setSnapshot] = useState(null)
  const [mentorCount, setMentorCount] = useState(0)
  const [reviewsReady, setReviewsReady] = useState(false)
  const progressTokenRef = useRef(null)
  const pollTimerRef = useRef(null)
  // 가은/Claude(2026-07-22): 실측 버그 — "분석 진행률이 85%에서 15%로 되돌아간다". 원인은
  // handleStartAnalyze의 유일한 중복 실행 방지가 `if (analyzing) return`(React state)뿐이라,
  // 클릭 이벤트 두 번이 setAnalyzing(true)의 리렌더 커밋보다 먼저 잇달아 들어오면 두 번 다
  // 그 가드를 통과한다 — 그러면 서로 다른 progressToken으로 분석이 두 벌 시작되고, 폴링
  // setInterval도 두 개가 생기는데 화면엔 snapshot 하나뿐이라 "나중에 도착한 응답"이
  // 이긴다. 먼저 시작한 분석이 85%(score_done)까지 가도, 뒤늦게 시작한 두 번째 분석의
  // 낮은 진행률(예: reviews_done 1/12 = 15%) 응답이 나중에 도착하면 화면이 되돌아간
  // 것처럼 보인다. state는 커밋까지 지연될 수 있어 신뢰할 수 없으므로, 클릭 즉시(동기적으로)
  // 갱신되는 ref로 잠근다.
  const startingRef = useRef(false)

  // 가은/Claude(2026-07-20): 실측 버그 — 분석 진행 중에 사이드바로 다른 단계로 이동하면
  // (unmount) 이 interval이 안 멈추고 계속 살아서 백엔드를 계속 두드렸다
  // (MentorSelectionPage.jsx에는 있던 언마운트 정리를 포팅하면서 빠뜨림).
  useEffect(() => {
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current)
        pollTimerRef.current = null
      }
    }
  }, [])

  function updateDoc(id, patch) {
    setDocuments((prev) => prev.map((doc) => (doc.id === id ? { ...doc, ...patch } : doc)))
  }

  async function uploadOne(file) {
    const id = `${file.name}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
    setDocuments((prev) => [...prev, { id, name: file.name, meta: formatFileSize(file.size), status: 'uploading', progress: 30 }])
    try {
      const doc = await uploadDocument(projectId, file, 'pdf', 'target')
      if (doc.status === 'conversion_failed') {
        updateDoc(id, { status: 'error', meta: doc.conversion_metadata?.conversion_error || '문서를 변환하지 못했습니다.' })
        return
      }
      // 가은/Claude(2026-07-21): 실측 요청("업로드가 느리다") — 색인이 백그라운드로 바뀌어
      // 업로드 응답이 "indexing"으로 즉시 돌아온다. 행을 "색인 중"으로 바꾸고 폴링으로
      // 완료를 확인한다. "분석 시작" 버튼은 계속 status==='done'(색인 완료)에만 열린다 —
      // 분석(멘토 추천·리뷰)이 색인된 청크를 근거로 쓰기 때문이다.
      if (doc.status === 'indexing') {
        updateDoc(id, { backendId: doc.id, status: 'embedding', meta: '문서를 색인하는 중...', progress: 75 })
        pollDocumentIndexing(projectId, doc.id, id, 'done', formatFileSize(file.size), updateDoc)
        return
      }
      updateDoc(id, { status: 'done', progress: 100 })
    } catch (err) {
      updateDoc(id, { status: 'error', meta: err.message })
    }
  }

  function addFiles(fileList) {
    const files = Array.from(fileList)
    const accepted = files.filter(isAcceptedDocument)
    const rejected = files.length - accepted.length
    setFileError(rejected > 0 ? 'PDF, DOCX, PPTX, HWP, HWPX 파일만 업로드할 수 있습니다.' : '')
    accepted.forEach(uploadOne)
  }

  const dropHandlers = {
    onDragOver: (e) => { e.preventDefault(); setIsDragging(true) },
    onDragLeave: (e) => { e.preventDefault(); setIsDragging(false) },
    onDrop: (e) => { e.preventDefault(); setIsDragging(false); addFiles(e.dataTransfer.files) },
  }

  const hasReadyDocument = documents.some((d) => d.status === 'done')

  // 가은/Claude(2026-07-20): "멘토링선택xx" — MentorSelectionPage의 후보 카드 선택 UI 없이,
  // getMentorCandidates()가 추천한 후보를 순서대로 최대 4명까지 자동으로 그대로 쓴다.
  async function handleStartAnalyze() {
    // 동기적 잠금 — React state(analyzing)는 커밋이 지연될 수 있어 빠른 두 번째 클릭이
    // 이 함수를 다시 통과할 수 있다(위 startingRef 주석 참고). ref는 대입 즉시 반영된다.
    if (startingRef.current || analyzing) return
    startingRef.current = true
    setCandidatesError('')
    setAnalyzing(true)
    try {
      const data = await getMentorCandidates(projectId)
      const autoSelected = data.candidates.slice(0, MAX_MENTORS).map((c) => c.persona_id)
      if (autoSelected.length < MIN_MENTORS) {
        throw new Error('추천 멘토가 충분하지 않습니다.')
      }
      setMentorCount(autoSelected.length)

      progressTokenRef.current =
        typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`
      const progressToken = progressTokenRef.current

      pollTimerRef.current = setInterval(() => {
        getAnalyzeProgress(projectId, progressToken).then((snap) => {
          if (!snap) return
          // 가은/Claude(2026-07-22): 방어적 이중 안전장치 — 위 startingRef 잠금으로 중복 실행
          // 자체를 막았지만, 네트워크 지연으로 폴링 요청·응답 순서가 뒤바뀔 가능성은 여전히
          // 남는다(요청 A가 늦게 도착). 같은 실행 안에서 진행률이 이미 도달한 지점보다 뒤로
          // 가는 스냅샷은 화면에 반영하지 않는다 — 진행률은 원래 한쪽으로만 진행해야 한다.
          setSnapshot((prev) => (prev && overallPercent(snap) < overallPercent(prev) ? prev : snap))
          if (snap.chair_done && pollTimerRef.current) {
            clearInterval(pollTimerRef.current)
            pollTimerRef.current = null
          }
        })
      }, ANALYZE_POLL_INTERVAL_MS)

      const result = await analyzeProject(projectId, autoSelected, progressToken)
      sessionStorage.setItem(`analysis:${projectId}`, JSON.stringify(result))
      setReviewsReady(true)
    } catch (err) {
      setCandidatesError(err.message)
      setAnalyzing(false)
      startingRef.current = false
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current)
        pollTimerRef.current = null
      }
    }
  }

  const reviewsDone = !!snapshot && snapshot.reviews_total > 0 && snapshot.reviews_done >= mentorCount
  const analyzePercent = overallPercent(snapshot)

  return (
    <div style={{ maxWidth: 720 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>
        <button type="button" className="rb-back-button" onClick={onBack} aria-label="이전 화면으로 이동">
          {'←'}
        </button>
        <div className="badge coral mono">기획서 업로드 · 분석</div>
      </div>
      <h2 style={{ margin: '0 0 20px', fontSize: 22, fontWeight: 700 }}>
        {analyzing ? '평가 대상 문서를 분석중이에요' : '평가 대상 문서를 업로드하세요'}
      </h2>

      {!analyzing && (
        <>
          <div
            className="card glass"
            style={{
              borderStyle: 'dashed',
              borderColor: isDragging ? 'var(--coral)' : 'var(--glass-border)',
              padding: 40, textAlign: 'center',
            }}
            {...dropHandlers}
          >
            <Upload size={26} color="var(--coral)" style={{ marginBottom: 14 }} />
            <div style={{ fontWeight: 600, marginBottom: 6 }}>{isDragging ? '여기에 놓으세요' : '기획서를 업로드하세요'}</div>
            <div style={{ fontSize: 12.5, color: 'var(--text-2)', marginBottom: 20 }}>PDF, DOCX, PPTX, HWP, HWPX</div>
            <button className="btn-primary" onClick={() => fileInputRef.current?.click()}>파일 선택</button>
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_DOCUMENT_EXTENSIONS.join(',')}
              multiple
              style={{ display: 'none' }}
              onChange={(e) => { addFiles(e.target.files); e.target.value = '' }}
            />
          </div>

          {fileError && <p style={{ color: 'var(--coral)', fontSize: 13, marginTop: 12 }}>{fileError}</p>}

          {documents.length > 0 && (
            <div className="card glass" style={{ marginTop: 16 }}>
              {documents.map((doc, i) => (
                <div key={doc.id} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '10px 0', borderTop: i > 0 ? '1px solid var(--glass-border)' : 'none',
                }}>
                  <div>
                    <div style={{ fontSize: 13.5, fontWeight: 600 }}>{doc.name}</div>
                    <div style={{ fontSize: 12, color: 'var(--text-2)' }}>{doc.meta}</div>
                  </div>
                  {(doc.status === 'uploading' || doc.status === 'embedding') && (
                    <ProgressBar
                      percent={doc.progress}
                      label={doc.status === 'uploading' ? '업로드 중' : '색인 중'}
                      color="linear-gradient(135deg, #7c5cea, #8b6ef0)"
                      trackColor="var(--bg-2)"
                    />
                  )}
                  {doc.status === 'done' && <span className="badge green mono">✓ 완료</span>}
                  {doc.status === 'warning' && <span className="badge amber mono">확인 필요</span>}
                  {doc.status === 'error' && <span className="badge coral mono">실패</span>}
                </div>
              ))}
            </div>
          )}

          {candidatesError && <p style={{ color: 'var(--coral)', fontSize: 13, marginTop: 12 }}>{candidatesError}</p>}

          <button
            className="btn-primary"
            style={{ marginTop: 24, width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
            disabled={!hasReadyDocument}
            onClick={handleStartAnalyze}
          >
            분석 시작 <ArrowRight size={15} />
          </button>
          {!hasReadyDocument && <p style={{ fontSize: 12, color: 'var(--text-2)', textAlign: 'center', marginTop: 8 }}>문서 업로드가 끝나면 분석을 시작할 수 있어요.</p>}
        </>
      )}

      {analyzing && (
        <div className="card glass">
          <p style={{ fontSize: 13, color: 'var(--text-1)', marginBottom: 16 }}>
            문서 피드백 준비중
          </p>
          <div className="progress-track" style={{ marginBottom: 6 }}>
            <div className="progress-fill" style={{ width: `${analyzePercent}%` }} />
          </div>
          <p style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 20 }}>{Math.round(analyzePercent)}%</p>

          {reviewsReady && (
            <button className="btn-primary" style={{ width: '100%' }} onClick={() => onFeedbackReady(projectId)}>
              피드백 확인하기 →
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* ---------------------------- 페이지 진입점 ---------------------------- */
export default function ReviewBoardPrototype() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  // 가은/Claude(2026-07-21): 실측 요청 — "내 프로젝트"에서 프로젝트를 불러오면 board에서
  // 올렸던 공고문·기획서·분석결과가 안 보이던 문제(구 ProjectDetailPage로 가서 board와
  // 완전히 무관한 화면을 보여줬음). ProjectListPage가 이제 /board?projectId=로 보낸다 —
  // DocumentUploadPage.jsx의 ?projectId= "이어서 하기" 패턴과 동일.
  const resumeProjectId = searchParams.get('projectId');
  const [resuming, setResuming] = useState(!!resumeProjectId);
  const [resumeError, setResumeError] = useState('');
  const [mode, setMode] = useState(null);
  const [stage, setStage] = useState("entry");
  const [projectId, setProjectId] = useState(null);
  const [targetDocuments, setTargetDocuments] = useState(null);
  const [entryLoading, setEntryLoading] = useState(false);
  const [entryError, setEntryError] = useState('');
  const [ideaSaving, setIdeaSaving] = useState(false);
  const [ideaSaveError, setIdeaSaveError] = useState('');
  // 가은/Claude(2026-07-20): 실측 버그 — URL로 실제 공고문을 수집해도 "공모전 분석"
  // 화면엔 항상 똑같은 고정 예시 카드만 나왔다(EntryScreen이 모은 문서 목록이 어디에도
  // 안 넘어갔음). AnalysisScreen에서 보여줄 수 있게 여기(부모)로 끌어올린다.
  const [criteriaDocuments, setCriteriaDocuments] = useState([]);
  // 용준/Claude(2026-07-21): "작성 전 → 주제 발굴" 실 연동 — 대화형 아이디어 회의 세션의
  // 최신 API 응답(session_id 포함)을 부모(이 컴포넌트)의 state로 관리한다. "이전 단계로
  // 갔다 돌아와도 회의 상태 유지"를 만족시키려면 IdeationScreen이 언마운트됐다 다시
  // 마운트돼도(사이드바로 다른 단계를 갔다 오는 경우) 값을 잃지 않아야 하기 때문이다 —
  // IdeationConversationScreen.jsx가 이 값이 이미 있으면 절대 start API를 다시 부르지
  // 않는다.
  const [ideationConv, setIdeationConv] = useState(null);

  const goNext = () => {
    const seq = (mode && FLOW_BY_MODE[mode]) || ["entry"];
    const i = seq.indexOf(stage);
    if (i < seq.length - 1) setStage(seq[i + 1]);
  };

  const goPrev = () => {
    const seq = (mode && FLOW_BY_MODE[mode]) || ["entry"];
    const i = seq.indexOf(stage);
    if (i > 0) setStage(seq[i - 1]);
  };

  // 가은/Claude(2026-07-20): projectId가 아직 없으면(공고 URL/파일을 하나도 안 넣고
  // 바로 "분석 시작"을 눌렀거나, EntryScreen의 URL/파일 액션이 이미 만들어뒀거나) 여기서
  // 한 번 더 보장한다 — DocumentUploadPage.jsx의 ensureProject()와 동일한 "지연 생성"
  // 패턴이라 URL/파일을 여러 번 넣어도 프로젝트가 중복 생성되지 않는다.
  const projectIdRef = useRef(null);
  const ideaProjectSavedRef = useRef(false);
  projectIdRef.current = projectId;
  async function ensureProject() {
    if (projectIdRef.current) return projectIdRef.current;
    // 가은/Claude(2026-07-21): 실측 요청 — "작성 전/작성 후" 중 뭘 골랐는지가 프로젝트에
    // 안 남아서, "내 프로젝트"에서 다시 불러오면 어느 흐름이었는지 알 방법이 없었다.
    // 이 시점엔 EntryScreen의 onModeSelect 콜백 덕분에 mode가 이미 채워져 있다.
    const project = await createProject({ title: "새 공모전 프로젝트", doc_type: "competition", flow_mode: mode });
    projectIdRef.current = project.id;
    setProjectId(project.id);
    return project.id;
  }

  async function handleConfirmIdeaProject(finalizedConversation = ideationConv) {
    if (ideaSaving) return;
    if (ideaProjectSavedRef.current) {
      goNext();
      return;
    }

    setIdeaSaveError('');
    setIdeaSaving(true);
    try {
      const confirmedTopic = finalizedConversation?.idea_proposal?.idea_name?.trim() || IDEA_PROJECT_TOPIC;
      // 가은/Claude(2026-07-21): 실측 버그 — EntryScreen에서 공고문을 먼저 등록해
      // ensureProject()로 이미 프로젝트가 만들어져 있는데도 여기서 항상 새 프로젝트를
      // 만들어버려서, 앞서 등록한 공고문이 붙은 프로젝트가 고아가 됐다. 이미 프로젝트가
      // 있으면 새로 만들지 않고 제목·설명만 아이디어 확정 값으로 갱신한다.
      const project = projectIdRef.current
        ? await updateProject(projectIdRef.current, { title: confirmedTopic, description: IDEA_PROJECT_MARKER, flow_mode: "pre" })
        : await createProject({
            title: confirmedTopic,
            doc_type: "competition",
            description: IDEA_PROJECT_MARKER,
            flow_mode: "pre",
          });
      ideaProjectSavedRef.current = true;
      projectIdRef.current = project.id;
      setProjectId(project.id);
      goNext();
    } catch (err) {
      setIdeaSaveError(err.message);
    } finally {
      setIdeaSaving(false);
    }
  }

  async function handleEnter(m) {
    setMode(m);
    // 가은/Claude(2026-07-21): 실측 요청 — 예전엔 "작성 전" 모드는 공고문을 안 넣었으면
    // 프로젝트를 아예 안 만들고 넘어가서 flow_mode를 저장할 데가 없었다. 이제 두 모드
    // 모두 여기서 프로젝트를 보장(ensureProject)해서 flow_mode가 항상 남는다.
    setEntryError("");
    setEntryLoading(true);
    try {
      await ensureProject();
      // 작성 후(문서 피드백)는 공모전 분석을 거치지 않고 바로 기획서 업로드·분석으로.
      setStage(m === "post" ? "upload" : "analysis");
    } catch (err) {
      setEntryError(err.message);
    } finally {
      setEntryLoading(false);
    }
  }

  // 재인/Claude(2026-07-21): 예전엔 분석 끝나면 기존 대화형 피드백 화면
  // (/feedback-chat)으로 이동했는데, 이제 그 대신 "AI 피드백"(워크벤치) 단계로
  // 이 페이지 안에서 이어지도록 바꿨다 - 워크벤치 안에서 필요하면 그 화면(위원
  // 소집)을 다시 불러오는 방식이라, 여기서는 완전히 벗어나지 않는다.
  function handleFeedbackReady() {
    setStage('workbench');
  }

  // 가은/Claude(2026-07-21): ?projectId=가 있으면 기존 프로젝트를 불러와 이어서 한다.
  // 프로젝트 설명이 IDEA_PROJECT_MARKER면 "작성 전" 흐름에서 주제를 확정한 프로젝트로
  // 보고 mode를 pre로 두고 분석 화면부터 보여준다(문서가 없어도 entry에 멈추지 않도록).
  // 그 외(post)는 문서가 하나라도 있으면 일단 "기획서 업로드·분석" 후보인데, 분석이
  // 이미 끝난 프로젝트(실측 제보 — 회의가 완료됐는데도 매번 업로드 화면이 다시 뜸)라면
  // getLatestMeeting()으로 확인해 업로드 화면을 건너뛰고 바로 워크벤치(결과)로 보낸다.
  // 회의가 없으면(404) 기존대로 업로드 화면. 아무 문서도 없으면 entry에 그대로 둔다.
  useEffect(() => {
    if (!resumeProjectId) return;
    let cancelled = false;
    setResuming(true);
    setResumeError('');
    Promise.all([getDocuments(resumeProjectId), getProject(resumeProjectId)])
      .then(async ([docs, project]) => {
        if (cancelled) return;
        projectIdRef.current = resumeProjectId;
        setProjectId(resumeProjectId);
        // 가은/Claude(2026-07-21): flow_mode가 주 판단 기준 — EntryScreen에서 모드를
        // 고르는 순간 저장된다. description 마커는 flow_mode가 없던 예전 프로젝트를 위한
        // 하위호환 폴백(아이디어 확정까지 끝난 프로젝트만 잡아낼 수 있었음).
        const isPreFlow = project.flow_mode === 'pre' || project.description === IDEA_PROJECT_MARKER;
        setMode(isPreFlow ? 'pre' : 'post');
        if (project.description === IDEA_PROJECT_MARKER) ideaProjectSavedRef.current = true;

        const criteriaDocs = docs.filter((d) => d.document_role === 'criteria');
        const targetDocs = docs.filter((d) => (d.document_role || 'target') === 'target');

        setCriteriaDocuments(
          criteriaDocs.map((d) => ({
            id: d.id,
            backendId: d.id,
            name: d.original_filename,
            meta: '이전에 등록한 문서',
            status: _resumedDocStatus(d.status),
          })),
        );
        setTargetDocuments(
          targetDocs.map((d) => ({
            id: d.id,
            name: d.original_filename,
            meta: formatFileSize(d.file_size),
            status: _resumedDocStatus(d.status),
          })),
        );

        // post 흐름에는 이제 공모전 분석 단계가 없다 — 문서(기획서든 공고문이든)가
        // 하나라도 있으면 기획서 업로드·분석 후보, 아무것도 없으면 entry 유지.
        if (isPreFlow) {
          setStage('analysis');
          return;
        }
        if (targetDocs.length === 0 && criteriaDocs.length === 0) return;

        const hasCompletedMeeting = await getLatestMeeting(resumeProjectId).then(() => true).catch(() => false);
        if (cancelled) return;
        setStage(hasCompletedMeeting ? 'workbench' : 'upload');
      })
      .catch((err) => { if (!cancelled) setResumeError(err.message); })
      .finally(() => { if (!cancelled) setResuming(false); });
    return () => { cancelled = true; };
  }, [resumeProjectId]);

  if (resuming) {
    return (
      <Shell active={stage} mode={mode} onNavigate={setStage} showNav={false}>
        <div style={{ maxWidth: 760, margin: "40px auto" }}>
          <div className="badge purple mono">불러오는 중</div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: "12px 0 24px" }}>이전에 등록한 프로젝트를 불러오고 있어요...</h1>
        </div>
      </Shell>
    );
  }

  return (
    <Shell active={stage} mode={mode} onNavigate={setStage} showNav={stage !== "entry"}>
      {resumeError && <p style={{ color: "var(--coral)", fontSize: 13, marginBottom: 16 }}>{resumeError}</p>}
      {stage === "entry" && (
        <EntryScreen
          onEnter={handleEnter}
          onModeSelect={setMode}
          loading={entryLoading}
          error={entryError}
          projectId={projectId}
          ensureProject={ensureProject}
          documents={criteriaDocuments}
          setDocuments={setCriteriaDocuments}
        />
      )}
      {stage === "analysis" && (
        <AnalysisScreen mode={mode} onNext={goNext} onBack={goPrev} projectId={projectId} />
      )}
      {stage === "ideation" && (
        <IdeationScreen
          projectId={projectId}
          criteriaDocuments={criteriaDocuments}
          ideationConv={ideationConv}
          setIdeationConv={setIdeationConv}
          onFinalized={handleConfirmIdeaProject}
          onBack={goPrev}
          saving={ideaSaving}
          saveError={ideaSaveError}
        />
      )}
      {stage === "ideation_result" && <IdeationResultScreen ideationConv={ideationConv} onBack={goPrev} />}
      {stage === "upload" && (
        <UploadAndAnalyzeScreen projectId={projectId} onFeedbackReady={handleFeedbackReady} onBack={goPrev} initialDocuments={targetDocuments} />
      )}
      {stage === "workbench" && <WorkbenchScreen projectId={projectId} onNext={goNext} />}
      {stage === "report" && <VersionTrackerTestPage embedded projectId={projectId} />}
    </Shell>
  );
}
