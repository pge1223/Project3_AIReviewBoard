import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getProjects, deleteProject, getLatestMeeting } from '../api/projectApi'
import StatusBadge from '../components/common/StatusBadge'
import { IDEA_PROJECT_MARKER } from './board/ReviewBoardPrototype'

// 가은/Claude(2026-07-21): 실측 제보 — "공모전 분석"(작성 전)이나 "기획서 업로드·분석"
// (작성 후)에서 더 못 넘어가고 이탈한 프로젝트가 "내 프로젝트"에 그대로 쌓였다.
// ensureProject()가 "분석 시작"만 눌러도(문서 하나 없이도) DB row를 만들기 때문 —
// status 필드는 늘 "pending" 고정이라 이걸로는 구분이 안 된다(ReviewBoardPrototype.jsx
// resume 로직과 동일하게 실제 신호로 판단): 작성 후는 회의(분석) 완료 여부(getLatestMeeting)
// 로 본다.
// ⚠️ 작성 전은 처음에 "아이디어 확정(IDEA_PROJECT_MARKER)까지 끝나야 유지"로 잘못
// 구현했었다 — 실측 결과 "공모전 분석은 끝났는데 아이디어 회의 직전"인 프로젝트까지
// 걸러져버렸다(분석 화면 로딩 중에 이탈한 것과 구분이 안 됐던 게 원인). 공모전 분석이
// 실제로 끝났는지는 has_announcement_analysis(백엔드 announcement_analysis_cache 존재
// 여부, PRJ-002/003 응답에 새로 추가)로 판단한다 — 아이디어 확정 전이라도 이거면 유지.
function hasMeaningfulProgress(project) {
  const isPreFlow = project.flow_mode === 'pre' || project.description === IDEA_PROJECT_MARKER
  if (isPreFlow) return project.description === IDEA_PROJECT_MARKER || !!project.has_announcement_analysis
  return getLatestMeeting(project.id).then(() => true).catch(() => false)
}

function FolderIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--purple)" strokeWidth="1.8">
      <path d="M3 6.5C3 5.67 3.67 5 4.5 5h4.4c.35 0 .68.14.93.38L11 6.5h8.5c.83 0 1.5.67 1.5 1.5v9.5c0 .83-.67 1.5-1.5 1.5h-15C3.67 19 3 18.33 3 17.5v-11Z" />
    </svg>
  )
}

function ChevronIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--text-2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 6l6 6-6 6" />
    </svg>
  )
}

// 가은/Claude(2026-07-20): "리스트에서 새 프로젝트를 왼->오로 밀면 삭제 버튼이 나오게"
// — 라이브러리 없이 pointer 이벤트로 직접 구현. 왼쪽 밑에 항상 깔려 있는 삭제 버튼을
// 앞의 행(row)을 오른쪽으로 끌어서(translateX) 드러내는 방식. 드래그였는지 탭이었는지는
// 이동 거리(CLICK_MOVE_THRESHOLD)로 구분해서, 살짝 흔들린 탭까지 드래그로 오인해 원래
// 있던 "행 클릭 -> 상세 이동" 동작을 깨지 않게 한다. 한 번에 하나의 행만 열려있게
// 부모(ProjectListPage)가 openRowId로 관리한다 — 다른 행을 열면 이전 행은 자동으로 닫힘.
// 가은/Claude(2026-07-20): "버튼 나온 데서 조금만 더 밀면 삭제 팝업 뜨게" — 버튼이 열리는
// REVEAL_WIDTH를 넘어 FULL_SWIPE_WIDTH까지 더 끌고 놓으면(iOS 메일 앱의 완전 스와이프
// 삭제와 동일한 패턴) 버튼을 따로 안 눌러도 바로 onDelete(확인 팝업)를 띄운다. 완전
// 스와이프 구간에 들어서면 버튼 배경/문구를 바꿔서 "놓으면 삭제"라는 걸 미리 알려준다.
const REVEAL_WIDTH = 88
const FULL_SWIPE_WIDTH = REVEAL_WIDTH + 70
const MAX_DRAG = FULL_SWIPE_WIDTH + 20
const CLICK_MOVE_THRESHOLD = 6

function SwipeableRow({ project, isOpen, isLast, deleting, onOpenChange, onDelete, onNavigate }) {
  const [dragX, setDragX] = useState(isOpen ? REVEAL_WIDTH : 0)
  const [dragging, setDragging] = useState(false)
  const gestureRef = useRef({ startX: 0, baseX: 0, moved: false })

  useEffect(() => {
    if (!dragging) setDragX(isOpen ? REVEAL_WIDTH : 0)
  }, [isOpen, dragging])

  function handlePointerDown(e) {
    gestureRef.current = { startX: e.clientX, baseX: isOpen ? REVEAL_WIDTH : 0, moved: false }
    setDragging(true)
    // 빠르게 끌면 포인터가 행 영역을 벗어날 수 있어서, capture로 이후 move/up 이벤트를
    // 계속 이 요소로 받는다(터치에서 특히 중요).
    e.currentTarget.setPointerCapture?.(e.pointerId)
  }

  function handlePointerMove(e) {
    if (!dragging) return
    const delta = e.clientX - gestureRef.current.startX
    if (Math.abs(delta) > CLICK_MOVE_THRESHOLD) gestureRef.current.moved = true
    setDragX(Math.max(0, Math.min(MAX_DRAG, gestureRef.current.baseX + delta)))
  }

  function endDrag() {
    if (!dragging) return
    setDragging(false)
    if (dragX >= FULL_SWIPE_WIDTH) {
      // 완전 스와이프 — 버튼을 따로 누르지 않아도 바로 삭제 확인을 띄운다. 확인 팝업에서
      // 취소하면 onDelete가 삭제를 진행하지 않으니, 버튼이 열려 있는 상태로만 되돌린다.
      setDragX(REVEAL_WIDTH)
      onOpenChange(project.id)
      onDelete(project.id)
      return
    }
    const shouldOpen = dragX > REVEAL_WIDTH / 2
    setDragX(shouldOpen ? REVEAL_WIDTH : 0)
    onOpenChange(shouldOpen ? project.id : null)
  }

  function handleRowClick() {
    if (gestureRef.current.moved) return // 드래그 끝의 클릭 이벤트는 무시
    if (isOpen) {
      onOpenChange(null) // 열려있는 상태에서 행을 탭하면 닫기만 한다
      return
    }
    onNavigate(project.id)
  }

  const pastFull = dragX >= FULL_SWIPE_WIDTH
  // 버튼 너비를 dragX만큼 늘려서, REVEAL_WIDTH를 넘어 더 끄는 동안 행과 버튼 사이에
  // 빈 틈(카드 배경)이 보이지 않고 버튼이 같이 늘어나는 것처럼 보이게 한다.
  const buttonWidth = Math.max(REVEAL_WIDTH, dragX)

  return (
    <div style={{ position: 'relative', overflow: 'hidden', borderBottom: isLast ? 'none' : '1px solid var(--glass-border)' }}>
      <button
        style={{
          ...styles.deleteButton,
          width: buttonWidth,
          background: pastFull ? '#a8432e' : '#c05339',
          transition: dragging ? 'none' : 'width 0.2s ease, background 0.15s ease',
        }}
        disabled={deleting}
        onClick={(e) => {
          e.stopPropagation()
          onDelete(project.id)
        }}
      >
        {deleting ? '삭제 중...' : pastFull ? '놓으면 삭제' : '삭제'}
      </button>
      <div
        style={{
          ...styles.row,
          transform: `translateX(${dragX}px)`,
          transition: dragging ? 'none' : 'transform 0.2s ease',
        }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
        onPointerCancel={endDrag}
        onClick={handleRowClick}
      >
        <div style={styles.rowLeft}>
          <FolderIcon />
          <div>
            <div style={styles.rowTitle}>{project.title}</div>
            <div style={styles.rowDate}>{String(project.created_at).slice(0, 10)} 생성</div>
          </div>
        </div>
        <div style={styles.rowRight}>
          <StatusBadge status={project.status} />
          <ChevronIcon />
        </div>
      </div>
    </div>
  )
}

export default function ProjectListPage() {
  const navigate = useNavigate()
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [openRowId, setOpenRowId] = useState(null)
  const [deletingId, setDeletingId] = useState(null)
  const [deleteError, setDeleteError] = useState('')

  useEffect(() => {
    let cancelled = false
    getProjects()
      .then(async (data) => {
        const flags = await Promise.all(data.map(hasMeaningfulProgress))
        if (cancelled) return
        setProjects(data.filter((_, i) => flags[i]))
        setLoading(false)
      })
      .catch((err) => {
        if (cancelled) return
        setError(err.message)
        setLoading(false)
      })
    return () => { cancelled = true }
  }, [])

  function handleDelete(projectId) {
    if (deletingId) return
    const project = projects.find((p) => p.id === projectId)
    if (!window.confirm(`"${project?.title ?? '이 프로젝트'}"를 삭제할까요? 되돌릴 수 없습니다.`)) {
      return
    }
    setDeleteError('')
    setDeletingId(projectId)
    deleteProject(projectId)
      .then(() => {
        setProjects((prev) => prev.filter((p) => p.id !== projectId))
        setOpenRowId(null)
      })
      .catch((err) => setDeleteError(err.message))
      .finally(() => setDeletingId(null))
  }

  const doneCount = projects.filter((p) => p.status === 'reviewed').length

  return (
    <div className="pl-root" style={styles.page}>
      {/* 가은/Claude(2026-07-21): 실측 요청 — "projects 테마도 board처럼" + 오른쪽 패널
          추가. board(ReviewBoardPrototype.jsx)의 --bg-0/--purple 등 팔레트·글래스 카드
          룩을 그대로 옮겨왔다. 이 페이지는 이제 board에서만 진입하는 게 자연스러워
          기존 StepSidebar(다른 레거시 화면들과 공유하는 컴포넌트라 직접 안 건드림) 대신
          board와 같은 톤의 자체 헤더로 바꿨다. */}
      <style>{`
        .pl-root{
          --bg-0:#faf8f4; --bg-1:#ffffff; --bg-2:#f1eee5;
          --glass: rgba(255,255,255,0.72); --glass-border: rgba(28,26,46,0.10);
          --purple:#7c5cea; --purple-dim: rgba(124,92,234,0.12);
          --coral:#e0603d; --coral-dim: rgba(224,96,61,0.12);
          --green:#16a37a; --green-dim: rgba(22,163,122,0.12);
          --text-0:#1c1a2e; --text-1:#5b5770; --text-2:#918d9f;
          --mono: 'JetBrains Mono', ui-monospace, monospace;
        }
        .pl-glass{ background:var(--glass); border:1px solid var(--glass-border); backdrop-filter: blur(14px); box-shadow: 0 2px 14px rgba(28,26,46,0.05); }
        .pl-row:hover{ background:var(--bg-2); }
        @media (max-width: 860px){
          .pl-root{ flex-direction:column !important; }
          .pl-panel{ display:none; }
        }
      `}</style>

      <main style={styles.main}>
        <div style={styles.headerRow}>
          <div className="badge-wrap" style={styles.brand}>AI Review Board</div>
        </div>

        <div style={styles.header}>
          <div style={styles.titleRow}>
            <button style={styles.backButton} onClick={() => navigate('/board')} aria-label="board로 돌아가기">&lt;</button>
            <h1 style={styles.title}>내 프로젝트</h1>
          </div>
        </div>

        {deleteError && <p style={styles.deleteErrorText}>{deleteError}</p>}

        <div className="pl-glass" style={styles.card}>
          {loading && <p style={styles.empty}>불러오는 중...</p>}
          {!loading && error && <p style={styles.empty}>{error}</p>}
          {!loading && !error && projects.length === 0 && <p style={styles.empty}>아직 프로젝트가 없습니다.</p>}

          {!error &&
            projects.map((project, i) => (
              <SwipeableRow
                key={project.id}
                project={project}
                isLast={i === projects.length - 1}
                isOpen={openRowId === project.id}
                deleting={deletingId === project.id}
                onOpenChange={setOpenRowId}
                onDelete={handleDelete}
                // 가은/Claude(2026-07-21): 실측 요청 — "내 프로젝트"에서 프로젝트를
                // 열면 board에서 올린 공고문·기획서·분석결과가 보여야 한다. 구
                // ProjectDetailPage(/projects/:id)는 board와 무관한 레거시 화면이라
                // 대신 /board로 이어서 하기(ReviewBoardPrototype.jsx의 ?projectId=
                // 처리부 참고) 보낸다.
                onNavigate={(id) => navigate(`/board?projectId=${id}`)}
              />
            ))}
        </div>
      </main>

      <aside className="pl-panel" style={styles.assistantPanel}>
        <div style={styles.assistantBubbleRow}>
          <div style={styles.assistantIcon}>✨</div>
          <div className="pl-glass" style={styles.assistantBubble}>
            지금까지 등록한 공모전 프로젝트예요. 프로젝트를 누르면 이전에 올린 공고문·기획서와
            분석 결과를 그대로 이어서 볼 수 있어요.
          </div>
        </div>

        <div className="pl-glass" style={styles.overviewBox}>
          <div style={styles.overviewTitle}>요약</div>
          <div style={styles.overviewRow}>
            <span style={styles.overviewLabel}>전체 프로젝트</span>
            <span style={styles.overviewValue}>{projects.length}개</span>
          </div>
          <div style={styles.overviewRow}>
            <span style={styles.overviewLabel}>검토 완료</span>
            <span style={styles.overviewValue}>{doneCount}개</span>
          </div>
        </div>
      </aside>
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100vh',
    display: 'flex',
    background:
      'radial-gradient(1100px 600px at 12% -10%, rgba(124,92,234,0.10), transparent 60%), ' +
      'radial-gradient(900px 500px at 100% 10%, rgba(22,163,122,0.07), transparent 55%), ' +
      'radial-gradient(800px 500px at 50% 110%, rgba(224,96,61,0.06), transparent 55%), ' +
      '#faf8f4',
    color: '#1c1a2e',
    fontFamily: "'Pretendard Variable', Pretendard, -apple-system, sans-serif",
  },
  main: { flex: 1, minWidth: 0, padding: '32px 40px', maxWidth: 760, overflowY: 'auto' },
  headerRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 },
  brand: { fontSize: 13, fontWeight: 700, letterSpacing: '0.02em', color: '#1c1a2e' },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    margin: '4px 0 20px',
  },
  titleRow: { display: 'flex', alignItems: 'center', gap: 10 },
  backButton: {
    background: 'transparent',
    border: 'none',
    borderRadius: 10,
    width: 32,
    height: 32,
    fontSize: 18,
    fontWeight: 700,
    lineHeight: 1,
    color: '#5b5770',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  title: {
    margin: 0,
    fontSize: 24,
    fontWeight: 700,
  },
  deleteErrorText: { color: '#c05339', fontSize: 13, marginBottom: 10 },
  card: {
    borderRadius: 16,
    padding: '4px 20px',
  },
  empty: {
    color: '#918d9f',
    fontSize: 14,
    padding: '24px 0',
    textAlign: 'center',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 4px',
    cursor: 'pointer',
    background: '#ffffff',
    touchAction: 'pan-y',
    userSelect: 'none',
  },
  rowLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  rowTitle: {
    fontSize: 15,
    fontWeight: 600,
    color: '#1c1a2e',
  },
  rowDate: {
    fontSize: 13,
    color: '#918d9f',
    marginTop: 4,
  },
  rowRight: {
    display: 'flex',
    alignItems: 'center',
    gap: 14,
  },
  deleteButton: {
    position: 'absolute',
    top: 0,
    left: 0,
    bottom: 0,
    width: REVEAL_WIDTH,
    border: 'none',
    background: '#c05339',
    color: '#fff',
    fontSize: 13,
    fontWeight: 700,
    cursor: 'pointer',
  },
  assistantPanel: {
    width: 300,
    flexShrink: 0,
    padding: '32px 20px',
  },
  assistantBubbleRow: { display: 'flex', gap: 10, marginBottom: 20 },
  assistantIcon: {
    width: 28,
    height: 28,
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #7c5cea, #8b6ef0)',
    color: '#0b0a16',
    fontSize: 13,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  assistantBubble: {
    borderRadius: 12,
    padding: 12,
    fontSize: 12.5,
    lineHeight: 1.6,
    color: '#5b5770',
  },
  overviewBox: {
    borderRadius: 12,
    padding: 14,
  },
  overviewTitle: { fontSize: 12, fontWeight: 700, color: '#918d9f', marginBottom: 10 },
  overviewRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 12.5,
    padding: '6px 0',
    borderTop: '1px solid rgba(28,26,46,0.10)',
  },
  overviewLabel: { color: '#918d9f' },
  overviewValue: { fontWeight: 600, color: '#1c1a2e' },
}
