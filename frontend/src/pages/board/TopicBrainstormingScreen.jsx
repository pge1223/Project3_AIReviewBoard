import { useEffect, useRef, useState } from 'react'
import { ArrowRight, RefreshCw } from 'lucide-react'
import {
  getIdeationConversation,
  getLatestIdeationConversation,
  replyIdeationConversation,
  startIdeationConversation,
} from '../../api/ideationConversationApi'
import { getAnnouncementAnalysis } from '../../api/documentApi'
import { CandidateCard, ideationSessionStorageKey } from './IdeationConversationScreen'
import {
  KEYWORD_RESELECT_MESSAGE,
  REGENERATE_MESSAGE,
  buildCompetitionDocumentText,
  candidateSelectMessage,
  classifyIdeationConvError,
  competitionNameFrom,
  keywordSelectMessage,
  resolveUseRag,
} from './ideationConversationHelpers'

// pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): 키워드 추천 → 사용자 선택 → 주제 목록
// 생성까지만 담당하는 가벼운 화면. IdeationScreen(대화형 라운드테이블)과 달리 스트리밍·
// 아바타·채팅 로그가 없다 — awaiting_keyword_selection/awaiting_candidate_selection 두
// 정지 지점에서 결정적 선택 UI만 보여주고, 선택이 끝나 phase가 refinement로 넘어가면
// 부모(ReviewBoardPrototype.jsx)가 phase를 보고 stage를 "ideation"으로 바꿔 이 컴포넌트를
// 내리고 IdeationScreen을 올린다.
//
// pge/Claude(2026-07-28, 실측 요청: "페이지처럼 넘기지 말고 위아래로 나오게") — 키워드
// 선택 섹션과 주제 후보 섹션을 phase로 서로 배타적으로 렌더링하던 것을 바꿨다. 이제 둘 다
// 데이터(keyword_options/idea_candidates)가 있으면 항상 같이 보이고, 위(키워드)에서 아래
// (주제 후보)로 쌓인다 — 주제가 생겨도 키워드 섹션이 사라지지 않는다. 다만 실제 키워드
// 선택/재추천 액션 버튼은 phase가 다시 'awaiting_keyword_selection'으로 돌아왔을 때만
// 보인다(백엔드가 그 phase에서만 "선택한 키워드: ..."/재추천 메시지를 올바르게 해석한다) —
// 그 전까지 키워드 섹션은 "이번에 고른 키워드" 요약만 보여주는 읽기 전용 상태다.
const KEYWORD_SOURCE_ORDER = ['trend', 'contest', 'user_issue']

const KEYWORD_SOURCE_LABEL = {
  trend: '요즘 트렌드',
  contest: '공모전 연관',
  user_issue: '직접 남긴 이슈',
}

// pge/Claude(2026-07-27, 실측 요청: "뭐가 어디서 나온 키워드인지 라벨로 분리") — 그룹
// 헤더뿐 아니라 칩 하나하나에도 출처 배지를 붙인다. 색상은 Shell(ReviewBoardPrototype.jsx)의
// 기존 팔레트를 그대로 재사용한다(새 색 추가 안 함).
const KEYWORD_SOURCE_BADGE = {
  trend: { label: '트렌드', color: 'var(--coral)', bg: 'var(--coral-dim)' },
  contest: { label: '공모전', color: 'var(--purple)', bg: 'var(--purple-dim)' },
  user_issue: { label: '내 이슈', color: 'var(--green)', bg: 'var(--green-dim)' },
}

// trend_search_status(백엔드가 keyword_recommendation 시점에 기록)를 그대로 문구로
// 바꾼다 — "트렌드 키워드가 안 보이는데 검색이 된 거냐"는 질문에 로그를 안 봐도 화면에서
// 바로 답이 되도록.
const TREND_STATUS_LABEL = {
  ok: { text: '네이버 실시간 검색 결과가 반영됐어요.', color: 'var(--green)' },
  failed: { text: '네이버 검색이 실패해 트렌드 키워드 없이 진행했어요.', color: 'var(--coral)' },
  skipped: { text: '트렌드 검색이 비활성 상태라 트렌드 키워드가 없어요.', color: 'var(--text-2)' },
}

const MAX_SELECTED_KEYWORDS = 3

export default function TopicBrainstormingScreen({
  projectId,
  criteriaDocuments,
  ideationConv,
  setIdeationConv,
  initialApplicationFormItems,
  onBack,
}) {
  const [starting, setStarting] = useState(!ideationConv)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState(null)
  const [selectedKeywordIds, setSelectedKeywordIds] = useState(() => new Set())
  const startedRef = useRef(false)

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
        useTrendSearch: true,
      }
      const data = await startIdeationConversation(payload)
      setIdeationConv(data)
      const key = ideationSessionStorageKey(projectId)
      if (key) sessionStorage.setItem(key, data.session_id)
    } catch (err) {
      setError(classifyIdeationConvError(err))
    } finally {
      setStarting(false)
    }
  }

  // 가은/Claude(2026-07-28, 실측 요청: "신청서 항목 선택 패널은 다음 페이지로 넘어가기
  // 직전에") — 신청서 항목 조회+선택 패널은 이제 AnalysisScreen(공모전 분석 화면)의 "AI
  // 위원과 주제 확정 시작" 버튼에서 끝낸다. 여기서는 그 결과(initialApplicationFormItems)를
  // 그대로 받아 쓸 뿐, 이 화면에서 다시 조회하거나 패널을 띄우지 않는다.
  async function runStart() {
    setStarting(true)
    setError(null)
    try {
      let analysis = { has_announcement: false }
      if (projectId) {
        analysis = await getAnnouncementAnalysis(projectId)
      }
      await startConversation(analysis, initialApplicationFormItems || [])
    } catch (err) {
      setError(classifyIdeationConvError(err))
      setStarting(false)
    }
  }

  // 부모가 이미 진행 중인 회의를 들고 있으면(다른 단계로 갔다 돌아온 경우) 다시 시작하지
  // 않는다 — IdeationConversationScreen.jsx의 같은 이름 effect와 동일한 세션 재개 규칙을
  // 그대로 따른다(sessionStorage 키를 공유하므로 어느 화면에서 시작했든 이어진다).
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

  // pge/Claude(2026-07-28, 실측 요청: "다른 키워드 추천받기 눌러도 키워드가 안 사라지고
  // 쌓이게") — 이전에는 매 전송 뒤 선택을 항상 초기화했다. "다른 키워드 추천받기"는 이제
  // 키워드 목록에 이어붙이기만 하므로(백엔드 keyword_recommendation 노드가 누적), 사용자가
  // 이미 고른 키워드까지 같이 날아가면 안 된다.
  //
  // pge/Claude(2026-07-28, 실측 요청: "주제 선택하면 이미 2턴 진행돼 있음") — singleTurn:true를
  // 항상 보낸다. 키워드 단계 응답에는 영향이 없고(그 응답들은 라운드테이블에 진입하지
  // 않는다), 주제를 선택해 refinement로 넘어가는 이번 응답에서만 실제로 의미가 있다 — 첫
  // 위원 발언 1건에서 멈춰서 페이지 전환 직후 화면에 이미 여러 턴이 지나가 있지 않게 한다.
  async function handleSend(text) {
    if (!ideationConv || sending) return
    setSending(true)
    setError(null)
    try {
      const data = await replyIdeationConversation(ideationConv.session_id, text, undefined, true)
      setIdeationConv(data)
    } catch (err) {
      setError(classifyIdeationConvError(err))
    } finally {
      setSending(false)
    }
  }

  function handleReselectKeywords() {
    setSelectedKeywordIds(new Set())
    handleSend(KEYWORD_RESELECT_MESSAGE)
  }

  function toggleKeyword(keywordId) {
    setSelectedKeywordIds((prev) => {
      const next = new Set(prev)
      if (next.has(keywordId)) {
        next.delete(keywordId)
      } else {
        if (next.size >= MAX_SELECTED_KEYWORDS) return prev
        next.add(keywordId)
      }
      return next
    })
  }

  function submitKeywordSelection() {
    const options = ideationConv?.keyword_options || []
    const selected = options.filter((opt) => selectedKeywordIds.has(opt.keyword_id)).map((opt) => opt.keyword)
    if (selected.length === 0) return
    setSelectedKeywordIds(new Set())
    handleSend(keywordSelectMessage(selected))
  }

  const phase = ideationConv?.phase
  const keywordOptions = ideationConv?.keyword_options || []
  // pge/Claude(2026-07-28, 실측 요청: "요즘 트렌드/공모전 연관 순서가 막 바뀌는데 고정으로") —
  // keyword_options 배열 안 등장 순서(LLM이 배치마다 다르게 나열)에 기대지 않고, 항상
  // KEYWORD_SOURCE_ORDER 순서로 그룹을 나열한다. 키워드가 없는 출처는 그냥 건너뛴다(빈 칸).
  const groupedKeywords = keywordOptions.reduce((acc, opt) => {
    const source = opt.source || 'contest'
    ;(acc[source] ||= []).push(opt)
    return acc
  }, {})
  const orderedKeywordGroups = KEYWORD_SOURCE_ORDER.filter((source) => (groupedKeywords[source] || []).length > 0).map(
    (source) => [source, groupedKeywords[source]],
  )
  const candidates = ideationConv?.idea_candidates || []
  const selectedKeywordIdsInState = new Set(ideationConv?.selected_keyword_ids || [])
  const keywordSectionActive = phase === 'awaiting_keyword_selection'
  const candidateSectionActive = phase === 'awaiting_candidate_selection'

  return (
    <div style={{ maxWidth: 880, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--purple)', marginBottom: 6 }}>주제 브레인스토밍</div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0 }}>관심 있는 키워드를 골라 주제를 만들어요</h1>
        </div>
        <button type="button" className="btn-ghost" onClick={onBack} style={{ flexShrink: 0 }}>이전 단계로</button>
      </div>

      {error && (
        <div className="card glass" style={{ padding: 14, marginBottom: 16 }}>
          <div style={{ color: 'var(--coral)', fontSize: 14.5 }}>{error.message}</div>
        </div>
      )}

      {starting && !ideationConv && (
        <p className="rb-ideation-notice">키워드를 추천받는 중이에요...</p>
      )}

      {!starting && keywordOptions.length > 0 && (
        <div className="card glass" style={{ padding: 20, marginBottom: 16 }}>
          <div style={{ fontSize: 15.5, fontWeight: 700, marginBottom: 6 }}>
            {keywordSectionActive
              ? `관심 있는 키워드를 선택해 주세요 (최대 ${MAX_SELECTED_KEYWORDS}개)`
              : '이번에 고른 키워드'}
          </div>
          {ideationConv?.trend_search_status && TREND_STATUS_LABEL[ideationConv.trend_search_status] && (
            <div style={{ fontSize: 13, color: TREND_STATUS_LABEL[ideationConv.trend_search_status].color, marginBottom: 14 }}>
              {TREND_STATUS_LABEL[ideationConv.trend_search_status].text}
            </div>
          )}
          {orderedKeywordGroups.map(([source, options]) => (
            <div key={source} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-2)', marginBottom: 8, textTransform: 'uppercase' }}>
                {KEYWORD_SOURCE_LABEL[source] || source}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {options.map((opt) => {
                  const selected = keywordSectionActive
                    ? selectedKeywordIds.has(opt.keyword_id)
                    : selectedKeywordIdsInState.has(opt.keyword_id)
                  const capReached = keywordSectionActive && !selected && selectedKeywordIds.size >= MAX_SELECTED_KEYWORDS
                  const disabled = !keywordSectionActive || capReached
                  const badge = KEYWORD_SOURCE_BADGE[opt.source]
                  return (
                    <button
                      key={opt.keyword_id}
                      type="button"
                      onClick={() => toggleKeyword(opt.keyword_id)}
                      disabled={disabled}
                      title={capReached ? `최대 ${MAX_SELECTED_KEYWORDS}개까지 선택할 수 있어요` : opt.rationale}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6,
                        padding: '8px 14px',
                        borderRadius: 999,
                        border: selected ? '1.5px solid var(--purple)' : '1px solid var(--glass-border)',
                        background: selected ? 'var(--purple-dim)' : 'var(--bg-1)',
                        color: selected ? 'var(--purple)' : 'var(--text-1)',
                        fontWeight: selected ? 700 : 500,
                        fontSize: 14,
                        cursor: !keywordSectionActive ? 'default' : capReached ? 'not-allowed' : 'pointer',
                        opacity: !keywordSectionActive ? (selected ? 1 : 0.45) : capReached ? 0.45 : 1,
                      }}
                    >
                      {badge && (
                        <span
                          style={{
                            fontSize: 11,
                            fontWeight: 700,
                            padding: '2px 6px',
                            borderRadius: 999,
                            color: badge.color,
                            background: badge.bg,
                          }}
                        >
                          {badge.label}
                        </span>
                      )}
                      {opt.keyword}
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
          {keywordSectionActive && (
            <div style={{ display: 'flex', gap: 10, marginTop: 12, flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn-primary"
                disabled={sending || selectedKeywordIds.size === 0}
                onClick={submitKeywordSelection}
              >
                {sending ? '주제를 만드는 중...' : `이 키워드로 주제 만들기 (${selectedKeywordIds.size}/${MAX_SELECTED_KEYWORDS})`}{' '}
                <ArrowRight size={15} />
              </button>
              <button type="button" className="btn-ghost" disabled={sending} onClick={() => handleSend(REGENERATE_MESSAGE)}>
                <RefreshCw size={14} /> 다른 키워드 추천받기
              </button>
            </div>
          )}
        </div>
      )}

      {!starting && candidates.length > 0 && (
        <div>
          <div style={{ fontSize: 13.5, color: 'var(--text-2)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            주제 후보
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: candidates.length > 1 ? 'repeat(auto-fit, minmax(220px, 1fr))' : '1fr',
              gap: 10,
              marginBottom: 14,
            }}
          >
            {candidates.map((c, i) => (
              <CandidateCard
                key={c.candidate_id || i}
                candidate={c}
                index={i}
                disabled={sending || !candidateSectionActive}
                onSelect={(idx) => handleSend(candidateSelectMessage(idx))}
              />
            ))}
          </div>
          {candidateSectionActive && (
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <button type="button" className="btn-ghost" disabled={sending} onClick={() => handleSend(REGENERATE_MESSAGE)}>
                <RefreshCw size={14} /> 다시 추천받기
              </button>
              <button type="button" className="btn-ghost" disabled={sending} onClick={handleReselectKeywords}>
                <RefreshCw size={14} /> 키워드 다시 선택하기
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
