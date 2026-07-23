// 작성자: 용준/Claude(2026-07-21)
// 목적: /board "작성 전 → 주제 발굴" 흐름을 실제 대화형 아이디어 회의 API
//       (ideationConversationApi.js → backend/app/api/routes/ideation_conversation_preview.py)에
//       연결하기 위한 순수 함수 모음. IdeationConversationScreen.jsx가 이 파일의 함수를
//       불러 쓴다 — 프런트 테스트 체계가 없는 저장소라 로직을 컴포넌트에서 분리해 최소한
//       수동/향후 유닛 테스트가 가능한 형태로 둔다(JSX나 React 상태에 의존하지 않음).
// import: 없음(순수 함수만).

// 공모전 분석 결과(documentApi.js::getAnnouncementAnalysis 응답, AnalysisScreen이 이미
// 쓰는 것과 같은 스키마 — has_announcement/official_facts/strategic_analysis/evidence/
// source_document_names/announcement_title, backend/app/schemas/document.py 기준)에서
// 실제 값이 존재하는 필드만 골라 회의 시작 API의 competition_document로 보낼 텍스트를
// 만든다. 없는 필드는 지어내지 않고 통째로 생략한다.
const MAX_COMPETITION_DOCUMENT_LENGTH = 2000

function truncateList(list, max) {
  if (!Array.isArray(list)) return []
  return list.filter((v) => typeof v === 'string' && v.trim()).slice(0, max)
}

export function competitionNameFrom(analysis) {
  const title = analysis?.announcement_title?.trim()
  if (title) return title
  const sourceName = analysis?.source_document_names?.[0]
  if (sourceName) return sourceName
  // 백엔드 start API는 competition_name을 필수로 요구한다 — 공고문을 하나도 등록하지
  // 않은 세션(has_announcement=false)에서도 회의를 시작할 수 있어야 하므로, AnalysisScreen이
  // 이미 쓰는 것과 같은 용어("공고문 미등록")로 안내성 기본값을 채운다(성과 수치·사례를
  // 지어내는 것과는 다르다 — 값이 없다는 사실 자체를 표현하는 라벨일 뿐).
  return '미등록 공모전'
}

export function buildCompetitionDocumentText(analysis) {
  if (!analysis || !analysis.has_announcement) return ''
  const facts = analysis.official_facts || {}
  const strategy = analysis.strategic_analysis || {}
  const sections = []

  sections.push(`[공모전명] ${competitionNameFrom(analysis)}`)

  if (strategy.core_intent?.trim()) {
    sections.push(`[핵심 과제]\n${strategy.core_intent.trim()}`)
  }

  const evaluationCriteria = truncateList(facts.evaluation_criteria, 12)
  if (evaluationCriteria.length > 0) {
    sections.push(`[심사 기준 · 평가에서 갈리는 지점]\n- ${evaluationCriteria.join('\n- ')}`)
  }

  const musts = truncateList(
    [
      ...(facts.submission_requirements || []),
      ...(facts.application_review_conditions || []),
      ...(facts.disqualification_rules || []),
    ],
    12,
  )
  if (musts.length > 0) {
    sections.push(`[반드시 지켜야 할 조건]\n- ${musts.join('\n- ')}`)
  }

  const eligibility = truncateList(facts.eligibility, 5)
  if (eligibility.length > 0) {
    sections.push(`[지원 대상과 제한 조건]\n- ${eligibility.join('\n- ')}`)
  }

  const keyDates = truncateList(facts.key_dates, 6)
  if (keyDates.length > 0) {
    sections.push(`[주요 일정]\n- ${keyDates.join('\n- ')}`)
  }

  const benefits = truncateList(facts.selection_benefits, 8)
  if (benefits.length > 0) {
    sections.push(`[선정 혜택]\n- ${benefits.join('\n- ')}`)
  }

  // 수상작·유사사례 경향: 이 시스템은 그 데이터 소스 자체가 없다(AnalysisScreen과 동일
  // 문구) — 지어내지 않고 자료 미확보 사실만 전달한다.
  sections.push('[수상작·유사사례 경향]\n자료 미확보 — 유사 공모전 사례 기반 분석은 아직 지원하지 않습니다.')

  const mainPoints = truncateList(
    [...(strategy.winning_points || []), ...(strategy.recommended_direction || []), ...(strategy.risk_flags || [])],
    6,
  )
  if (mainPoints.length > 0) {
    sections.push(`[공고문에서 확인된 주요 내용]\n- ${mainPoints.join('\n- ')}`)
  }

  const text = sections.join('\n\n')
  if (text.length <= MAX_COMPETITION_DOCUMENT_LENGTH) return text
  return `${text.slice(0, MAX_COMPETITION_DOCUMENT_LENGTH - 1)}…`
}

// RAG 사용 여부는 "유효한 projectId와 색인 완료 상태가 확인될 때만 true"(요청 사항 그대로)
// — documents 배열의 status는 EntryScreen이 이미 쓰는 값('done'/'embedding'/'warning'/
// 'error')이다. 색인이 끝났다고 보는 상태는 'done'뿐이다(embedding=진행 중, warning=
// 색인은 됐지만 내용이 비어있음, error=실패 — 어느 쪽도 RAG에 쓸 수 있는 완료 상태가
// 아니다).
export function resolveUseRag(projectId, criteriaDocuments) {
  if (!projectId) return false
  return (criteriaDocuments || []).some((doc) => doc.status === 'done')
}

// 화면에 노출되는 전문가/진행자/사용자 표시 메타 — 실제 speaker_id는
// ai/meeting/graph/ideation_conv_nodes.py(_speaker_fields는 persona_cards.json에서
// display_name/role을 가져오지만, speaker_id 자체는 호출부가 "planning_expert"/
// "dev_expert"/"ideation_facilitator"/"user" 고정값으로 넘긴다)와
// ai/meeting/tests/test_ideation_conv_graph.py·test_ideation_discovery_graph.py의 스크립트
// 스텁이 검증하는 값 그대로다. badgeClass는 ReviewBoardPrototype.jsx Shell이 이미 정의한
// .badge.purple/.coral/.green 클래스를 재사용한다(새 색을 만들지 않는다).
export const SPEAKER_META = {
  planning_expert: { label: '기획 위원', badgeClass: 'purple', align: 'left' },
  dev_expert: { label: '개발 위원', badgeClass: 'coral', align: 'left' },
  ideation_facilitator: { label: '진행자', badgeClass: 'green', align: 'left' },
  user: { label: '나', badgeClass: null, align: 'right' },
}

export function speakerMetaFor(message) {
  return (
    SPEAKER_META[message?.speaker_id] || { label: message?.speaker_name || '알 수 없음', badgeClass: null, align: 'left' }
  )
}

export const FEASIBILITY_LABEL = { high: '높음', medium: '보통', low: '낮음' }

// API 응답의 phase(영문 상태 slug)를 사용자에게 보여줄 한국어 라벨로 바꾼다 — 영문
// phase를 화면에 그대로 노출하지 않기 위함이다. candidate_generation/전문가 의견
// 생성 중/최종 결과 생성 중은 실제 phase 값이 아니라(그래프가 정지하는 지점만 API로
// 노출되므로) starting/sending/finalizing 같은 로컬 진행 상태에 대응한다.
// 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — awaiting_planning_answer/
// awaiting_developer_answer(1:1 인터뷰 전용 phase)는 새 세션에서는 더 이상 도달하지 않지만,
// 이 변경 이전에 시작된 세션(인메모리 세션, TTL 30분)이 여전히 이 phase로 남아 있을 수
// 있어 라벨은 그대로 둔다(하위 호환).
const PHASE_LABEL_KO = {
  awaiting_candidate_selection: '후보 선택 대기',
  awaiting_planning_answer: '기획 위원 답변 대기',
  awaiting_developer_answer: '개발 위원 답변 대기',
  awaiting_user_decision: '위원 논의 완료 · 의견은 선택 사항',
  discussion_complete: '위원 논의 완료',
  finalized: '완료',
  failed: '실패',
}

export function statusLabelFor({ phase, starting, sending, finalizing, interrupting }) {
  // 용준/Claude(2026-07-22, 요청: "잠시만" 버튼) — 취소 확인을 기다리는 동안 보여줄 상태.
  if (interrupting) return '회의를 잠시 멈추고 있어요...'
  if (starting) return '아이디어 후보 생성 중'
  if (finalizing) return '최종 결과 생성 중'
  if (sending) {
    if (phase === 'awaiting_developer_answer' || phase === 'awaiting_user_decision') return '위원들이 논의하는 중'
    return '응답을 준비하는 중'
  }
  return PHASE_LABEL_KO[phase] || phase
}

// phase별로 사용자가 지금 무엇을 더 해야 하는지 안내하는 문구(요청: "비활성 상태에서는
// 사용자가 무엇을 더 해야 하는지 안내"). awaiting_user_decision(요청 6번, 라운드테이블이
// 한 라운드를 마치고 멈춘 지점)은 확정 버튼이 활성화되지만, "매 발언마다 답할 의무가
// 없다"는 것을 안내 문구로도 드러낸다 — 답하지 않아도 확정하거나 자유롭게 의견을 남길 수
// 있다.
export function nextActionGuideFor(phase) {
  switch (phase) {
    case 'awaiting_candidate_selection':
      return '후보를 선택하거나("1번"), 결합("1번과 2번 결합"), 다시 추천을 요청해야 다음 단계로 진행할 수 있어요.'
    case 'awaiting_planning_answer':
      return '기획 위원의 질문에 답변해야 개발 위원의 질문으로 넘어갈 수 있어요.'
    case 'awaiting_developer_answer':
      return '개발 위원의 질문에 답변해야 두 위원의 의견을 볼 수 있어요.'
    case 'awaiting_user_decision':
      return '위원들의 논의가 한 라운드 끝났어요. 답할 의무는 없어요 — 의견이 있으면 남기고, 없으면 바로 확정할 수 있어요.'
    default:
      return ''
  }
}

// 사용자가 후보 카드의 "이 후보 선택" 버튼을 눌렀을 때 reply API로 보낼 메시지.
// ideation_conv_discovery.py::_NUMERIC_SELECT_RE가 "1", "1번", "1번째" 형태를 코드로
// 결정적으로(LLM 호출 없이) 처리하므로 "n번" 형태로 보낸다 — 프런트가 후보를 자체
// 결합/해석하지 않고 항상 백엔드 candidate_selection 로직을 거치게 하기 위함이다.
export function candidateSelectMessage(index) {
  return `${index + 1}번`
}

// 자유 입력 없이 바로 보낼 수 있는 후보 단계 빠른 요청 문구 — 자유 입력으로도 동일하게
// 지원되지만(요청 사항), 버튼으로도 같은 reply API를 타도록 제공한다.
export const REGENERATE_MESSAGE = '다시 추천'
export const EXPERT_RECOMMEND_MESSAGE = '전문가 추천'

// 응답 캡션은 structured의 speaker_id만 신뢰하지 않고 실제 message_id가 가리키는 메시지와
// 교차 검증한다. 불일치/누락/자기 참조면 잘못된 관계를 화면에 표시하지 않는다.
export function resolveRespondingToSpeakerId(message, allMessages = []) {
  const targetMessageId = message?.structured?.responding_to_message_id
  const declaredSpeakerId = message?.structured?.responding_to_speaker_id
  if (!targetMessageId || !declaredSpeakerId) return null
  const targetMessage = allMessages.find((candidate) => candidate?.message_id === targetMessageId)
  const actualSpeakerId = targetMessage?.speaker_id
  if (!actualSpeakerId || actualSpeakerId !== declaredSpeakerId || actualSpeakerId === message?.speaker_id) {
    return null
  }
  return actualSpeakerId
}

// fetch가 아예 실패한 경우(TypeError: Failed to fetch 등, 서버 자체에 연결할 수 없음),
// ENABLE_IDEATION_PREVIEW=false로 라우터가 등록되지 않아 모든 요청이 404 "Not Found"를
// 주는 경우, 세션이 만료·소실된 경우(개발용 인메모리 저장소라 서버 재시작 시에도
// 똑같이 404), 그 외(LLM 호출 실패 등 502)를 구분해 사용자에게 다른 안내를 보여준다.
// handleResponse()(ideationConversationApi.js)가 매번 `new Error(data.detail || ...)`로
// 던지므로, 이 함수는 그 message 문자열만 보고 분류한다 — 백엔드 응답 바디를 다시 파싱하지
// 않는다.
export function classifyIdeationConvError(err) {
  const message = err?.message || ''
  if (err?.name === 'TypeError' || /Failed to fetch|NetworkError|network error/i.test(message)) {
    return {
      type: 'network',
      message: '백엔드에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.',
    }
  }
  if (message === 'Not Found') {
    return {
      type: 'disabled',
      message: '아이디어 회의 API가 비활성화되어 있습니다. 백엔드의 ENABLE_IDEATION_PREVIEW 설정을 확인해 주세요.',
    }
  }
  if (message.includes('세션을 찾을 수 없') || message.includes('만료')) {
    return {
      type: 'session_expired',
      message: '회의 세션이 만료되었거나 서버가 재시작되었습니다. 새로 시작해 주세요.',
    }
  }
  return {
    type: 'llm_failure',
    message: message || '요청 처리 중 알 수 없는 오류가 발생했습니다.',
  }
}
