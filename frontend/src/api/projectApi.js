import { API_BASE_URL, parseApiResponse, clearExpiredSession } from './client'

function authHeaders() {
  const token = localStorage.getItem('auth_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function getProjects() {
  const res = await fetch(`${API_BASE_URL}/projects/`, {
    headers: { ...authHeaders() },
  })
  return parseApiResponse(res, '프로젝트 목록을 불러오지 못했습니다.')
}

export async function getProject(projectId) {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}`, {
    headers: { ...authHeaders() },
  })
  return parseApiResponse(res, '프로젝트를 불러오지 못했습니다.')
}

export async function createProject({ title, doc_type, description, flow_mode }) {
  const res = await fetch(`${API_BASE_URL}/projects/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ title, doc_type, description, flow_mode }),
  })
  return parseApiResponse(res, '프로젝트를 생성하지 못했습니다.')
}

export async function updateProject(projectId, { title, description, flow_mode }) {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ title, description, flow_mode }),
  })
  return parseApiResponse(res, '프로젝트를 수정하지 못했습니다.')
}

// PRJ-004 — 벡터 청크/문서/회의/프로젝트를 백엔드가 한 번에 정리한다(projects.py 참고).
export async function deleteProject(projectId) {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}`, {
    method: 'DELETE',
    headers: { ...authHeaders() },
  })
  return parseApiResponse(res, '프로젝트를 삭제하지 못했습니다.')
}

// committee: 선택한 멘토 persona_id 2~4개(MentorSelectionPage). 생략하면 백엔드가
// rubric_mapping의 전체 committee를 그대로 쓴다(기존 호출 호환).
// progressToken: 진짜 진행률 폴링용(getAnalyzeProgress) — 호출부가 POST 전에 미리
// crypto.randomUUID() 등으로 만들어 넘긴다. analyze()는 완료될 때까지 응답이 없는 동기
// 호출이라, 그 사이 상태를 보려면 별도 토큰이 필요하다(meeting_id는 완료 후에만 앎).
export async function analyzeProject(projectId, committee, progressToken) {
  const body = committee || progressToken ? { committee, progress_token: progressToken } : undefined
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: body ? JSON.stringify(body) : undefined,
  })
  return parseApiResponse(res, '분석을 시작하지 못했습니다.')
}

// FeedbackProgressPage가 analyzeProject() 진행 중 주기적으로 불러 실제 진행 상황(위원별
// 검토/채점/위원장 종합 단계)을 읽는다. 토큰을 아직 서버가 모르거나(POST가 아직 도착 전)
// 이미 끝나서 지워졌으면 백엔드가 빈 기본값을 준다 — 최종 완료 판단은 이 값이 아니라
// analyzeProject()의 resolve/reject로 한다.
// 가은/Claude(2026-07-21): 폴링 호출이라 실패해도 조용히 null만 반환하던 기존 계약은
// 그대로 두되, 401(토큰 만료)만은 로그인 화면으로 보내도록 따로 잡는다.
export async function getAnalyzeProgress(projectId, progressToken) {
  const res = await fetch(
    `${API_BASE_URL}/projects/${projectId}/analyze/progress?token=${encodeURIComponent(progressToken)}`,
    { headers: { ...authHeaders() } },
  )
  if (res.status === 401) clearExpiredSession()
  if (!res.ok) return null
  return res.json()
}

// STEP7 "대화형 피드백" 화면 — 저장된 회의 결과를 근거로 위원장이 후속 질문에 답한다.
// history: 이번 세션에서 오간 이전 질문/답변([{question, answer}, ...]) — 서버가 대화를
// 저장하지 않는 stateless 호출이라 문맥 유지하려면 매번 그대로 다시 넘겨야 한다.
export async function askCommittee(projectId, question, history) {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ question, history }),
  })
  return parseApiResponse(res, '답변을 받아오지 못했습니다.')
}

// 가은/Claude(2026-07-17): "결과 정리"(ProjectDetailPage)가 위원장 종합(백그라운드로
// 늦게 끝남)을 기다리는 동안 폴링용으로 쓴다 — RPT-001 기존 엔드포인트 재사용, 새
// 엔드포인트 아님. 최신 Mongo 회의 기록을 그대로 반환하므로 백그라운드 작업이
// chair_summary를 patch하면 다음 폴링에서 바로 보인다.
export async function getProjectReport(projectId) {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}/report`, {
    headers: { ...authHeaders() },
  })
  return parseApiResponse(res, '결과를 불러오지 못했습니다.')
}

// 경이/Claude(2026-07-23, RPT-004 C단계): 버전 비교 — 최근 2개 회의(직전 vs 이번 수정본)를
// build_revision_comparison으로 비교한 결과. 회의가 1개뿐이면 {available:false}. 종합 리포트
// (VersionTrackerTestPage)가 v1.0→v1.1 이전/현재 막대·해결/신규/잔존 뱃지를 그릴 때 쓴다.
export async function getProjectComparison(projectId) {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}/comparison`, {
    headers: { ...authHeaders() },
  })
  return parseApiResponse(res, '버전 비교 결과를 불러오지 못했습니다.')
}

// 가은/Claude(2026-07-21): "내 프로젝트"에서 이어서 열 때, 분석이 이미 끝난
// 프로젝트인지(회의 존재 여부) 싸게 확인하는 용도 — MTG-005 기존 엔드포인트.
// getProjectReport()와 달리 impl_guides(LLM 호출 포함)를 계산하지 않아 존재 확인
// 용도로는 이쪽이 맞다. 회의가 없으면 백엔드가 404를 던진다.
export async function getLatestMeeting(projectId) {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}/meetings/latest`, {
    headers: { ...authHeaders() },
  })
  return parseApiResponse(res, '회의 결과가 없습니다.')
}

// STEP4 "공모전 분석" 화면 — 문서 성격 태그 + 추천 멘토 후보(도메인 고정 4명 + LLM이 붙인
// fit_tag)를 가져온다. 실제 OpenAI 호출 1회(짧은 프롬프트라 수 초 내).
export async function getMentorCandidates(projectId) {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}/mentor-candidates`, {
    method: 'POST',
    headers: { ...authHeaders() },
  })
  return parseApiResponse(res, '추천 멘토를 불러오지 못했습니다.')
}
