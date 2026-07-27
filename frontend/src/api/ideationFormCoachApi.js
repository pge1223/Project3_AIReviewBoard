import { API_BASE_URL } from './client'

function authHeaders() {
  const token = localStorage.getItem('auth_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function handleResponse(res) {
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || '신청 양식 코치 요청에 실패했습니다.')
  return data
}

export async function startIdeationFormCoach({
  competitionName,
  competitionDocument,
  selectedIdea,
  ideaCanvas,
  applicationFormItems,
  legacyContext,
  projectId,
  useRag,
  model,
}) {
  const res = await fetch(`${API_BASE_URL}/ideation-form-coach/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    signal: AbortSignal.timeout(120000),
    body: JSON.stringify({
      competition_name: competitionName,
      competition_document: competitionDocument,
      selected_idea: selectedIdea,
      idea_canvas: ideaCanvas || {},
      application_form_items: applicationFormItems,
      legacy_context: legacyContext || {},
      project_id: projectId || undefined,
      use_rag: !!useRag,
      model: model || undefined,
    }),
  })
  return handleResponse(res)
}

export async function replyIdeationFormCoach(sessionId, message, model) {
  const res = await fetch(`${API_BASE_URL}/ideation-form-coach/${sessionId}/reply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    signal: AbortSignal.timeout(120000),
    body: JSON.stringify({ message, model: model || undefined }),
  })
  return handleResponse(res)
}

export async function finalizeIdeationFormCoach(sessionId, model) {
  const res = await fetch(`${API_BASE_URL}/ideation-form-coach/${sessionId}/finalize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ model: model || undefined }),
  })
  return handleResponse(res)
}

export async function updateIdeationFormDraft(sessionId, fieldId, value) {
  const res = await fetch(`${API_BASE_URL}/ideation-form-coach/${sessionId}/draft/${fieldId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ value }),
  })
  return handleResponse(res)
}

export async function getIdeationFormCoach(sessionId) {
  const res = await fetch(`${API_BASE_URL}/ideation-form-coach/${sessionId}`, {
    headers: { ...authHeaders() },
  })
  return handleResponse(res)
}
