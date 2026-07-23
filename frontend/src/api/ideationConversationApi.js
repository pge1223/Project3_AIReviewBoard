import { API_BASE_URL } from './client'
import { parseNdjsonLine, splitNdjsonLines } from '../pages/board/ideationStreamReducer'

// 용준/Claude(2026-07-20): 개발용 "대화형 아이디어 발전 회의" 프리뷰 API 호출부.
// backend/app/api/routes/ideation_conversation_preview.py(ENABLE_IDEATION_PREVIEW=true일
// 때만 존재하는 라우터)를 그대로 호출한다. 기존 ideationApi.js(배치형)와 같은 컨벤션
// (axios 없이 fetch, authHeaders() 로컬 재정의)을 따른다.
function authHeaders() {
  const token = localStorage.getItem('auth_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function handleResponse(res) {
  const data = await res.json()
  if (!res.ok) {
    throw new Error(data.detail || '아이디어 회의 프리뷰 요청에 실패했습니다.')
  }
  return data
}

export async function startIdeationConversation({
  competitionName,
  competitionDocument,
  userIdea,
  maxRounds = 3,
  useRag = false,
  projectId,
  model,
  // 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입): getApplicationFormAnalysis()
  // 결과의 items를 그대로 넘기면 discussion 프롬프트에 참고 자료로만 주입된다(질문
  // 주제·순서는 안 바뀜). 순수 추가 파라미터 — 비워도 기존 호출부와 동일하게 동작한다.
  applicationFormItems,
}) {
  const res = await fetch(`${API_BASE_URL}/ideation-conversation/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      competition_name: competitionName,
      competition_document: competitionDocument,
      user_idea: userIdea,
      max_rounds: maxRounds,
      use_rag: useRag,
      project_id: useRag ? projectId : undefined,
      model: model || undefined,
      application_form_items: applicationFormItems || undefined,
    }),
  })
  return handleResponse(res)
}

export async function replyIdeationConversation(sessionId, message, model) {
  const res = await fetch(`${API_BASE_URL}/ideation-conversation/${sessionId}/reply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ message, model: model || undefined }),
  })
  return handleResponse(res)
}

// 용준/Claude(2026-07-21, 요청: 실시간 스트리밍): POST /reply/stream(NDJSON, 백엔드
// backend/app/api/routes/ideation_conversation_preview.py::reply_conversation_stream)을
// 호출해 이벤트를 도착하는 즉시 onEvent로 넘긴다. 응답 전체를 기다리는 replyIdeationConversation
// 과 달리 fetch().json()을 쓰지 않고 ReadableStream을 직접 읽는다 — 실제 OpenAI 토큰이
// 여기로 그대로 흘러온다(백엔드가 만들어내는 가짜 지연이 아니다).
//
// TextDecoder({stream:true})로 디코딩해야 하는 이유: 한글은 UTF-8에서 멀티바이트라,
// 네트워크 청크 경계가 한 글자의 바이트 중간에서 잘릴 수 있다 — stream 옵션 없이 매
// 청크를 독립적으로 decode하면 그 잘린 바이트가 깨진 문자(�)로 나온다. stream:true는
// 디코더가 "다음 청크와 이어붙여야 할 수도 있는 불완전한 바이트"를 내부에 들고 있다가
// 다음 decode() 호출에서 이어붙인다.
//
// NDJSON 한 줄이 여러 청크로 나뉘거나 한 청크에 여러 줄이 들어있는 문제는
// splitNdjsonLines()(순수 함수, ideationStreamReducer.js)로 처리한다 — 완성된 줄만 꺼내
// 파싱하고, 끝나지 않은 나머지는 버퍼에 남겨 다음 청크와 이어붙인다.
// 용준/Claude(2026-07-22, 요청: "잠시만" 버튼 — 질문 대상 선택): targetSpeakerId가 주어지면
// (planning_expert/dev_expert/both) 백엔드가 reply_ideation_conversation 대신
// reply_to_interjection으로 라우팅해, 지정한 위원이 먼저 답하고 상대 위원이 반드시 검토하도록
// 회의를 재개한다. content 문자열을 분석해 대상을 추측하지 않고, 버튼 선택 결과를 그대로
// 필드로 전달한다(요청 사항 그대로) — 값이 없으면 기존 /reply/stream과 완전히 동일하게 동작한다
// (하위 호환, optional 필드).
export async function replyIdeationConversationStream(
  sessionId,
  message,
  {
    model,
    signal,
    onEvent,
    targetSpeakerId,
    opinionTargetSpeakerId,
    interruptedSpeakerId,
    interruptedRequestId,
    activeIssueId,
  } = {},
) {
  const res = await fetch(`${API_BASE_URL}/ideation-conversation/${sessionId}/reply/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      message,
      model: model || undefined,
      target_speaker_id: targetSpeakerId || undefined,
      opinion_target_speaker_id: opinionTargetSpeakerId || undefined,
      interrupted_speaker_id: interruptedSpeakerId || undefined,
      interrupted_request_id: interruptedRequestId || undefined,
      active_issue_id: activeIssueId || undefined,
    }),
    signal,
  })
  await readNdjsonStream(res, onEvent)
}

// 가은/Claude(2026-07-22, 요청: 회의 시작 대기 체감 개선 1단계): POST /start/stream —
// startIdeationConversation과 같은 페이로드로 시작하되, 진행 이벤트(phase: "아이디어 후보를
// 만들고 있습니다" 등)를 도착하는 즉시 onEvent로 넘긴다. 최종 결과는 type:"state" 이벤트로
// 온다. 스트리밍 플래그가 꺼져 있으면 404가 나므로 호출부가 기존 동기식 start로 폴백한다
// (reply 쪽과 같은 패턴).
export async function startIdeationConversationStream(
  {
    competitionName,
    competitionDocument,
    userIdea,
    maxRounds = 3,
    useRag = false,
    projectId,
    model,
    applicationFormItems, // 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입) — startIdeationConversation과 동일한 순수 추가 필드.
  },
  { signal, onEvent } = {},
) {
  const res = await fetch(`${API_BASE_URL}/ideation-conversation/start/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      competition_name: competitionName,
      competition_document: competitionDocument,
      user_idea: userIdea,
      max_rounds: maxRounds,
      use_rag: useRag,
      project_id: useRag ? projectId : undefined,
      model: model || undefined,
      application_form_items: applicationFormItems || undefined,
    }),
    signal,
  })
  await readNdjsonStream(res, onEvent)
}

// reply/start 스트리밍이 공유하는 NDJSON 응답 읽기 — 원래 replyIdeationConversationStream
// 안에 있던 로직을 그대로 추출했다(동작 변화 없음).
async function readNdjsonStream(res, onEvent) {
  if (!res.ok || !res.body) {
    let detail
    try {
      const data = await res.json()
      detail = data.detail
    } catch {
      // 본문이 JSON이 아니면(예: 프록시가 끊은 경우) 기본 메시지로 대체한다.
    }
    throw new Error(detail || '아이디어 회의 스트리밍 요청에 실패했습니다.')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const { lines, remainder } = splitNdjsonLines(buffer)
      buffer = remainder
      for (const line of lines) {
        const event = parseNdjsonLine(line)
        if (event) onEvent?.(event)
      }
    }
    // 마지막 남은 버퍼(정상 흐름에서는 서버가 항상 줄 끝에 개행을 붙이므로 비어 있어야
    // 하지만, 방어적으로 마지막 조각도 확인한다) + 디코더에 남아있을 수 있는 잔여 바이트를
    // 마저 비운다.
    buffer += decoder.decode()
    const event = parseNdjsonLine(buffer)
    if (event) onEvent?.(event)
  } finally {
    try {
      reader.releaseLock()
    } catch {
      // 이미 스트림이 취소/종료된 상태면 releaseLock이 예외를 던질 수 있다 — 무시한다.
    }
  }
}

// 용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소): 진행 중인 스트리밍 요청을 취소한다.
// requestId를 생략하면 "지금 활성 요청 아무거나"를 취소한다(멱등 — 활성 요청이 이미 없어도
// 에러 없이 성공 응답을 받는다). 응답의 session_locked=false를 확인한 뒤에만 다음 reply를
// 보내야 세션 lock 409를 피할 수 있다.
export async function cancelIdeationConversation(sessionId, requestId) {
  const res = await fetch(`${API_BASE_URL}/ideation-conversation/${sessionId}/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ request_id: requestId || undefined }),
  })
  return handleResponse(res)
}

export async function finalizeIdeationConversation(sessionId, model) {
  const res = await fetch(`${API_BASE_URL}/ideation-conversation/${sessionId}/finalize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ model: model || undefined }),
  })
  return handleResponse(res)
}

export async function getIdeationConversation(sessionId) {
  const res = await fetch(`${API_BASE_URL}/ideation-conversation/${sessionId}`, {
    headers: { ...authHeaders() },
  })
  return handleResponse(res)
}
