// 작성자: 용준/Claude(2026-07-21)
// 목적: 대화형 아이디어 회의의 NDJSON 스트리밍 이벤트를 다루는 순수 함수 모음. React
//       상태(useState)나 fetch/ReadableStream에 전혀 의존하지 않아, 프런트 테스트 도구가
//       없는 이 저장소에서도 Node로 바로 실행해 검증할 수 있다(scripts/manual-verify-
//       ideation-stream-reducer.mjs 참고).
//
//       IdeationConversationScreen.jsx는 이 모듈의 applyStreamEvent()로 "지금 스트리밍
//       중인(아직 canonical이 아닌) 메시지 목록"만 별도로 들고 있다가, 서버가 최종
//       state 이벤트를 보내면 그 임시 목록을 통째로 버리고 canonical ideationConv를
//       그대로 쓴다 — 그래서 스트리밍 미리보기와 최종 메시지가 절대 동시에 중복 렌더링될
//       수 없다(둘 중 하나만 화면에 남는다).
// import: 없음(순수 함수만).

export function createEmptyStreamState() {
  return { phaseLabel: null, messages: [], requestId: null }
}

// 방어 코드(요청: "중복 버그" 진단) — 실제 OpenAI 호출로 여러 라운드를 재현했지만
// 서버 state/스트리밍 이벤트 계층에서는 message_id 중복을 재현하지 못했다(계획 문서 1번
// 참고). 정확한 트리거를 찾지 못했으므로, 혹시 모를 회귀에 대비해 message_id 기준으로만
// 중복을 제거하는 방어선을 추가한다 — content 비교로 지우는 방식은 서로 다른 라운드에서
// 같은 문장이 실제로 반복될 수 있으므로 절대 쓰지 않는다(요청 그대로).
export function dedupeMessagesById(messages) {
  const seen = new Set()
  const result = []
  for (const m of messages || []) {
    if (!m || seen.has(m.message_id)) continue
    seen.add(m.message_id)
    result.push(m)
  }
  return result
}

// NDJSON 응답은 한 줄(\n으로 끝남)이 이벤트 하나다. fetch의 ReadableStream은 청크 경계가
// 줄 경계와 전혀 무관하게 잘리므로(한 줄이 여러 청크로 나뉘거나, 한 청크에 여러 줄이
// 들어있을 수 있다), 누적 버퍼에서 "완성된 줄들"과 "아직 끝나지 않은 나머지"를 분리하는
// 순수 함수로 뽑아둔다.
export function splitNdjsonLines(buffer) {
  const lines = []
  let rest = buffer
  let newlineIndex = rest.indexOf('\n')
  while (newlineIndex !== -1) {
    lines.push(rest.slice(0, newlineIndex))
    rest = rest.slice(newlineIndex + 1)
    newlineIndex = rest.indexOf('\n')
  }
  return { lines, remainder: rest }
}

// 빈 줄(서버가 안 보내지만 방어적으로)과 JSON 파싱 실패 줄은 건너뛴다 — 스트림 중간에
// 깨진 한 줄 때문에 전체 파싱이 멈추면 안 된다.
export function parseNdjsonLine(line) {
  const trimmed = line.trim()
  if (!trimmed) return null
  try {
    return JSON.parse(trimmed)
  } catch {
    return null
  }
}

// 서버가 실제로 보내는 이벤트 타입(백엔드 ideation_conversation_streaming.py와 반드시
// 맞춰야 한다): phase / message_start / message_delta / message_end / message_reset /
// state / error.
export function applyStreamEvent(state, event) {
  if (!event || typeof event !== 'object' || !event.type) return state

  switch (event.type) {
    // 용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소) — 이번 스트리밍 요청의 request_id를
    // 기록해 둔다. 이후 이벤트(message_start 등)에 실려오는 request_id와 비교해, 혹시라도
    // 이전 요청의 늦은 이벤트가 섞여 들어오면(구조적으로는 요청마다 큐가 분리돼 있어 발생하지
    // 않지만, 방어적으로) 무시할 수 있게 한다.
    case 'request_started':
      return { ...state, requestId: event.request_id || state.requestId }

    case 'phase':
      return { ...state, phaseLabel: event.label || null }

    case 'message_start': {
      if (state.requestId && event.request_id && event.request_id !== state.requestId) return state
      const newMessage = {
        message_id: event.message_id,
        speaker_id: event.speaker_id,
        speaker_name: event.speaker_name,
        // content는 서버에서 실제로 받은 전체 텍스트(표시 속도와 무관하게 즉시 갱신).
        // displayedContent는 advanceDisplay()가 매 프레임 조금씩만 content 쪽으로
        // 따라잡게 하는 값이다 — 이 둘을 분리해야 "실제 수신"과 "화면 표시"의 속도를
        // 독립적으로 제어할 수 있다(타이핑 효과의 핵심).
        content: '',
        displayedContent: '',
        done: false,
        // 'streaming'(수신 중) | 'reviewing'(검증 실패로 재검토 대기) — 신규 메시지는
        // 항상 'streaming'으로 시작한다.
        status: 'streaming',
      }
      // 서버가 supersedes_message_id를 실어 보내면, 이 새 스트림이 어떤 검토 중(reviewing)
      // 말풍선의 재시도인지 명확히 안다(용준/Claude, 2026-07-23, 요청: 스트리밍 UX 버그
      // 수정 — 검증 실패로 사라졌던 말풍선이 최종 state가 올 때까지 화면에서 통째로
      // 비었다가 뒤늦게 나타나던 문제). speaker_id만으로 "같은 위원의 다음 발언이니
      // 재시도겠지"라고 추정하지 않는다 — 다른 위원의 정상적인 다음 발언을 오인해 지우면
      // 안 되므로, 백엔드가 명시적으로 알려준 경우에만 자리를 그대로 이어받는다.
      const supersedesId = event.supersedes_message_id
      if (supersedesId) {
        const idx = state.messages.findIndex((m) => m.message_id === supersedesId)
        if (idx !== -1) {
          const messages = state.messages.slice()
          messages[idx] = newMessage
          return { ...state, phaseLabel: null, messages }
        }
      }
      return { ...state, phaseLabel: null, messages: [...state.messages, newMessage] }
    }

    case 'message_delta':
      if (!event.delta) return state
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.message_id === event.message_id ? { ...m, content: m.content + event.delta } : m,
        ),
      }

    case 'message_end':
      return {
        ...state,
        messages: state.messages.map((m) => (m.message_id === event.message_id ? { ...m, done: true } : m)),
      }

    case 'message_reset':
      // 용준/Claude(2026-07-23, 요청: 스트리밍 UX 버그 수정) — 구조화 응답 검증 실패로
      // 재시도(또는 grounding 재시도, 또는 재시도 없이 safe fallback 대기)할 때 서버가
      // 보낸다. 예전에는 말풍선을 배열에서 완전히 지웠는데, 그러면 재시도 스트림이 시작
      // 되기 전까지(또는 fallback이 담긴 canonical state가 올 때까지) 화면이 통째로 비어
      // "발언이 사라졌다가 회의 끝에 뒤늦게 나타나는" 것처럼 보였다. 이제는 말풍선을 지우지
      // 않고 'reviewing' 상태로만 전환해 흐리게 표시하고, displayedContent를 content
      // 끝까지 스냅해 타이핑 커서가 중간에 멈춘 것처럼 보이지 않게 한다 — 실제 교체(같은
      // 자리에 재시도 말풍선을 잇는 것, 또는 canonical state로 완전히 대체하는 것)는
      // message_start의 supersedes_message_id 처리와 호출부의 state 처리가 각각 담당한다.
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.message_id === event.message_id
            ? {
                ...m,
                status: 'reviewing',
                reviewReason: event.reason || null,
                willRetry: event.will_retry !== false,
                displayedContent: m.content,
                done: true,
              }
            : m,
        ),
      }

    // 'state'(최종 canonical state)와 'error'는 이 리듀서가 다루지 않는다 — 호출부가
    // 직접 처리해서 ideationConv/오류 배너로 반영하고, 스트리밍 임시 상태는 통째로
    // 초기화(createEmptyStreamState())한다.
    default:
      return state
  }
}

// 아래 세 함수는 "실제 LLM 델타 수신"(content)과 "화면에 글자를 드러내는 속도"
// (displayedContent)를 분리하기 위한 순수 함수다. React state/타이머에 전혀 의존하지
// 않으므로 호출부(IdeationConversationScreen.jsx)가 requestAnimationFrame 루프 안에서
// 매 프레임 advanceDisplay를 부르기만 하면 된다 — "완성된 응답을 다 받은 뒤 재생하는
// 가짜 타이핑"이 아니라, content 자체는 서버 델타가 도착하는 즉시 이미 갱신되어 있고
// displayedContent만 그 뒤를 조금씩 따라간다.

// 메시지별로 아직 화면에 드러내지 못한 글자 수(content.length - displayedContent.length,
// 음수면 0)를 모두 더한다 — 출력 속도를 동적으로 조절하는 데 쓴다(밀린 글자가 많을수록
// 한 프레임에 더 많이 드러내 지연이 누적되지 않게 한다).
export function pendingCharCount(state) {
  return state.messages.reduce((sum, m) => sum + Math.max(0, (m.content?.length || 0) - (m.displayedContent?.length || 0)), 0)
}

// 모든 메시지가 content 끝까지 드러났는지(더 이상 밀린 글자가 없는지) 확인한다. 메시지가
// 하나도 없으면(스트리밍 대상 메시지가 없던 요청) 자명하게 true다 — 호출부가 이 값으로
// "이제 canonical state로 교체해도 되는지"를 판단한다. message_end(done) 여부는 일부러
// 보지 않는다 — 요청 사항("message_end가 와도 남은 글자 큐를 끝까지 표시")대로, 서버가
// 이미 끝을 알렸어도 화면 표시가 따라잡기 전에는 완료로 치지 않기 위함이다.
export function isFullyDisplayed(state) {
  return state.messages.every((m) => (m.displayedContent?.length || 0) >= (m.content?.length || 0))
}

// 한 프레임에 메시지별로 최대 charsPerTick 글자씩 displayedContent를 content 쪽으로
// 전진시킨다. 이미 따라잡은 메시지는 그대로 반환해(불필요한 재렌더 방지) 변화가 없으면
// 원래 state 참조를 그대로 돌려준다.
export function advanceDisplay(state, charsPerTick) {
  let changed = false
  const messages = state.messages.map((m) => {
    const content = m.content || ''
    const displayed = m.displayedContent || ''
    if (displayed.length >= content.length) return m
    changed = true
    return { ...m, displayedContent: content.slice(0, displayed.length + charsPerTick) }
  })
  return changed ? { ...state, messages } : state
}

// pendingCharCount가 클수록 한 프레임에 더 많은 글자를 드러내 밀린 출력이 쌓이지 않게
// 한다(요청: "출력 큐가 너무 길어지면 출력 속도를 동적으로 높여 지연 누적 방지"). 값은
// 실험적으로 고른 임계치다 — 정확한 튜닝보다는 "밀릴수록 빨라진다"는 방향성이 핵심.
export function charsPerTickFor(pendingChars) {
  if (pendingChars > 300) return 12
  if (pendingChars > 120) return 6
  if (pendingChars > 40) return 3
  if (pendingChars > 10) return 2
  return 1
}
