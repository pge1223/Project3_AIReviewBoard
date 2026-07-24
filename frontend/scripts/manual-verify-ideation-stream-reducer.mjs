// 작성자: 용준/Claude(2026-07-21)
// 목적: 프런트 테스트 프레임워크(vitest/jest 등)가 아직 없는 저장소라, NDJSON 파서와
//       스트림 리듀서(src/pages/board/ideationStreamReducer.js)를 새 테스트 러너 없이
//       Node 내장 assert만으로 검증하는 수동 스크립트다. CI에 자동으로 물리지 않는다 —
//       실행: `node frontend/scripts/manual-verify-ideation-stream-reducer.mjs`
import assert from "node:assert/strict";
import {
  advanceDisplay,
  applyStreamEvent,
  charsPerTickFor,
  createEmptyStreamState,
  dedupeMessagesById,
  isFullyDisplayed,
  parseNdjsonLine,
  pendingCharCount,
  splitNdjsonLines,
} from "../src/pages/board/ideationStreamReducer.js";
import { resolveRespondingToSpeakerId } from "../src/pages/board/ideationConversationHelpers.js";

function test(name, fn) {
  try {
    fn();
    console.log(`OK   ${name}`);
  } catch (err) {
    console.error(`FAIL ${name}`);
    throw err;
  }
}

test("splitNdjsonLines: 완성된 줄과 남은 버퍼를 분리한다", () => {
  const { lines, remainder } = splitNdjsonLines('{"a":1}\n{"b":2}\n{"c"');
  assert.deepEqual(lines, ['{"a":1}', '{"b":2}']);
  assert.equal(remainder, '{"c"');
});

test("splitNdjsonLines: 완성된 줄이 하나도 없으면 lines가 빈 배열이다", () => {
  const { lines, remainder } = splitNdjsonLines('{"partial":"tex');
  assert.deepEqual(lines, []);
  assert.equal(remainder, '{"partial":"tex');
});

test("splitNdjsonLines: 한 청크에 여러 줄이 섞여도 전부 분리한다", () => {
  const { lines, remainder } = splitNdjsonLines("a\nb\nc\nd");
  assert.deepEqual(lines, ["a", "b", "c"]);
  assert.equal(remainder, "d");
});

test("parseNdjsonLine: 빈 줄/깨진 JSON은 null", () => {
  assert.equal(parseNdjsonLine(""), null);
  assert.equal(parseNdjsonLine("   "), null);
  assert.equal(parseNdjsonLine("{invalid"), null);
  assert.deepEqual(parseNdjsonLine('{"type":"phase","label":"진행 중"}'), {
    type: "phase",
    label: "진행 중",
  });
});

test("applyStreamEvent: phase -> message_start -> message_delta(여러 번) -> message_end 순서로 누적된다", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "phase", label: "답변의 의도를 확인하고 있습니다" });
  assert.equal(state.phaseLabel, "답변의 의도를 확인하고 있습니다");

  state = applyStreamEvent(state, {
    type: "message_start",
    message_id: "STREAM-1",
    speaker_id: "planning_expert",
    speaker_name: "기획 전문가",
  });
  assert.equal(state.phaseLabel, null); // 메시지가 시작되면 단계 안내 문구는 사라진다.
  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].content, "");
  assert.equal(state.messages[0].done, false);

  state = applyStreamEvent(state, { type: "message_delta", message_id: "STREAM-1", delta: "사용자가 선택한 " });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "STREAM-1", delta: "후보는 AI 기반 " });
  assert.equal(state.messages[0].content, "사용자가 선택한 후보는 AI 기반 ");

  state = applyStreamEvent(state, { type: "message_end", message_id: "STREAM-1" });
  assert.equal(state.messages[0].done, true);
  assert.equal(state.messages[0].content, "사용자가 선택한 후보는 AI 기반 "); // 끝나도 내용은 유지된다.
});

test("applyStreamEvent: 한글이 여러 delta로 쪼개져도 순서대로 이어붙는다", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "dev_expert", speaker_name: "개발 전문가" });
  const parts = ["업", "무", " 자", "동", "화", " AI ", "스", "킬"];
  for (const p of parts) {
    state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: p });
  }
  assert.equal(state.messages[0].content, "업무 자동화 AI 스킬");
});

// 용준/Claude(2026-07-23, 요청: 스트리밍 UX 버그 수정 — 검증 실패로 말풍선이 사라졌다가
// 회의가 끝난 뒤에야 확정 발언이 뒤늦게 나타나는 문제) 이전에는 message_reset이 해당
// 메시지를 배열에서 완전히 지웠다. 그러면 재시도 스트림이 시작되기 전까지(또는 fallback을
// 담은 canonical state가 올 때까지) 화면이 통째로 비어 보였다. 이제는 지우지 않고
// status:'reviewing'으로만 전환한다 — 아래 테스트들이 그 수명주기 전체를 검증한다.

test("applyStreamEvent: message_reset은 메시지를 지우지 않고 'reviewing' 상태로 전환한다(화면이 비지 않음)", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: "잘못된 절반짜리 응답" });
  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].status, "streaming");

  state = applyStreamEvent(state, {
    type: "message_reset",
    message_id: "M1",
    reason: "missing_or_empty_field:judgment_or_reason",
    will_retry: true,
  });
  // 지워지지 않고 그대로 남아있다 — 화면이 빈 상태로 보이지 않는다.
  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].status, "reviewing");
  assert.equal(state.messages[0].willRetry, true);
  assert.equal(state.messages[0].reviewReason, "missing_or_empty_field:judgment_or_reason");
  // 타이핑 커서가 중간에 멈춘 것처럼 보이지 않도록 즉시 끝까지 스냅한다.
  assert.equal(state.messages[0].displayedContent, state.messages[0].content);
});

// ---------------------------------------------------------------------------
// 필수 테스트 케이스 1~5 (요청: 스트리밍 UX 버그 수정)
// ---------------------------------------------------------------------------

test("케이스 1: 정상 스트리밍(start -> delta -> end) — 중복/공백 없이 메시지 하나만 남는다", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "planning_expert", speaker_name: "기획 전문가", request_id: "REQ-1" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: "정상 발언입니다" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "M1" });

  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].status, "streaming");
  assert.equal(state.messages[0].done, true);
  assert.equal(state.messages[0].content, "정상 발언입니다");
  // 'state' 이벤트는 호출부(IdeationConversationScreen.jsx)가 처리해 canonical로
  // 원자적으로 교체한다 — 리듀서 자체는 손대지 않는다(기존 계약, 아래에서 별도 확인).
});

test("케이스 2: 구조화 검증 1차 실패 후 재시도 성공 — 첫 초안은 reviewing으로 바뀐 뒤 재시도 말풍선으로 '같은 자리'에서 교체된다(최종 1건만)", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: "필수 필드 누락된 초안" });
  assert.equal(state.messages.length, 1);

  state = applyStreamEvent(state, { type: "message_reset", message_id: "M1", reason: "missing_or_empty_field:judgment_or_reason", will_retry: true });
  assert.equal(state.messages.length, 1); // 화면이 비지 않는다.
  assert.equal(state.messages[0].status, "reviewing");

  // 백엔드는 재시도 message_start에 supersedes_message_id를 실어 보낸다.
  state = applyStreamEvent(state, { type: "message_start", message_id: "M2", speaker_id: "planning_expert", speaker_name: "기획 전문가", supersedes_message_id: "M1" });
  // 배열 길이가 늘지 않는다 — 같은 자리(같은 인덱스)에서 교체됐다(추가가 아니다).
  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].message_id, "M2");
  assert.equal(state.messages[0].status, "streaming");

  state = applyStreamEvent(state, { type: "message_delta", message_id: "M2", delta: "재시도로 정상 생성된 발언" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "M2" });

  assert.equal(state.messages.length, 1); // 최종적으로 메시지는 한 건만 존재한다.
  assert.equal(state.messages[0].content, "재시도로 정상 생성된 발언");
  assert.equal(state.messages[0].done, true);
});

test("케이스 3: 두 번 실패 후 safe fallback 대기 — 중간에 빈 대화창이 되지 않고, 재시도가 없다는 신호(will_retry=false)를 그대로 유지한다", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "dev_expert", speaker_name: "개발 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: "1차 실패 초안" });
  state = applyStreamEvent(state, { type: "message_reset", message_id: "M1", reason: "missing_or_empty_field:judgment_or_reason", will_retry: true });
  assert.equal(state.messages.length, 1); // 1차 실패 후에도 화면이 비지 않는다.

  state = applyStreamEvent(state, { type: "message_start", message_id: "M2", speaker_id: "dev_expert", speaker_name: "개발 전문가", supersedes_message_id: "M1" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M2", delta: "2차 실패 초안" });
  // 두 번째 실패는 더 이상 재시도가 없다(will_retry=false) — 서버가 safe fallback을 만들어
  // canonical state로 보낼 때까지 검토 중 표시를 유지해야 한다.
  state = applyStreamEvent(state, { type: "message_reset", message_id: "M2", reason: "missing_or_empty_field:judgment_or_reason", will_retry: false });

  assert.equal(state.messages.length, 1); // 여전히 빈 대화창이 아니다.
  assert.equal(state.messages[0].status, "reviewing");
  assert.equal(state.messages[0].willRetry, false);
  // 이 시점 이후 더 이상 message_start가 오지 않는다 — IdeationConversationScreen.jsx가
  // 최종 'state' 이벤트를 받으면 streamState 전체를 canonical로 원자적으로 교체하고,
  // canonical에는 fallback 메시지 하나만 존재한다(백엔드 test_ideation_structured_failure_
  // recovery.py::test_discussion_uses_safe_expert_judgment_instead_of_failing_session이
  // safe_fallback=true인 메시지 한 건만 저장됨을 이미 검증한다).
});

test("케이스 4: grounding retry — 이전 초안과 재시도 초안이 동시에 남지 않는다(최종 1건)", () => {
  let state = createEmptyStreamState();
  // 첫 발언은 정상적으로 스트리밍이 끝난다(구조는 유효했지만 근거 연결에 실패한 경우라
  // message_end까지는 정상 도착한다 — _ground_and_finalize_claims가 그 다음에 재시도한다).
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: "근거 연결 실패한 초안" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "M1" });
  assert.equal(state.messages.length, 1);

  // ideation_conv_nodes.py::_ground_and_finalize_claims가 재시도 llm_call 전에
  // discard_streamed_prompt(reason="grounding_retry")를 먼저 부른다.
  state = applyStreamEvent(state, { type: "message_reset", message_id: "M1", reason: "grounding_retry", will_retry: true });
  assert.equal(state.messages.length, 1); // 지워지지 않는다.
  assert.equal(state.messages[0].status, "reviewing");

  state = applyStreamEvent(state, { type: "message_start", message_id: "M2", speaker_id: "planning_expert", speaker_name: "기획 전문가", supersedes_message_id: "M1" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M2", delta: "재시도로 근거가 연결된 발언" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "M2" });

  // 이전 초안(M1)과 재시도 초안(M2)이 동시에 남지 않는다 — 정확히 한 건만 존재한다.
  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].message_id, "M2");
  assert.equal(state.messages[0].content, "재시도로 근거가 연결된 발언");
});

test("케이스 5: 서로 다른 위원의 연속 발언 — supersedes_message_id가 없으면 정상 발언을 재시도로 오인해 제거하지 않는다", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "P1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "P1", delta: "기획 위원의 정상 발언" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "P1" });

  // 개발 위원의 다음 발언은 재시도가 아니다 — supersedes_message_id가 없다. speaker_id가
  // 다르다는 이유만으로도, 같다는 이유만으로도 이전 메시지를 지우면 안 된다(요청: "단순히
  // speaker_id가 같다는 이유만으로 메시지를 교체하지 마세요").
  state = applyStreamEvent(state, { type: "message_start", message_id: "D1", speaker_id: "dev_expert", speaker_name: "개발 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "D1", delta: "개발 위원의 정상 발언" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "D1" });

  assert.equal(state.messages.length, 2); // 둘 다 남아있다 — 기획 위원 발언이 사라지지 않았다.
  assert.equal(state.messages[0].message_id, "P1");
  assert.equal(state.messages[0].content, "기획 위원의 정상 발언");
  assert.equal(state.messages[1].message_id, "D1");
  assert.equal(state.messages[1].content, "개발 위원의 정상 발언");
});

test("supersedes_message_id가 가리키는 메시지를 찾지 못하면(이미 처리됨 등) 안전하게 새 메시지로 추가한다", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M2", speaker_id: "planning_expert", speaker_name: "기획 전문가", supersedes_message_id: "M1-NOT-FOUND" });
  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].message_id, "M2");
});

test("applyStreamEvent: 서로 다른 message_id의 delta가 섞여도 각자 정확히 누적된다(두 페르소나 순차 스트리밍)", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "P1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "P1", delta: "기획 의견" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "P1" });
  state = applyStreamEvent(state, { type: "message_start", message_id: "D1", speaker_id: "dev_expert", speaker_name: "개발 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "D1", delta: "개발 의견" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "D1" });

  assert.equal(state.messages.length, 2);
  assert.equal(state.messages[0].speaker_id, "planning_expert");
  assert.equal(state.messages[0].content, "기획 의견");
  assert.equal(state.messages[1].speaker_id, "dev_expert");
  assert.equal(state.messages[1].content, "개발 의견");
});

test("applyStreamEvent: 'state'/'error' 이벤트는 리듀서가 건드리지 않는다(호출부가 직접 처리)", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  const before = state;
  state = applyStreamEvent(state, { type: "state", state: { phase: "finalized" } });
  assert.equal(state, before); // 참조까지 그대로 — 아무것도 안 바뀐다.
  state = applyStreamEvent(state, { type: "error", code: "llm_failure", message: "실패" });
  assert.equal(state, before);
});

// ---------------------------------------------------------------------------
// 실시간 타이핑 큐(advanceDisplay/isFullyDisplayed/pendingCharCount) 검증.
// IdeationConversationScreen.jsx의 rAF 루프가 매 프레임 advanceDisplay를 부르고,
// pendingFinalRef(최종 state 대기)가 있어도 isFullyDisplayed가 true가 될 때까지는
// canonical로 교체하지 않는다 — 그 핵심 로직을 React 없이 순수 함수 레벨에서 재현한다.
// ---------------------------------------------------------------------------

test("advanceDisplay: message_delta 하나만 와도(message_end 전) displayedContent가 즉시 채워지기 시작한다", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: "사용자가 선택한 후보는" });
  assert.equal(state.messages[0].displayedContent, ""); // 델타 직후에는 아직 화면에 안 그려졌다.
  assert.equal(isFullyDisplayed(state), false);

  state = advanceDisplay(state, 2); // rAF 한 프레임 흉내.
  assert.notEqual(state.messages[0].displayedContent, ""); // message_end 이전인데도 화면 텍스트가 이미 채워짐.
  assert.ok(state.messages[0].content.startsWith(state.messages[0].displayedContent));
});

test("isFullyDisplayed: 최종 state가 먼저 도착해도(=pendingFinalRef 존재) 타이핑이 안 끝났으면 false", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "dev_expert", speaker_name: "개발 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: "업무 자동화 AI 스킬" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "M1" });
  // 실제 컴포넌트에서는 이 시점에 서버 'state' 이벤트가 먼저 도착해 pendingFinalRef에
  // 담기지만, streamState 자체(=화면에 남아있는 임시 메시지)는 전혀 건드리지 않는다.
  assert.equal(isFullyDisplayed(state), false); // displayedContent가 아직 "" 이므로 false.

  // 타이핑 루프가 모든 글자를 다 드러낼 때까지 반복한다.
  let guard = 0;
  while (!isFullyDisplayed(state) && guard < 1000) {
    state = advanceDisplay(state, charsPerTickFor(pendingCharCount(state)));
    guard += 1;
  }
  assert.ok(guard > 1, "한 프레임 만에 전부 드러나면 타이핑 효과가 아니라 즉시 표시다");
  assert.equal(isFullyDisplayed(state), true); // 모든 글자가 출력된 뒤에만 true — 이 시점에만 canonical 교체가 허용된다.
  assert.equal(state.messages[0].displayedContent, state.messages[0].content);
});

test("advanceDisplay: 출력 큐가 길수록(pendingCharCount 큼) charsPerTickFor가 더 빨리 진행시킨다(지연 누적 방지)", () => {
  const short = "짧은 문장";
  const long = "매우 긴 문장이 여러 번 반복되어 출력 큐에 많이 쌓여 있는 상황을 흉내낸다".repeat(4);

  let shortState = createEmptyStreamState();
  shortState = applyStreamEvent(shortState, { type: "message_start", message_id: "S1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  shortState = applyStreamEvent(shortState, { type: "message_delta", message_id: "S1", delta: short });

  let longState = createEmptyStreamState();
  longState = applyStreamEvent(longState, { type: "message_start", message_id: "L1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  longState = applyStreamEvent(longState, { type: "message_delta", message_id: "L1", delta: long });

  const shortTicks = charsPerTickFor(pendingCharCount(shortState));
  const longTicks = charsPerTickFor(pendingCharCount(longState));
  assert.ok(longTicks > shortTicks, `밀린 글자가 많을수록(${long.length}) 더 빨리 진행해야 한다(짧은 문장 ${short.length}자 대비)`);
});

test("advanceDisplay: message_end 이후에도 남은 글자를 끝까지 드러낸다(중간에 끊기지 않음)", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "dev_expert", speaker_name: "개발 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: "한 줄\n다음 줄\n세 번째 줄" });
  state = applyStreamEvent(state, { type: "message_end", message_id: "M1" });
  assert.equal(state.messages[0].done, true);
  assert.equal(state.messages[0].displayedContent, ""); // done이어도 아직 안 그려짐.

  while (!isFullyDisplayed(state)) {
    state = advanceDisplay(state, 3);
  }
  // 줄바꿈이 슬라이싱 과정에서 깨지지 않고 그대로 유지된다.
  assert.equal(state.messages[0].displayedContent, "한 줄\n다음 줄\n세 번째 줄");
});

test("pendingCharCount: 완전히 따라잡은 메시지는 0을 기여한다", () => {
  let state = createEmptyStreamState();
  state = applyStreamEvent(state, { type: "message_start", message_id: "M1", speaker_id: "planning_expert", speaker_name: "기획 전문가" });
  state = applyStreamEvent(state, { type: "message_delta", message_id: "M1", delta: "짧은 문장" });
  assert.equal(pendingCharCount(state), "짧은 문장".length);
  state = advanceDisplay(state, 100); // 한 프레임에 충분히 커서 완전히 따라잡는다.
  assert.equal(pendingCharCount(state), 0);
});

test("dedupeMessagesById: 같은 message_id가 두 번 있으면 첫 번째만 남긴다(회귀 방지 — 중복 버그 진단)", () => {
  const messages = [
    { message_id: "MSG-1", content: "첫 번째" },
    { message_id: "MSG-2", content: "두 번째" },
    { message_id: "MSG-1", content: "첫 번째" }, // 완전히 같은 id — 중복
  ];
  const result = dedupeMessagesById(messages);
  assert.deepEqual(
    result.map((m) => m.message_id),
    ["MSG-1", "MSG-2"],
  );
});

test("dedupeMessagesById: 서로 다른 id의 동일한 content는 절대 지우지 않는다(content 비교 삭제 금지)", () => {
  const messages = [
    { message_id: "MSG-1", content: "좋은 의견입니다" },
    { message_id: "MSG-2", content: "좋은 의견입니다" }, // 다른 라운드의 같은 문장 — 지우면 안 됨.
  ];
  const result = dedupeMessagesById(messages);
  assert.equal(result.length, 2);
});

test("dedupeMessagesById: 빈/undefined 입력에도 안전하다", () => {
  assert.deepEqual(dedupeMessagesById(undefined), []);
  assert.deepEqual(dedupeMessagesById([]), []);
});

test("resolveRespondingToSpeakerId: 실제 message_id와 화자가 일치할 때만 대상을 반환한다", () => {
  const messages = [
    { message_id: "PLAN-1", speaker_id: "planning_expert" },
    {
      message_id: "DEV-1",
      speaker_id: "dev_expert",
      structured: {
        responding_to_message_id: "PLAN-1",
        responding_to_speaker_id: "planning_expert",
      },
    },
  ];
  assert.equal(resolveRespondingToSpeakerId(messages[1], messages), "planning_expert");
});

test("resolveRespondingToSpeakerId: 선언 화자 불일치·없는 메시지·자기 참조는 숨긴다", () => {
  const target = { message_id: "PLAN-1", speaker_id: "planning_expert" };
  const base = { message_id: "DEV-1", speaker_id: "dev_expert" };
  assert.equal(resolveRespondingToSpeakerId({ ...base, structured: {
    responding_to_message_id: "PLAN-1", responding_to_speaker_id: "dev_expert",
  } }, [target]), null);
  assert.equal(resolveRespondingToSpeakerId({ ...base, structured: {
    responding_to_message_id: "MISSING", responding_to_speaker_id: "planning_expert",
  } }, [target]), null);
  assert.equal(resolveRespondingToSpeakerId({ ...target, structured: {
    responding_to_message_id: "PLAN-1", responding_to_speaker_id: "planning_expert",
  } }, [target]), null);
});

console.log("\nAll ideationStreamReducer manual checks passed.");
