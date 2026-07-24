// 재인/Claude(2026-07-23): 아이디어 회의 아바타 연동 — "다음 화자를 언제 부를지"
// 타이밍만 담당하는 순수 유틸. 대화 로직(누가 다음 화자인지, 무슨 말을 하는지)은
// 전혀 모른다 - 그냥 "이 시간 지나면 콜백 호출해줘"만 안다.
//
// 왜 signal 도착 시점이 아니라 video의 실제 play 이벤트부터 재는가: 코랩이 보내는
// duration_ms(오디오 길이)는 항상 재생 시작보다 먼저 도착한다(버퍼링 때문에 데이터가
// 재생 위치보다 앞서 온다는 게 committee_video_streaming_architecture.md의 tts_end
// 버그에서 이미 실측 확인된 사실). "도착 즉시" 타이머를 걸면 실제 재생과 어긋난다 -
// 그래서 반드시 video 엘리먼트의 진짜 play 이벤트를 기준으로 삼는다.
//
// 왜 "끝나기 N초 전"에 부르는가: 다음 화자의 텍스트 생성 + TTS + 립싱크 영상 생성이
// 그 자체로 시간이 걸리므로(미리 만들어두지 않으면 발화 사이에 정적이 생김), 지금
// 화자가 아직 말하는 도중에 미리 트리거해서 그 지연을 숨긴다.
// 재인/Claude(2026-07-24, 실측: "기획위원 말하고 개발위원 텍스트 나오는 타이밍이 조금
// 늦은 거 같아"): 원래 3000이었는데, 다음 턴 파이프라인(LLM 호출+RAG 검증 -> TTS ->
// MuseTalk 영상 생성)을 전부 합치면 3초보다 오래 걸리는 경우가 많아 보여서 8000으로
// 늘림 — 실측하면서 더 조정 가능(짧으면 여전히 끊기고, 너무 길면 재생 시작하자마자
// 다음 사람이 요청돼서 부자연스러움).
export const NEXT_SPEAKER_LEAD_MS = 8000

/**
 * videoEl이 재생을 시작(play 이벤트)하는 순간부터 카운트해서,
 * (durationMs - leadMs) 시점에 onFire를 정확히 한 번 호출한다.
 *
 * durationMs가 leadMs보다 짧으면(아주 짧은 발화) 음수 지연이 나오는데, 그 경우
 * setTimeout(fn, 0)으로 즉시 호출한다 - 음수를 그대로 넘기면 브라우저가 0으로
 * 취급하긴 하지만, 의도를 코드에 명시적으로 남기기 위해 Math.max(0, ...)로
 * 직접 클램프한다.
 *
 * 반환값: 타이머를 취소하는 함수. "잠시만" 눌러서 회의가 중단되면 반드시 이 함수를
 * 호출해서 예약된 다음 화자 호출이 새어나가지 않게 해야 한다(handleInterject 쪽에서
 * 훅으로 연결 예정).
 */
export function schedulePacingTimer(videoEl, durationMs, onFire, leadMs = NEXT_SPEAKER_LEAD_MS) {
  if (!videoEl) return () => {}

  let timeoutId = null
  let cancelled = false

  function handlePlay() {
    if (cancelled) return
    const delay = Math.max(0, durationMs - leadMs)
    console.log('[avatar-debug] pacing timer: play event fired, scheduling', { durationMs, leadMs, delay })
    timeoutId = setTimeout(() => {
      console.log('[avatar-debug] pacing timer: setTimeout elapsed, calling onFire', { delay })
      if (!cancelled) onFire()
    }, delay)
  }

  // { once: true } - 이 영상이 재생되는 동안 딱 한 번만(첫 play) 반응한다. 버퍼링으로
  // 일시정지->재개가 반복돼도(pause 후 다시 play) 타이머를 중복으로 걸지 않는다 -
  // 이미 한 번 걸린 타이머는 실제 시계 기준으로 흐르므로 재생이 잠깐 멈췄다 다시
  // 시작해도 그대로 유효하다(버퍼링 몇백ms 정도는 3초 마진 안에서 흡수됨).
  videoEl.addEventListener('play', handlePlay, { once: true })

  return function cancel() {
    cancelled = true
    videoEl.removeEventListener('play', handlePlay)
    if (timeoutId !== null) {
      clearTimeout(timeoutId)
      timeoutId = null
    }
  }
}
