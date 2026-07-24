import { useEffect, useRef, useState } from 'react'
import { getAvailableSpeakers, openMediaStreamSocket } from '../../api/mediaApi'
import { SPEAKER_META } from './ideationConversationHelpers'
import { schedulePacingTimer } from './avatarPacingTimer'

// 재인/Claude(2026-07-23): 아이디어 회의(작성 전 모드)용 아바타 연동. WebSocket +
// MediaSource 스트리밍 자체는 frontend/src/components/meeting/CommitteeVideoStage.jsx
// (위원회 리뷰용, 검증 완료)의 핵심 메커니즘을 그대로 재사용한다 - 새로 발명한 부분이
// 아니다. 다른 점은 두 가지:
//   1. 위원 구성이 문서마다 바뀌는 리뷰 화면과 달리, 여기는 항상 진행자·기획·개발
//      3명 고정이라 "슬롯에 누구를 배정할지" 로직 자체가 필요 없다.
//   2. 리뷰 화면은 media_script(완성된 발언 배열)를 큐에 넣고 순서대로 재생하는
//      구조였지만, 여기는 대화가 실시간으로 이어지므로 "다음 화자를 언제, 누구를
//      부를지"를 이 컴포넌트가 아니라 부모(IdeationConversationScreen)가 결정한다.
//      이 컴포넌트는 그저 (speakerId, text)가 오면 스트리밍하고, 재생이 실제로
//      시작된 뒤 "duration_ms - 3초" 시점에 onNeedNextSpeaker(현재 speakerId)를
//      불러서 부모가 다음 화자를 부를 신호만 준다 - 누가 다음인지·무슨 API를
//      부를지는 전혀 모른다(용준님 그래프 로직과 분리 유지).
//
// 코랩 PERSONA_MAP 배정 확정(2026-07-23, 사용자 확인): 진행자=아바타C, 기획=아바타A,
// 개발=아바타B. 위원회 리뷰 화면(CommitteeVideoStage.jsx)에서 쓰던 것과 동일한
// persona_a/b/c 자산·TTS를 그대로 재사용한다.
const AVATAR_SLOTS = {
  ideation_facilitator: { colabSpeakerId: 'persona_c', idleVideo: '/mock-videos/persona_c/avata_c_rf.mp4' },
  planning_expert: { colabSpeakerId: 'business_strategy', idleVideo: '/mock-videos/persona_a/avata_rf.mp4' },
  dev_expert: { colabSpeakerId: 'technical_feasibility', idleVideo: '/mock-videos/persona_b/avata_b_rf.mp4' },
}

const MIME_CODEC = 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"'
const RESUME_THRESHOLD = 1.5
const PAUSE_THRESHOLD = 0.15

const TILE_ORDER = ['ideation_facilitator', 'planning_expert', 'dev_expert']
const AVATAR_TILE_WIDTH = 230

function AvatarTileFrame({ speakerId, videoRefs, speaking, statusText }) {
  const meta = SPEAKER_META[speakerId]
  if (!videoRefs.current[speakerId]) videoRefs.current[speakerId] = {}
  return (
    <div
      style={{
        position: 'relative',
        width: AVATAR_TILE_WIDTH,
        aspectRatio: '9 / 16',
        background: 'var(--bg-1)',
        border: '1px solid var(--glass-border)',
        borderRadius: 12,
        overflow: 'hidden',
      }}
    >
      {/* muted는 JSX 속성으로 고정하지 않는다 - CommitteeVideoStage와 같은 이유로,
          switchToStream()이 명령형으로 video.muted를 바꾸는데 React가 리렌더 때마다
          되돌려버리면 오디오 초기화 도중 음소거가 깜빡인다. */}
      <video
        ref={(el) => { videoRefs.current[speakerId].idle = el }}
        playsInline
        loop
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
      />
      {/* 재인/Claude(2026-07-23): CommitteeVideoStage.jsx의 videoStreamVisible과 같은 이유 —
          평소엔 투명해서 안 보이고, 실제로 재생이 걸린(speaking===true) 순간에만 드러난다.
          이 토글을 빠뜨리면 스트림 video는 계속 재생되니 오디오는 들리는데(재생 자체는
          되고 있으므로) 화면은 대기 루프만 계속 보이는 상태가 된다 — 처음 이 파일을 쓸 때
          이 부분을 빠뜨려서 실제로 그 증상이 재현됐었다. */}
      <video
        ref={(el) => { videoRefs.current[speakerId].stream = el }}
        playsInline
        style={{
          position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover',
          opacity: speaking ? 1 : 0, pointerEvents: 'none',
        }}
      />
      <span
        className={`badge ${meta.badgeClass} mono`}
        style={{
          position: 'absolute', left: 6, bottom: 6, fontSize: 10,
          boxShadow: speaking ? '0 0 0 2px var(--purple, #7c5cff)' : 'none',
        }}
      >
        {meta.label}
      </span>
      {statusText && (
        <span
          style={{
            position: 'absolute', right: 6, top: 6, fontSize: 9.5, padding: '3px 7px',
            borderRadius: 999, background: 'rgba(0,0,0,0.55)', color: '#ffd479',
          }}
        >
          {statusText}
        </span>
      )}
    </div>
  )
}

// speakerId를 하나 지정해서 그 화자의 말풍선(text)을 실제로 스트리밍 재생한다.
// 반환값(Promise)은 재생이 완전히 끝났을 때(또는 에러) resolve된다 - 부모가 순서를
// 제어할 때 await로 쓸 수 있게. onNeedNextSpeaker는 duration_ms 기반 타이머가 울리는
// 즉시(아직 재생 중일 때) 호출된다 - "재생이 끝나야" 호출되는 게 아니라는 점이 중요.
//
// 재인/Claude(2026-07-24, 요청: "tail 시작 + 텍스트 준비되면 바로 코랩 요청 보내기"):
// mySeq/readyToPlaySeqRef/onTailStarted 세 개가 추가됐다 - 위원회 리뷰 화면
// (CommitteeVideoStage.jsx)이 이미 쓰던 "코랩의 tts_end(speech_seconds)로 tail 시작
// 지점을 받고, 실제 재생 위치(video.currentTime)가 그 지점을 지났을 때만 믿는다"는
// 패턴을 그대로 가져왔다. 다른 점: 저기는 tail 시작 시점에 다음 위원을 화면에도 바로
// 공개하지만(겹침 허용), 여기는 "요청/생성만" tail 시작 시점에 미리 보내고 실제
// video.play()(화면 전환)는 여전히 이전 항목이 완전히 끝난(ended) 뒤로 미룬다(readyToPlaySeqRef
// 게이트) - 데모 하루 전이라 화면에 동시에 두 위원이 겹쳐 보이는 위험까지 새로 만들고
// 싶지 않아서, "생성을 미리 시작"하는 것만 우선 해결한다.
function streamOneLine({
  speakerId, text, videoRefs, onNeedNextSpeaker, setSpeaking, setStatus,
  mySeq = 0, readyToPlaySeqRef = null, onTailStarted,
}) {
  return new Promise((resolve) => {
    const slot = AVATAR_SLOTS[speakerId]
    const video = videoRefs.current[speakerId]?.stream
    if (!slot || !video) {
      resolve()
      return
    }

    let mediaSource = null
    let sourceBuffer = null
    let switched = false
    let cancelPacingTimer = null
    const appendQueue = []
    let appending = false
    let streamDone = false
    let stale = false
    // 코랩의 audio_ready(status) 메시지로 채워진다 - switchToStream()이 이 값을 읽어
    // 타이머를 건다. const가 아니라 객체(ref처럼 mutable)인 이유는 값이 도착하는
    // 시점(ws.onmessage)이 switchToStream 호출 시점보다 항상 먼저이긴 하지만, 이
    // 순서를 코드 구조에 의존하지 않고 명시적으로 표현하기 위함이다.
    const durationMsRef = { current: 0 }
    // 코랩의 tts_end(speech_seconds) - "영상의 몇 초 지점부터 tail인지". 신호 도착
    // 시점을 바로 믿지 않고(데이터가 항상 재생 위치보다 먼저 옴), monitor tick에서
    // video.currentTime이 이 값을 실제로 지났을 때만 tailStarted로 확정한다
    // (CommitteeVideoStage.jsx의 speechEndVideoTime/speechEndApplied와 동일한 이유).
    let speechEndVideoTime = null
    let tailStartedFired = false
    function fireTailStartedOnce() {
      if (tailStartedFired) return
      tailStartedFired = true
      console.log('[avatar-debug] tail started', { speakerId })
      onTailStarted?.()
    }

    function pump() {
      if (stale || appending || appendQueue.length === 0 || !sourceBuffer || sourceBuffer.updating) return
      appending = true
      const chunk = appendQueue.shift()
      try {
        sourceBuffer.appendBuffer(chunk)
      } catch (e) {
        appending = false
        console.error('[IdeationAvatarStage] appendBuffer 실패', e)
      }
    }

    function maybeEndStream() {
      if (!stale && streamDone && appendQueue.length === 0 && sourceBuffer && !sourceBuffer.updating
          && mediaSource && mediaSource.readyState === 'open') {
        try { mediaSource.endOfStream() } catch (e) { /* 재생엔 영향 없음 */ }
      }
    }

    function bufferedAhead() {
      try {
        if (!sourceBuffer || sourceBuffer.buffered.length === 0) return 0
        const idx = sourceBuffer.buffered.length - 1
        return sourceBuffer.buffered.end(idx) - video.currentTime
      } catch (e) {
        return null
      }
    }

    function finish() {
      if (stale) return
      stale = true
      cancelPacingTimer?.()
      video.removeEventListener('ended', onEnded)
      setSpeaking(false)
      // tts_end가 끝내 안 온 경우(에러로 일찍 끝나는 등) 다음 항목 예열이 영원히
      // 못 걸리면 안 되니 안전망으로 여기서도 한 번 확정한다.
      fireTailStartedOnce()
      resolve()
    }

    function onEnded() {
      finish()
    }
    video.addEventListener('ended', onEnded)

    const monitorId = setInterval(() => {
      if (stale) { clearInterval(monitorId); return }
      if (!switched) return
      const ahead = bufferedAhead()
      if (ahead === null) {
        clearInterval(monitorId)
        setStatus('영상 스트림 오류 - 건너뜀')
        finish()
        return
      }
      // 실제로 재생 중일 때(currentTime이 진짜로 흐르고 있을 때)만 tail 시작 지점을
      // 지났는지 확인한다 - 아직 내 차례가 아니라 paused 상태면 currentTime이 안
      // 움직이므로 이 체크도 자연히 대기한다.
      if (!video.paused && speechEndVideoTime !== null && video.currentTime >= speechEndVideoTime) {
        fireTailStartedOnce()
      }
      if (video.paused) {
        const bufferedEnough = ahead >= RESUME_THRESHOLD || (streamDone && ahead > 0.05)
        const myTurn = readyToPlaySeqRef ? readyToPlaySeqRef.current === mySeq : true
        if (bufferedEnough && myTurn) {
          video.play().then(() => setSpeaking(true)).catch(() => {})
        }
      } else if (ahead < PAUSE_THRESHOLD && !streamDone) {
        video.pause()
      }
    }, 200)

    function switchToStream() {
      switched = true
      video.muted = false
      setStatus('')
      mediaSource = new MediaSource()
      video.src = URL.createObjectURL(mediaSource)
      mediaSource.addEventListener('sourceopen', () => {
        if (stale) return
        sourceBuffer = mediaSource.addSourceBuffer(MIME_CODEC)
        sourceBuffer.mode = 'sequence'
        sourceBuffer.addEventListener('updateend', () => { appending = false; pump(); maybeEndStream() })
        pump()
      })
      // 재인/Claude(2026-07-23): 재생이 실제로 시작되는 순간(play 이벤트)부터
      // "duration_ms - 3초" 뒤에 다음 화자를 미리 호출한다 - avatarPacingTimer.js
      // 참고(signal 도착 시점이 아니라 반드시 실제 play 이벤트 기준인 이유도 거기 적혀있음).
      // durationMsRef는 아래 ws.onmessage의 audio_ready에서 채워진다.
      console.log('[avatar-debug] switchToStream', { speakerId, durationMs: durationMsRef.current })
      cancelPacingTimer = schedulePacingTimer(video, durationMsRef.current, () => {
        console.log('[avatar-debug] pacing timer FIRED -> onNeedNextSpeaker', { speakerId })
        onNeedNextSpeaker?.(speakerId)
      })
    }

    const ws = openMediaStreamSocket()
    ws.onopen = () => {
      if (stale) { ws.close(); return }
      ws.send(JSON.stringify({ speaker_id: slot.colabSpeakerId, text }))
    }
    ws.onmessage = (event) => {
      if (stale) return
      if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data)
        if (msg.type === 'status' && msg.message === 'audio_ready') {
          // 코랩이 TTS 완료 직후(영상 생성 시작 전) 보내주는 오디오 길이 - 아직
          // switchToStream()이 안 불렸을 수도 있으므로 ref에 먼저 저장해둔다.
          durationMsRef.current = msg.duration_ms || 0
          setStatus('영상 생성 중...')
        } else if (msg.type === 'tts_end') {
          // 값만 저장한다 - 실제 확정은 monitor tick이 video.currentTime으로 이
          // 지점을 진짜 지날 때 한다(fireTailStartedOnce 참고, CommitteeVideoStage와
          // 동일한 이유 - 신호 도착 시점을 그대로 믿으면 아직 발화 중인데 다음 항목
          // 예열이 시작돼버린다).
          speechEndVideoTime = typeof msg.speech_seconds === 'number' ? msg.speech_seconds : 0
        } else if (msg.type === 'done') {
          streamDone = true
          maybeEndStream()
        } else if (msg.type === 'error') {
          setStatus('에러: ' + msg.message)
          finish()
        }
      } else {
        if (!switched) switchToStream()
        appendQueue.push(event.data)
        pump()
      }
    }
    ws.onerror = () => {
      if (!stale) { setStatus('연결 에러'); finish() }
    }

    // cleanup을 위해 Promise 바깥에서도 접근 가능하게 해야 하지만, 이 함수는
    // Promise만 반환하므로 취소는 상위(useEffect cleanup)에서 stale 처리 대신
    // ws.close()/video 정지로 별도 관리한다 - 다음 이터레이션에서 필요시 보강.
  })
}

// speakerId -> text 딕셔너리를 받아 순서대로(TILE_ORDER 기준 아님, 호출부가 정한
// 순서 그대로) 스트리밍한다. 부모가 "다음 화자 누구"를 결정하므로 이 컴포넌트는
// playQueue(배열)를 prop으로 받아 순차 소비만 한다.
export default function IdeationAvatarStage({ playQueue, onConsumed, onNeedNextSpeaker }) {
  const videoRefs = useRef({})
  const [speakingId, setSpeakingId] = useState(null)
  const [statusMap, setStatusMap] = useState({})
  const queueRef = useRef([])
  // playQueue는 부모가 계속 늘려서 넘겨주는(이미 재생한 항목도 안 지우는) 배열이다 -
  // 그래서 매번 queueRef를 통째로 덮어쓰면 이미 재생 끝난 항목까지 다시 큐에 들어가
  // 반복 재생되는 버그가 난다. consumedCountRef로 "지금까지 이 배열에서 몇 개를 이미
  // 내부 큐에 넣었는지"를 기억해두고, 그 뒤에 새로 늘어난 부분만 이어붙인다.
  const consumedCountRef = useRef(0)
  // 재인/Claude(2026-07-24, 요청: "tail 시작 + 텍스트 준비되면 바로 코랩 요청 보내기"):
  // 예전엔 while+await로 완전히 순차 처리했다(이전 항목 영상이 100% ended돼야 다음
  // 항목의 코랩 요청 자체가 나갔다) - 그래서 다음 텍스트가 미리 와 있어도 그걸 코랩에
  // 던지는 시점은 전혀 안 당겨졌다. 이제 "생성 요청"과 "화면 재생"의 게이트를 분리한다:
  //   - pendingRef: 지금 미리 생성 중인(아직 자기 tail도 안 시작한) 항목 하나. 이게
  //     있고 아직 tail을 안 시작했으면 그 다음 항목은 아직 예열을 시작하지 않는다
  //     (한 번에 최대 "재생 중 1개 + 예열 중 1개"까지만 허용 - 파이프라인을 깊게 만들
  //     필요는 없고, 바로 다음 것만 미리 당겨오면 됨).
  //   - nextSeqRef/readyToPlaySeqRef: 화면 전환(video.play()) 자체는 여전히 "내 차례"
  //     (=이전 항목이 진짜로 ended됨)가 와야만 허용한다 - 위원이 화면에 동시에 두 명
  //     겹쳐 보이는 위험까지 새로 만들고 싶지 않아서, 화면 전환 순서는 그대로 두고
  //     생성 요청만 미리 보낸다(streamOneLine의 myTurn 게이트 참고).
  const pendingRef = useRef(null)
  const nextSeqRef = useRef(0)
  const readyToPlaySeqRef = useRef(0)
  // 재인/Claude(2026-07-24, 실측: "진행자가 두 번 연속 말하니 영상 스트림 오류 · 건너뜀"):
  // 위원마다 전용 <video> 타일이 있지만, 같은 화자(예: 진행자 질문 -> 나중에 진행자
  // 정리)가 연속으로 두 번 말하면 그 두 항목은 같은 <video> 엘리먼트를 공유한다. 앞
  // 항목이 아직 재생 중인데(아직 ended 안 됨) 뒤 항목을 미리 예열하면, 뒤 항목의
  // switchToStream()이 그 video.src/MediaSource를 통째로 새로 갈아끼워서 앞 항목의
  // SourceBuffer를 무효화시켜버린다(실측 확인: 로그에 media/stream WS가 겹쳐 열리고
  // 그 직후 진행자 타일에서 스트림 에러가 남). 그래서 "같은 speakerId가 지금 아직
  // 안 끝났으면" 그 화자의 다음 항목은 예열을 미룬다(다른 화자끼리는 그대로 미리 시작).
  const activeSpeakerIdsRef = useRef(new Set())
  // 재인/Claude(2026-07-23, 실측: "phase가 계속 옛날 값(awaiting_candidate_selection)으로
  // 찍혀서 절대 못 넘어감" — 콘솔 로그로 확정): 큐 진행 콜백(onTailStarted/.then 등)이
  // 클로저로 onNeedNextSpeaker를 직접 캡처하면, 부모(IdeationConversationScreen)가
  // 리렌더될 때마다 최신 ideationConv를 담은 새 handleAvatarNeedNextSpeaker를 내려줘도
  // 이미 진행 중인 콜백은 자기가 만들어질 때 캡처한 옛날 버전만 계속 참조한다 - 그래서
  // 몇 분이 지나도 phase가 라운드 시작 전 값(awaiting_candidate_selection)으로 영원히
  // 고정돼 매번 막혔다. ref에 항상 최신 콜백을 담아두고, maybeStartNextRef 안에서는
  // 그 ref를 통해서만 호출한다(아래 maybeStartNextRef도 같은 이유로 매 렌더 뒤 최신
  // 클로저로 다시 채워둔다).
  const onNeedNextSpeakerRef = useRef(onNeedNextSpeaker)
  useEffect(() => {
    onNeedNextSpeakerRef.current = onNeedNextSpeaker
  })

  useEffect(() => {
    TILE_ORDER.forEach((id) => {
      const video = videoRefs.current[id]?.idle
      if (!video) return
      video.muted = true
      video.src = AVATAR_SLOTS[id].idleVideo
      video.play().catch(() => {})
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 재인/Claude(2026-07-24): "다음 항목을 지금 예열해도 되는가"의 유일한 판단 지점.
  // 여러 곳(effect, tail 시작 콜백, 항목 완료 콜백)에서 반복 호출해도 안전하도록
  // 멱등적으로 짰다 - 예열할 게 없거나(큐 빔) 이미 예열 중인 게 아직 tail도 안
  // 시작했으면 그냥 조용히 리턴한다. onNeedNextSpeakerRef와 같은 이유로 매 렌더 뒤
  // effect에서 최신 클로저를 다시 담아둔다(렌더 도중 ref.current를 직접 쓰지 않는다).
  const maybeStartNextRef = useRef(() => {})
  useEffect(() => {
    maybeStartNextRef.current = () => {
      if (pendingRef.current && !pendingRef.current.tailStarted) return
      if (queueRef.current.length === 0) return
      const nextSpeakerId = queueRef.current[0].speakerId
      if (activeSpeakerIdsRef.current.has(nextSpeakerId)) {
        console.log('[avatar-debug] skip prefetch: 같은 화자가 아직 재생 중', { nextSpeakerId })
        return
      }
      const item = queueRef.current.shift()
      const mySeq = nextSeqRef.current
      nextSeqRef.current += 1
      const entry = { speakerId: item.speakerId, tailStarted: false }
      pendingRef.current = entry
      activeSpeakerIdsRef.current.add(item.speakerId)
      setStatusMap((prev) => ({ ...prev, [item.speakerId]: '요청 중...' }))
      console.log('[avatar-debug] prefetch start', { speakerId: item.speakerId, mySeq })

      streamOneLine({
        speakerId: item.speakerId,
        text: item.text,
        videoRefs,
        mySeq,
        readyToPlaySeqRef,
        onTailStarted: () => {
          if (entry.tailStarted) return
          entry.tailStarted = true
          if (pendingRef.current === entry) pendingRef.current = null
          maybeStartNextRef.current()
        },
        // 재인/Claude(2026-07-23, 실측: "이 말 끝나기 3초 전에, 다음 기획자 말은 끝나지도
        // 않았는데 개발자 말이 먼저 튀어나옴"): 큐에 이미 재생 대기 중인 다음 대사가
        // 있거나(예: 고정 문구 여러 개가 한 응답에 몰려 들어온 경우) 이미 예열 중인 항목이
        // 있으면 요청을 또 보내지 않는다 — "다음 사람 준비시켜줘"는 정말로 준비된 게
        // 하나도 없을 때만 의미가 있다.
        // 재인/Claude(2026-07-24, 실측: "오늘은... 다음이 영원히 안 나옴" - 콘솔 로그로
        // 확정): pacing timer는 leadMs(8초) 때문에 거의 항상 "내 tail이 시작되기 전"에
        // 미리 울린다 - 그 시점엔 pendingRef.current가 다름 아닌 나 자신(entry)이다(아직
        // tailStarted 전이라 안 지워졌을 뿐). "pendingRef.current가 있으면 스킵"이라고만
        // 쓰면 매번 자기 자신을 "이미 준비된 다음 항목"으로 오판해서 영원히 요청을 안
        // 보낸다 - pendingRef가 나 자신이 아니라 "다른" 항목을 가리킬 때만 진짜로 스킵해야
        // 한다.
        onNeedNextSpeaker: (id) => {
          if (queueRef.current.length > 0 || (pendingRef.current && pendingRef.current !== entry)) {
            console.log('[avatar-debug] skip onNeedNextSpeaker: 이미 대기/예열 중', { id })
            return
          }
          console.log('[avatar-debug] nothing queued/pending, proceeding onNeedNextSpeaker', { id })
          onNeedNextSpeakerRef.current?.(id)
        },
        setSpeaking: (v) => setSpeakingId(v ? item.speakerId : null),
        setStatus: (s) => setStatusMap((prev) => ({ ...prev, [item.speakerId]: s })),
      }).then(() => {
        // 이 항목이 화면에서 완전히 끝났다 - 다음 순번(mySeq+1)이 이제 재생을 시작해도 되고,
        // 같은 화자의 다음 항목도 이제 그 <video> 엘리먼트를 안전하게 넘겨받을 수 있다.
        readyToPlaySeqRef.current = mySeq + 1
        activeSpeakerIdsRef.current.delete(item.speakerId)
        onConsumed?.(item)
        if (pendingRef.current === entry) pendingRef.current = null
        maybeStartNextRef.current()
      })
    }
  })

  useEffect(() => {
    const all = playQueue || []
    const newItems = all.slice(consumedCountRef.current)
    consumedCountRef.current = all.length
    if (newItems.length > 0) queueRef.current.push(...newItems)
    maybeStartNextRef.current()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playQueue])

  // 재인/Claude(2026-07-24, 요청: "아바타 C(진행자)가 위야" - 스크린샷에 표시된 배치):
  // 기존엔 세로로 3개를 쭉 쌓았는데, 진행자는 위에 가운데로, 기획·개발 위원은 그 아래
  // 나란히 두 칸으로 바꿨다.
  // 재인/Claude(2026-07-24, 실측: "캔버스와의 간격/정렬이 이상함"): 컨테이너 너비를 콘텐츠에
  // 맡겨두면(기획+개발 두 칸=230*2+gap) 이 자리에 맞춰둔 그리드 트랙(468px)보다 살짝
  // 넓어져서 오른쪽 캔버스 위치가 같이 밀렸다. 아래 gap을 8로 줄여 두 칸 폭을 정확히
  // 468(=230*2+8)로 맞추고, 바깥 컨테이너도 그 폭으로 고정해서 진행자가 그 안에서
  // 정확히 가운데 오도록 했다(IdeationConversationScreen.jsx의 gridTemplateColumns
  // '1fr 468px 160px 320px'와 맞물림).
  return (
    <div style={{ width: 468, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <AvatarTileFrame
        speakerId="ideation_facilitator"
        videoRefs={videoRefs}
        speaking={speakingId === 'ideation_facilitator'}
        statusText={statusMap.ideation_facilitator}
      />
      {/* 재인/Claude(2026-07-24, 요청: "A/B 사이 간격 균일하게, 그 위에 진행자 딱 가운데"):
          픽셀 단위로 하나씩 밀던 marginLeft/absolute 실험은 다 걷어내고, 원래대로
          단순한 flex 두 칸 + 진행자 중앙 정렬로 되돌렸다 - justifyContent: 'center'로
          두 칸 자체를 468px 칼럼 안에서 가운데 두면 A/B 사이 간격(gap)도 균일하고
          진행자도 그 칼럼 중앙(alignItems: 'center')에 자연스럽게 맞는다. */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 8 }}>
        <AvatarTileFrame
          speakerId="planning_expert"
          videoRefs={videoRefs}
          speaking={speakingId === 'planning_expert'}
          statusText={statusMap.planning_expert}
        />
        <AvatarTileFrame
          speakerId="dev_expert"
          videoRefs={videoRefs}
          speaking={speakingId === 'dev_expert'}
          statusText={statusMap.dev_expert}
        />
      </div>
    </div>
  )
}
