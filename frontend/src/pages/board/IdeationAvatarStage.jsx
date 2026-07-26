import { useEffect, useRef, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
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

// 상태 문구가 에러 계열인지 판별 — 요청: "사진 위에 에러 메시지를 겹쳐서 띄우는 방식은
// 제거하고, 에러가 있을 경우 카드 밖 상단의 작은 경고 배너로 표시". 정상 진행 상태
// (요청 중.../영상 생성 중... 등)는 그대로 카드 안에 작게 남기고, 에러만 밖으로 뺀다.
function isErrorStatus(statusText) {
  return /에러|오류|실패/.test(statusText || '')
}

// 용준/Claude(2026-07-26, 요청: "세 카드 모두 동일한 크기·비율의 피라미드 배치, 진행자만
// 커지지 않게") — 이전엔 "지금 말하는 위원"을 다른 두 명보다 큰 카드(다른 padding/
// border-radius/높이/폰트 크기, gridColumn:'1/-1'로 2칸 전체 폭)로 보여줬는데, 그래서
// 진행자가 우연히 기본 발언자였을 때 항상 카드 자체가 더 크고 이미지 비율도 달라 보였다.
// 이제 세 카드는 이 컴포넌트 하나로 완전히 동일한 크기·padding·border-radius·이미지
// 비율을 쓴다 — "누가 지금 말하는지"는 style prop(부모가 넘기는 border/그림자)로만
// 표현하고, 크기는 절대 건드리지 않는다. 위치(진행자=상단 중앙, 기획/개발=하단 좌우)도
// 더 이상 "누가 말하는지"에 따라 동적으로 바뀌지 않고 역할별로 고정된다(아래
// ROLE_GRID_STYLE 참고) — 스트리밍 재생 대상(videoRefs 키)은 항상 speakerId로 고정이라
// 이 변경과 무관하게 안전하다.
function AvatarTileFrame({ speakerId, videoRefs, speaking, statusText, style }) {
  const [thumbnailFailed, setThumbnailFailed] = useState(false)
  const meta = SPEAKER_META[speakerId]
  if (!videoRefs.current[speakerId]) videoRefs.current[speakerId] = {}
  const hasError = isErrorStatus(statusText)

  return (
    <div
      style={{
        background: 'var(--bg-1)',
        // 요청: "허용 - border-color 변경, box-shadow 추가"만으로 발언 중 강조 표현.
        // 카드 크기(padding/border-radius)는 speaking 여부와 무관하게 항상 동일하다.
        border: speaking ? '2px solid var(--purple)' : '1px solid var(--glass-border)',
        borderRadius: 14,
        padding: 10,
        boxShadow: speaking ? '0 4px 16px rgba(124,92,234,0.18)' : '0 1px 3px rgba(28,26,46,0.04)',
        transition: 'border-color .2s ease, box-shadow .2s ease',
        ...style,
      }}
    >
      <div
        style={{
          position: 'relative',
          width: '100%',
          // 용준/Claude(2026-07-26, 요청: "실제 원본 비율 확인 후 그 비율을 그대로 적용") —
          // 아바타 원본 mp4(frontend/public/mock-videos/persona_{a,b,c})를 MP4 박스
          // 구조(moov>trak>mdia>minf>stbl>stsd>avc1)까지 직접 파싱해 실측한 해상도가
          // 세 파일 모두 720x1280(9:16)으로 동일했다 — 임의로 3:4나 4:3을 강제하지 않고
          // 이 실측값을 그대로 쓴다. 세 역할 모두 항상 같은 비율이라 카드 크기도
          // 자연히 항상 동일해진다.
          aspectRatio: '9 / 16',
          background: '#e4f1fb',
          borderRadius: 10,
          overflow: 'hidden',
        }}
      >
        {thumbnailFailed ? (
          // 요청: 이미지(영상) 로드 실패 시에도 사진 자리 위에 문구를 겹치지 않고, 그
          // 자리를 대체하는 대체 UI로 보여준다.
          <div
            style={{
              position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', gap: 6, padding: 6,
              color: 'var(--text-2)', textAlign: 'center',
            }}
          >
            <AlertTriangle size={18} />
            <span style={{ fontSize: 12, lineHeight: 1.4 }}>영상을 불러오지 못했어요</span>
          </div>
        ) : (
          <>
            {/* muted는 JSX 속성으로 고정하지 않는다 - CommitteeVideoStage와 같은 이유로,
                switchToStream()이 명령형으로 video.muted를 바꾸는데 React가 리렌더 때마다
                되돌려버리면 오디오 초기화 도중 음소거가 깜빡인다. 이미지·영상 모두 같은
                박스(위 aspectRatio 9/16)를 쓰고 object-fit:contain으로 잘리지 않게
                맞춘다 - 요청 7번("이미지와 영상에 같은 비율 적용")과 동일한 이유. */}
            <video
              ref={(el) => { videoRefs.current[speakerId].idle = el }}
              playsInline
              loop
              onError={() => setThumbnailFailed(true)}
              style={{
                position: 'absolute', inset: 0, width: '100%', height: '100%',
                objectFit: 'contain', objectPosition: 'center bottom',
              }}
            />
            {/* 재인/Claude(2026-07-23): CommitteeVideoStage.jsx의 videoStreamVisible과 같은
                이유 — 평소엔 투명해서 안 보이고, 실제로 재생이 걸린(speaking===true) 순간에만
                드러난다. 이 토글을 빠뜨리면 스트림 video는 계속 재생되니 오디오는 들리는데
                (재생 자체는 되고 있으므로) 화면은 대기 루프만 계속 보이는 상태가 된다. */}
            <video
              ref={(el) => { videoRefs.current[speakerId].stream = el }}
              playsInline
              style={{
                position: 'absolute', inset: 0, width: '100%', height: '100%',
                objectFit: 'contain', objectPosition: 'center bottom',
                opacity: speaking ? 1 : 0, pointerEvents: 'none',
              }}
            />
          </>
        )}
      </div>

      <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span
          className={`badge ${meta.badgeClass} mono`}
          style={{ fontSize: 13, fontWeight: 600, padding: '4px 9px' }}
        >
          {meta.label}
        </span>
        {/* 요청: "'발언 중' 배지"로 강조하는 방법 중 하나 — speaking일 때만 role 배지
            옆에 작은 pill을 하나 더 붙인다(카드/이미지 크기에는 영향 없음). */}
        {speaking && !hasError && (
          <span
            style={{
              fontSize: 13, fontWeight: 600, color: 'var(--purple)',
              background: 'var(--purple-dim)', borderRadius: 999, padding: '2px 7px',
            }}
          >
            발언 중
          </span>
        )}
      </div>
      <div style={{ marginTop: 6, minHeight: 18 }}>
        {statusText && !hasError ? (
          <span style={{ fontSize: 14, fontWeight: 500, color: '#625d72' }}>{statusText}</span>
        ) : (
          <span style={{ fontSize: 14, fontWeight: 500, color: '#625d72' }}>
            {speaking ? '실시간 발언 중' : '대기 중'}
          </span>
        )}
      </div>
    </div>
  )
}

// 용준/Claude(2026-07-26, 요청: "진행자는 위쪽 중앙, 기획 위원은 아래쪽 왼쪽, 개발
// 위원은 아래쪽 오른쪽 — 누가 말하는지와 무관하게 항상 이 배치") — 예전엔 지금 말하는
// 사람이 큰 카드(gridColumn:'1/-1', 다른 크기)로 옮겨 다녔지만, 이제 위치는 역할별로
// 고정이고 크기도 세 카드가 항상 같다. 진행자만 grid-column을 2칸 다 차지하게 두되
// width는 한 칸만큼만(요청 예시의 calc((100% - gap)/2))으로 제한하고 justifySelf로
// 가운데 정렬해, "두 칸을 차지하되 실제 카드 크기는 한 칸과 동일"을 만족시킨다.
const AVATAR_GRID_GAP = 8
const ROLE_GRID_STYLE = {
  ideation_facilitator: {
    gridColumn: '1 / -1',
    gridRow: 1,
    justifySelf: 'center',
    width: `calc((100% - ${AVATAR_GRID_GAP}px) / 2)`,
  },
  planning_expert: { gridColumn: 1, gridRow: 2 },
  dev_expert: { gridColumn: 2, gridRow: 2 },
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

  // 에러 상태가 하나라도 있으면 카드 행 위에 작은 경고 배너 하나로 모아 보여준다(특정
  // 사진 위에 겹치지 않음). 위치/크기는 더 이상 누가 말하는지에 따라 바뀌지 않으므로
  // (위 ROLE_GRID_STYLE 참고) 여기서는 이 배너 표시 여부만 계산한다.
  const errorEntries = TILE_ORDER
    .map((id) => ({ id, statusText: statusMap[id] }))
    .filter((entry) => isErrorStatus(entry.statusText))

  return (
    <div>
      {errorEntries.length > 0 && (
        <div
          role="alert"
          style={{
            display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10,
            padding: '6px 10px', borderRadius: 8, fontSize: 13.5, lineHeight: 1.4,
            background: 'var(--coral-dim)', color: 'var(--coral)',
          }}
        >
          <AlertTriangle size={12} style={{ flexShrink: 0 }} />
          <span>
            {errorEntries
              .map((entry) => `${SPEAKER_META[entry.id]?.label || entry.id} 영상 연결 오류`)
              .join(' · ')}
          </span>
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: AVATAR_GRID_GAP }}>
        {TILE_ORDER.map((speakerId) => (
          <AvatarTileFrame
            key={speakerId}
            speakerId={speakerId}
            videoRefs={videoRefs}
            speaking={speakingId === speakerId}
            statusText={statusMap[speakerId]}
            style={ROLE_GRID_STYLE[speakerId]}
          />
        ))}
      </div>
    </div>
  )
}
