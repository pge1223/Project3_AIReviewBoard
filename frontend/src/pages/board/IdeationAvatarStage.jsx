import { useEffect, useRef, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { getAvailableSpeakers, openMediaStreamSocket } from '../../api/mediaApi'
import { SPEAKER_META } from './ideationConversationHelpers'

// 재인/Claude(2026-07-23): 아이디어 회의(작성 전 모드)용 아바타 연동. WebSocket +
// MediaSource 스트리밍 자체는 frontend/src/components/meeting/CommitteeVideoStage.jsx
// (위원회 리뷰용, 검증 완료)의 핵심 메커니즘을 그대로 재사용한다 - 새로 발명한 부분이
// 아니다. 다른 점은 두 가지:
//   1. 위원 구성이 문서마다 바뀌는 리뷰 화면과 달리, 여기는 항상 진행자·기획·개발
//      3명 고정이라 "슬롯에 누구를 배정할지" 로직 자체가 필요 없다.
//   2. 리뷰 화면은 media_script(완성된 발언 배열)를 큐에 넣고 순서대로 재생하는
//      구조였지만, 여기는 대화가 실시간으로 이어지므로 "다음 화자를 언제, 누구를
//      부를지"를 이 컴포넌트가 아니라 부모(IdeationConversationScreen)가 결정한다.
//      이 컴포넌트는 그저 playQueue(이미 텍스트가 확정된 항목들)를 순서대로 스트리밍
//      재생하고, 각 항목이 실제로 화면에 나오기 시작할 때 onRevealed()를 불러서
//      부모가 그 항목의 채팅 텍스트를 공개하게 신호만 준다.
//
// 재인/Claude(2026-07-24, 요청: "텍스트는 미리 다 뽑아두고 코랩만 순서대로 처리"):
// 예전엔 이 컴포넌트가 "재생 끝나기 8초 전"에 onNeedNextSpeaker를 불러 부모에게 다음
// 텍스트를 요청했다(avatarPacingTimer.js). 이제 텍스트 생성은 부모가 아바타 타이밍과
// 무관하게 알아서 미리 다 해두므로, 이 컴포넌트는 더 이상 "다음 텍스트를 요청"하지
// 않는다 - playQueue에 이미 들어온 것만 순서대로 소비한다. 그래서 onNeedNextSpeaker/
// schedulePacingTimer 관련 코드를 전부 제거했다(avatarPacingTimer.js 파일도 삭제).
//
// 코랩 PERSONA_MAP 배정 확정(2026-07-23, 사용자 확인): 진행자=아바타C, 기획=아바타A,
// 개발=아바타B. 위원회 리뷰 화면(CommitteeVideoStage.jsx)에서 쓰던 것과 동일한
// persona_a/b/c 자산·TTS를 그대로 재사용한다.
const AVATAR_SLOTS = {
  ideation_facilitator: { colabSpeakerId: 'persona_c', idleVideo: '/mock-videos/persona_c/avata_c_rf.mp4' },
  planning_expert: { colabSpeakerId: 'business_strategy', idleVideo: '/mock-videos/persona_a/avata_rf.mp4' },
  dev_expert: { colabSpeakerId: 'technical_feasibility', idleVideo: '/mock-videos/persona_b/avata_b_rf.mp4' },
}

// 재인/Claude(2026-07-25, 요청: "제스처 랜덤으로 다양하게 보고 싶은데"): 위원회 리뷰
// 화면(CommitteeVideoStage.jsx)의 pickGestureIndex와 동일한 패턴을 그대로 가져왔다.
// 코랩 PERSONA_MAP의 세 화자(business_strategy/technical_feasibility/persona_c) 모두
// video_paths가 [rf(0번, 정지 자세), 제스처1(1번), 제스처2(2번)] 3개로 등록돼 있음을
// 확인했다(musetalk_setup_ideation_conv.ipynb). 매 발화마다 이 중 하나를 골라
// gesture_index로 실어 보내면 매번 같은 자세로만 말하지 않는다.
const GESTURE_INDICES = [0, 1, 2]

const MIME_CODEC = 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"'
const RESUME_THRESHOLD = 1.5
const PAUSE_THRESHOLD = 0.15
// 재인/Claude(2026-07-24, 요청: "전 사람 tail 시작하고 2초 후, 다음 발화가 준비됐을
// 때만 전환"): 이전 사람이 완전히 ended될 때까지 기다리지 않고, tail(idle로 넘어가는
// 구간) 시작 후 이 시간만 지나면 다음 사람으로 넘어갈 수 있다 - 단, 다음 영상이 실제로
// 재생 시작 가능한 만큼 버퍼링돼 있어야 한다(streamOneLine의 bufferedEnough 체크와
// 함께 걸림 - 준비 안 됐으면 이 시간이 지나도 계속 대기).
const NEXT_SPEAKER_GATE_DELAY_MS = 2000

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
// 제어할 때 await로 쓸 수 있게.
//
// 재인/Claude(2026-07-24, 요청: "텍스트 나오는 대로 바로 코랩에 큐 쌓기" ->
// "전 사람 tail 시작 2초 후, 다음 발화 준비됐을 때만 전환"): 코랩 요청(생성) 자체는
// 이 함수가 불리는 즉시 나간다 - "언제 요청을 보낼지"는 호출부(IdeationAvatarStage의
// maybeStartNextRef)가 텍스트가 큐에 들어오는 대로 바로 결정한다(더 이상 tail 시점을
// 기다리지 않음). 이 함수 자신은 오직 "화면 전환" 타이밍만 담당한다: mySeq/
// readyToPlaySeqRef로 위원회 리뷰 화면(CommitteeVideoStage.jsx)이 쓰던 "코랩의
// tts_end(speech_seconds)로 tail 시작 지점을 받고, 실제 재생 위치(video.currentTime)가
// 그 지점을 지났을 때만 믿는다"는 패턴을 그대로 가져와, tail 시작 후
// NEXT_SPEAKER_GATE_DELAY_MS(2초) 뒤에 다음 순번이 화면에 나올 수 있도록 게이트를
// 연다 - 이전 항목이 완전히 ended될 때까지 기다리지 않는다(idle로 넘어가는 tail 구간이
// 남아 있어도 다음 사람이 자연스럽게 이어 말함). 실제 전환은 이 게이트 + 다음 영상이
// 재생 가능할 만큼 버퍼링됐는지(bufferedEnough) 둘 다 만족해야 일어난다.
function streamOneLine({
  speakerId, text, videoRefs, setSpeaking, setStatus, gestureIndex,
  mySeq = 0, readyToPlaySeqRef = null, onRevealed, timelineStartAt = null, activeSpeakerIdsRef = null,
  onGenerationDone,
}) {
  return new Promise((resolve) => {
    // 재인/Claude(2026-07-25, 요청: "버퍼링 걸리는 구간을 제대로 잡게 모든 거에 로그"):
    // 이 항목 하나의 생애주기 전체(요청 시작 -> 청크 도착 -> 버퍼 상태 변화 -> 재생/정지 ->
    // 종료)를 timelineStartAt(부모가 마운트 시점에 한 번 잡아둔 공통 기준점) 기준
    // 경과시간(t=Xms)으로 찍는다 - 서로 다른 화자의 로그를 같은 시간축에서 직접 비교할
    // 수 있게 하기 위함이다.
    const t0 = timelineStartAt ?? performance.now()
    const elapsed = () => Math.round(performance.now() - t0)
    function log(label, extra) {
      console.log(`[avatar-debug][${speakerId}][t=${elapsed()}ms]`, label, {
        mySeq,
        activeSpeakers: activeSpeakerIdsRef ? [...activeSpeakerIdsRef.current] : undefined,
        ...extra,
      })
    }
    log('streamOneLine 시작', { textLength: text?.length, gestureIndex })

    const slot = AVATAR_SLOTS[speakerId]
    const video = videoRefs.current[speakerId]?.stream
    if (!slot || !video) {
      log('중단: slot 또는 video 엘리먼트 없음')
      resolve()
      return
    }

    let mediaSource = null
    let sourceBuffer = null
    let switched = false
    const appendQueue = []
    let appending = false
    let streamDone = false
    let stale = false
    let gateTimeoutId = null
    // 버퍼링(재생 중 정지) 구간을 정확히 잡기 위한 상태 - 아래 monitor tick 참고.
    let chunkCount = 0
    let totalBytes = 0
    let lastChunkAt = null
    let hasEverPlayed = false
    let stallStartedAt = null
    // 코랩의 tts_end(speech_seconds) - "영상의 몇 초 지점부터 tail인지". 신호 도착
    // 시점을 바로 믿지 않고(데이터가 항상 재생 위치보다 먼저 옴), monitor tick에서
    // video.currentTime이 이 값을 실제로 지났을 때만 tailStarted로 확정한다
    // (CommitteeVideoStage.jsx의 speechEndVideoTime/speechEndApplied와 동일한 이유).
    let speechEndVideoTime = null
    let tailStartedFired = false
    let revealedFired = false
    let generationDoneFired = false
    // 재인/Claude(2026-07-25, 실측: "겹치는 동안 청크 도착 간격이 2배로 늘어남 -> 재생
    // 중이던 쪽이 버퍼링"): "생성 완료"와 "재생 완료"는 다른 시점이다 - 코랩이 이 항목의
    // 프레임을 전부 만들어서 우리 쪽에 다 보낸 순간(done)이 곧 코랩 GPU가 비는 시점이고,
    // 그 뒤로는 이미 다 받아둔 데이터를 화면에 재생만 하는 것뿐이라 코랩 자원을 더 안 쓴다.
    // 그래서 다음 항목의 코랩 요청은 "재생이 끝날 때"가 아니라 "생성이 끝날 때"(done)
    // 바로 보내면 된다 - 그러면 코랩에서 두 항목의 GPU 작업이 절대 겹치지 않으면서도,
    // (생성 속도가 실시간 재생보다 빠른 한) 다음 항목이 화면에 필요해지기 훨씬 전부터
    // 미리 준비를 시작할 수 있다.
    function fireGenerationDoneOnce() {
      if (generationDoneFired) return
      generationDoneFired = true
      log('생성 완료 신호 전달 (코랩 자원 반납)')
      onGenerationDone?.()
    }
    function fireTailStartedOnce() {
      if (tailStartedFired) return
      tailStartedFired = true
      log('tail 시작 확정', { speechEndVideoTime, currentTime: video.currentTime })
      if (readyToPlaySeqRef) {
        gateTimeoutId = setTimeout(() => {
          // Math.max로 단조 증가만 허용 - 이 타이머가 늦게(예: finish()의 안전망 경로로
          // tailStarted가 뒤늦게 확정된 경우) 도는 사이 더 뒤 항목이 이미 게이트를
          // 열어놨을 수 있는데, 그걸 다시 되돌리면 안 된다.
          const before = readyToPlaySeqRef.current
          readyToPlaySeqRef.current = Math.max(readyToPlaySeqRef.current, mySeq + 1)
          log('전환 게이트 오픈 (tail+2초 경과)', { before, after: readyToPlaySeqRef.current })
        }, NEXT_SPEAKER_GATE_DELAY_MS)
      }
    }

    function pump() {
      if (stale || appending || appendQueue.length === 0 || !sourceBuffer || sourceBuffer.updating) return
      appending = true
      const chunk = appendQueue.shift()
      try {
        sourceBuffer.appendBuffer(chunk)
      } catch (e) {
        appending = false
        log('appendBuffer 실패', { error: String(e), queueLength: appendQueue.length })
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
      video.removeEventListener('ended', onEnded)
      setSpeaking(false)
      log('finish() 호출', {
        reason: video.error ? 'video-error' : (streamDone ? 'ended-or-done' : 'unknown'),
        chunkCount, totalBytes, currentTime: video.currentTime,
      })
      // tts_end가 끝내 안 온 경우(에러로 일찍 끝나는 등) 다음 항목 예열/전환 게이트가
      // 영원히 못 걸리면 안 되니 안전망으로 여기서도 한 번 확정한다.
      fireTailStartedOnce()
      // done이 끝내 안 온 경우(에러 등)도 마찬가지 - "생성 중" 게이트가 영원히 안 풀리면
      // 그 뒤로 아무도 코랩에 요청을 못 보내게 되므로 안전망으로 여기서도 확정한다.
      fireGenerationDoneOnce()
      // 완전히 끝났으니 다음 순번 게이트도 확실히 열어둔다(위 fireTailStartedOnce의
      // 2초 지연 타이머보다 먼저 여기 도달할 수도 있으므로 Math.max로 안전하게).
      if (readyToPlaySeqRef) {
        readyToPlaySeqRef.current = Math.max(readyToPlaySeqRef.current, mySeq + 1)
      }
      if (gateTimeoutId !== null) clearTimeout(gateTimeoutId)
      resolve()
    }

    function onEnded() {
      log('video "ended" 이벤트 발생 (자연 종료)')
      finish()
    }
    video.addEventListener('ended', onEnded)

    // 재인/Claude(2026-07-25, 요청: "말 끝나고 tail 영상으로 넘어갈 때마다 멈췄다가
    // 나오는 것 같다 - 그 부분에 집중적으로 로그"): 예전엔 200ms마다 무조건 틱 로그를
    // 찍었는데(동시성-버퍼링 문제 잡을 때는 필요했지만) 지금은 그 문제가 해결돼서 다시
    // 보면 너무 시끄럽다. 이제는 "본 발화 -> tail 전환 지점" 근처(speechEndVideoTime
    // 기준 앞뒤 몇 초)에서만 매 틱 상세 로그를 남기고, 그 구간 밖에서는 조용히 있는다 -
    // 그래야 콘솔에서 "말 끝나는 순간" 주변만 바로 찾아볼 수 있다. 버퍼링 시작/종료
    // 마커(⚠️/✅)는 구간과 무관하게 항상 남긴다 - 혹시 다른 지점에서 또 걸리면 놓치지
    // 않기 위해.
    const TAIL_WATCH_WINDOW_SEC = 3
    function nearTailBoundary() {
      if (speechEndVideoTime === null) return false
      return Math.abs(video.currentTime - speechEndVideoTime) <= TAIL_WATCH_WINDOW_SEC
    }
    const monitorId = setInterval(() => {
      if (stale) { clearInterval(monitorId); return }
      if (!switched) return
      const ahead = bufferedAhead()
      if (ahead === null) {
        log('SourceBuffer 상태 읽기 실패 - 스트림 중단')
        clearInterval(monitorId)
        setStatus('영상 스트림 오류 - 건너뜀')
        finish()
        return
      }
      // 진짜 재생 중(hasEverPlayed) 이었는데 지금 멈춰 있으면 버퍼링 정지 구간의 시작.
      if (hasEverPlayed && video.paused && stallStartedAt === null && !streamDone) {
        stallStartedAt = performance.now()
        log('⚠️ 버퍼링 시작 (재생 중 정지)', {
          ahead: ahead.toFixed(3), chunkCount, totalBytes,
          nearTailBoundary: nearTailBoundary(), speechEndVideoTime, currentTime: video.currentTime.toFixed(2),
          tailAlreadyStarted: tailStartedFired,
        })
      }
      if (nearTailBoundary() || stallStartedAt !== null) {
        log('tail 근접 tick', {
          ahead: ahead.toFixed(3), paused: video.paused, currentTime: video.currentTime.toFixed(2),
          speechEndVideoTime, streamDone, chunkCount, totalBytes, tailStartedFired,
        })
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
          if (stallStartedAt !== null) {
            log('✅ 버퍼링 종료 (재생 재개)', {
              stallDurationMs: Math.round(performance.now() - stallStartedAt),
              nearTailBoundary: nearTailBoundary(), tailStartedFired, currentTime: video.currentTime.toFixed(2),
            })
            stallStartedAt = null
          }
          video.play().then(() => {
            hasEverPlayed = true
            setSpeaking(true)
            // 재인/Claude(2026-07-24, 실측: "버퍼링으로 일시정지->재개되면 다음 사람
            // 텍스트가 조기 노출됨"): 재생 중 버퍼 부족으로 pause/resume이 반복되면
            // video.play()가 여러 번 불려서 이 .then()도 매번 다시 실행된다 - 첫
            // 재생 시작 때 딱 한 번만 onRevealed를 불러야, 같은 항목 때문에
            // avatarRevealedCount가 여러 번 올라가서 다음 사람 텍스트가 앞당겨
            // 공개되는 걸 막을 수 있다.
            if (!revealedFired) {
              revealedFired = true
              onRevealed?.()
            }
          }).catch(() => {})
        }
      } else if (ahead < PAUSE_THRESHOLD && !streamDone) {
        log('video.pause() 호출 (버퍼 부족)', { ahead: ahead.toFixed(3) })
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
      log('switchToStream (첫 바이너리 청크 도착, 대기 루프->스트림 전환)')
    }

    log('WebSocket 연결 시도')
    const ws = openMediaStreamSocket()
    ws.onopen = () => {
      if (stale) { ws.close(); return }
      log('WebSocket 연결됨 - 요청 전송', { colabSpeakerId: slot.colabSpeakerId, gestureIndex })
      ws.send(JSON.stringify({ speaker_id: slot.colabSpeakerId, text, gesture_index: gestureIndex }))
    }
    ws.onmessage = (event) => {
      if (stale) return
      if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data)
        if (msg.type === 'status' && msg.message === 'audio_ready') {
          log('코랩 status: audio_ready (TTS 완료, 영상 생성 시작)', { colabElapsed: msg.elapsed })
          setStatus('영상 생성 중...')
        } else if (msg.type === 'tts_end') {
          // 값만 저장한다 - 실제 확정은 monitor tick이 video.currentTime으로 이
          // 지점을 진짜 지날 때 한다(fireTailStartedOnce 참고, CommitteeVideoStage와
          // 동일한 이유 - 신호 도착 시점을 그대로 믿으면 아직 발화 중인데 다음 항목
          // 예열이 시작돼버린다).
          speechEndVideoTime = typeof msg.speech_seconds === 'number' ? msg.speech_seconds : 0
          log('코랩 status: tts_end (tail 시작 지점 수신)', { speechEndVideoTime })
        } else if (msg.type === 'done') {
          streamDone = true
          log('코랩 status: done (전체 프레임 전송 완료)', { colabElapsed: msg.elapsed, chunkCount, totalBytes })
          maybeEndStream()
          // 재생이 아직 안 끝났어도(화면엔 아직 재생 중), 코랩 쪽 이 항목 GPU 작업은
          // 여기서 이미 끝났다 - 다음 항목이 코랩을 바로 써도 겹칠 게 없다.
          fireGenerationDoneOnce()
        } else if (msg.type === 'error') {
          log('코랩 status: error', { message: msg.message })
          setStatus('에러: ' + msg.message)
          finish()
        }
      } else {
        chunkCount += 1
        const size = event.data.byteLength ?? event.data.size ?? 0
        totalBytes += size
        const now = performance.now()
        const gapMs = lastChunkAt !== null ? Math.round(now - lastChunkAt) : null
        lastChunkAt = now
        log('바이너리 청크 도착', {
          chunkCount, size, totalBytes, gapSinceLastChunkMs: gapMs,
          bufferedAheadNow: bufferedAhead()?.toFixed?.(3),
        })
        if (!switched) switchToStream()
        appendQueue.push(event.data)
        pump()
      }
    }
    ws.onerror = (e) => {
      log('WebSocket onerror', { error: String(e) })
      if (!stale) { setStatus('연결 에러'); finish() }
    }
    ws.onclose = (e) => {
      log('WebSocket onclose', { code: e.code, reason: e.reason, wasClean: e.wasClean })
    }

    // cleanup을 위해 Promise 바깥에서도 접근 가능하게 해야 하지만, 이 함수는
    // Promise만 반환하므로 취소는 상위(useEffect cleanup)에서 stale 처리 대신
    // ws.close()/video 정지로 별도 관리한다 - 다음 이터레이션에서 필요시 보강.
  })
}

// playQueue(이미 텍스트가 확정된 항목 배열)를 순서대로(호출부가 정한 순서 그대로)
// 스트리밍 재생한다. 부모가 "다음 화자 누구/무슨 텍스트"를 이미 다 정해서 넘겨주므로
// 이 컴포넌트는 순차 소비 + 화면 전환 타이밍만 담당한다.
export default function IdeationAvatarStage({ playQueue, onConsumed, onRevealed }) {
  const videoRefs = useRef({})
  // 재인/Claude(2026-07-25, 요청: "버퍼링 걸리는 구간을 제대로 잡게 모든 거에 로그"):
  // 컴포넌트가 처음 마운트된 시점을 공통 기준점으로 잡아서, 서로 다른 화자의
  // streamOneLine 로그가 전부 같은 시간축(t=Xms)으로 찍히게 한다 - 콘솔에서 시간순으로
  // 정렬해서 보면 "몇 명이 동시에 활성 상태였는지"와 "버퍼링 시작/종료 시점"을 바로
  // 대조해볼 수 있다.
  const timelineStartRef = useRef(performance.now())
  // 재인/Claude(2026-07-25, 요청: "진행자 tail 재생 중에 기획자가 이미 시작해도 둘 다
  // 화면에 보여야 함" - 실측: "말하던 애 립싱크 영상이 끊기고 TTS만 나옴"): 원래
  // speakingId 하나(문자열)로 "지금 말하는 사람 1명"만 표현했는데, 지금 설계는 전 사람이
  // tail 재생 중일 때 다음 사람이 이미 재생을 시작하는 겹침 구간을 의도적으로 허용한다
  // (readyToPlaySeqRef 게이트가 "완전히 ended"가 아니라 "tail 시작+2초"라서). 겹침
  // 구간엔 두 위원이 동시에 "말하는 중"이어야 하는데, 값이 하나뿐이면 나중 사람이
  // setSpeaking(true)를 부르는 순간 이전 사람 값이 덮어써져 opacity가 0으로 꺼져버렸다
  // (영상만 안 보이고, 그 <video> 엘리먼트 자체는 아무도 안 멈췄으니 오디오는 계속
  // 재생됨 - 그게 "영상은 끊기고 TTS만 나온다"의 정체였다). 위원별로 각자 재생 상태를
  // 따로 갖도록 맵으로 바꿨다 - statusMap과 같은 패턴.
  const [speakingMap, setSpeakingMap] = useState({})
  const [statusMap, setStatusMap] = useState({})
  const queueRef = useRef([])
  // playQueue는 부모가 계속 늘려서 넘겨주는(이미 재생한 항목도 안 지우는) 배열이다 -
  // 그래서 매번 queueRef를 통째로 덮어쓰면 이미 재생 끝난 항목까지 다시 큐에 들어가
  // 반복 재생되는 버그가 난다. consumedCountRef로 "지금까지 이 배열에서 몇 개를 이미
  // 내부 큐에 넣었는지"를 기억해두고, 그 뒤에 새로 늘어난 부분만 이어붙인다.
  const consumedCountRef = useRef(0)
  // 재인/Claude(2026-07-24~25, 요청: "텍스트 나오는 대로 바로 코랩에 큐 쌓기" -> 실측:
  // "동시에 2개 생성되면 청크 간격이 2배로 늘어나면서 재생 중이던 쪽이 버퍼링"): 처음엔
  // "텍스트만 준비되면 즉시 요청"이었는데, 실측해보니 코랩이 여러 요청을 진짜로 완전히
  // 독립적으로 처리하지 못하고 GPU/대역폭을 나눠 쓰면서 서로 느려졌다(청크 간격이
  // 정확히 2배로 뜀). 그래서 "생성 요청"과 "화면 재생"의 게이트를 이렇게 분리한다:
  //   - "생성 요청"은 generatingRef로 전역에서 딱 1개만 허용한다 - 코랩이 이 항목의
  //     프레임을 전부 보낸 순간(done 메시지, streamOneLine의 onGenerationDone)이 곧
  //     코랩 GPU가 비는 시점이므로, 그 즉시 다음 항목 생성을 시작한다. "재생"이 끝날
  //     때까지 기다릴 필요가 없다 - 이미 다 받아둔 데이터를 화면에 트는 건 코랩 자원을
  //     더 안 쓰기 때문이다. done은 보통 실시간 재생보다 빠르게 오므로(생성이 실시간보다
  //     빠름), 다음 항목이 화면에 필요해지는 시점보다 훨씬 일찍 미리 준비를 시작할 수
  //     있다.
  //   - 다만 같은 화자가 연속으로 말하는 경우엔 "생성"은 끝났어도 "재생"은 아직 안
  //     끝났을 수 있어서(그 화자가 아직 화면에서 말하는 중), activeSpeakerIdsRef(아래
  //     참고)로 그 경우만 별도로 막는다 - 같은 <video> 엘리먼트를 공유하기 때문.
  //   - "화면 재생"(video.play())은 nextSeqRef/readyToPlaySeqRef로 "내 차례"를 강제한다
  //     - "이전 항목 tail 시작 후 2초 경과"(streamOneLine의 fireTailStartedOnce/
  //     NEXT_SPEAKER_GATE_DELAY_MS 참고) + 이 항목 자신의 bufferedEnough(재생 가능할
  //     만큼 버퍼링됨) 둘 다 만족해야 실제로 넘어간다(streamOneLine의 myTurn 게이트).
  const nextSeqRef = useRef(0)
  const readyToPlaySeqRef = useRef(0)
  // 재인/Claude(2026-07-25): "지금 코랩에 생성 요청이 나가 있고 아직 done을 못 받은
  // 항목이 있는가"를 나타내는 전역(화자 무관) 게이트. true인 동안엔 다음 항목의 생성
  // 요청 자체를 미룬다 - 위 주석 참고.
  const generatingRef = useRef(false)
  // 재인/Claude(2026-07-24, 실측: "진행자가 두 번 연속 말하니 영상 스트림 오류 · 건너뜀"):
  // 위원마다 전용 <video> 타일이 있지만, 같은 화자(예: 진행자 질문 -> 나중에 진행자
  // 정리)가 연속으로 두 번 말하면 그 두 항목은 같은 <video> 엘리먼트를 공유한다. 앞
  // 항목이 아직 재생 중인데(아직 ended 안 됨) 뒤 항목을 미리 예열하면, 뒤 항목의
  // switchToStream()이 그 video.src/MediaSource를 통째로 새로 갈아끼워서 앞 항목의
  // SourceBuffer를 무효화시켜버린다(실측 확인: 로그에 media/stream WS가 겹쳐 열리고
  // 그 직후 진행자 타일에서 스트림 에러가 남). 그래서 "같은 speakerId가 지금 아직
  // 안 끝났으면" 그 화자의 다음 항목은 예열을 미룬다(다른 화자끼리는 그대로 미리 시작).
  const activeSpeakerIdsRef = useRef(new Set())
  // 재인/Claude(2026-07-25, 요청: "제스처 랜덤으로 다양하게"): speakerId -> 직전 발화에
  // 썼던 gesture_index. CommitteeVideoStage.jsx의 lastGestureIndexRef/pickGestureIndex와
  // 동일한 패턴 - 직전 값만 제외하고 나머지 중 랜덤으로 골라서 매번 같은 자세로만
  // 말하지 않게 한다.
  const lastGestureIndexRef = useRef({})
  function pickGestureIndex(speakerKey) {
    const last = lastGestureIndexRef.current[speakerKey]
    const options = GESTURE_INDICES.filter((i) => i !== last)
    const choice = options[Math.floor(Math.random() * options.length)]
    lastGestureIndexRef.current[speakerKey] = choice
    return choice
  }
  // 재인/Claude(2026-07-24): onRevealed도 매 렌더 뒤 최신 클로저로 다시 채워둔다(부모가
  // 리렌더될 때마다 새 함수를 내려줘도 이미 진행 중인 콜백이 옛날 버전만 계속 참조하는
  // 문제를 피하려고 - 예전 onNeedNextSpeakerRef와 같은 이유, 2026-07-23 실측 버그 참고
  // 기록은 git 히스토리에 남아있음).
  const onRevealedRef = useRef(onRevealed)
  useEffect(() => {
    onRevealedRef.current = onRevealed
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

  // 재인/Claude(2026-07-25, 요청: "생성 완료되면 바로 다음 걸 코랩에 보내기"): "지금 큐
  // 맨 앞 항목을 당장 생성 시작해도 되는가"를 반복 확인한다 - 여러 곳(effect, 생성완료
  // 콜백, 재생완료 콜백)에서 반복 호출해도 안전하도록 멱등적으로 짰다. 대기 조건 둘:
  //   1. generatingRef - 코랩에서 지금 뭔가 생성 중(done 안 옴)이면 전역으로 대기.
  //   2. activeSpeakerIdsRef - 큐 맨 앞 항목의 화자가 아직 화면에서 재생 중이면 대기
  //      (같은 <video> 엘리먼트 충돌 방지, 생성 자체는 끝났어도 재생이 안 끝났을 수 있음).
  // 한 번에 최대 1개만 새로 시작한다(생성은 전역으로 1개뿐이므로 while로 여러 개를
  // 연달아 시작할 이유가 없다 - 시작하자마자 generatingRef가 막아서 다음 반복은 어차피
  // 못 들어감).
  const maybeStartNextRef = useRef(() => {})
  useEffect(() => {
    maybeStartNextRef.current = () => {
      const tElapsed = () => Math.round(performance.now() - timelineStartRef.current)
      if (generatingRef.current) {
        console.log(`[avatar-debug][queue][t=${tElapsed()}ms]`, '스킵: 코랩이 이미 다른 항목 생성 중')
        return
      }
      if (queueRef.current.length === 0) return
      const nextSpeakerId = queueRef.current[0].speakerId
      if (activeSpeakerIdsRef.current.has(nextSpeakerId)) {
        console.log(`[avatar-debug][queue][t=${tElapsed()}ms]`, '스킵: 같은 화자가 아직 화면에서 재생 중', {
          nextSpeakerId, activeSpeakers: [...activeSpeakerIdsRef.current], queueLength: queueRef.current.length,
        })
        return
      }
      const item = queueRef.current.shift()
      const mySeq = nextSeqRef.current
      nextSeqRef.current += 1
      generatingRef.current = true
      activeSpeakerIdsRef.current.add(item.speakerId)
      setStatusMap((prev) => ({ ...prev, [item.speakerId]: '요청 중...' }))
      console.log(`[avatar-debug][queue][t=${tElapsed()}ms]`, '생성 시작', {
        speakerId: item.speakerId, mySeq,
        activeSpeakersNow: [...activeSpeakerIdsRef.current],
        remainingQueueLength: queueRef.current.length,
      })

      streamOneLine({
        speakerId: item.speakerId,
        text: item.text,
        gestureIndex: pickGestureIndex(item.speakerId),
        videoRefs,
        mySeq,
        readyToPlaySeqRef,
        timelineStartAt: timelineStartRef.current,
        activeSpeakerIdsRef,
        onGenerationDone: () => {
          generatingRef.current = false
          console.log(`[avatar-debug][queue][t=${tElapsed()}ms]`, '코랩 자원 반납 - 다음 항목 생성 시작 시도', { speakerId: item.speakerId, mySeq })
          maybeStartNextRef.current()
        },
        onRevealed: () => onRevealedRef.current?.(),
        setSpeaking: (v) => setSpeakingMap((prev) => ({ ...prev, [item.speakerId]: v })),
        setStatus: (s) => setStatusMap((prev) => ({ ...prev, [item.speakerId]: s })),
      }).then(() => {
        // 이 항목이 화면에서 완전히 끝났다 - 다음 순번이 이제 재생을 시작해도 되고,
        // 같은 화자의 다음 항목도 이제 그 <video> 엘리먼트를 안전하게 넘겨받을 수 있다.
        readyToPlaySeqRef.current = Math.max(readyToPlaySeqRef.current, mySeq + 1)
        activeSpeakerIdsRef.current.delete(item.speakerId)
        console.log(`[avatar-debug][queue][t=${tElapsed()}ms]`, '항목 재생 완료 - 화자 슬롯 반납', {
          speakerId: item.speakerId, mySeq, activeSpeakersNow: [...activeSpeakerIdsRef.current],
        })
        onConsumed?.(item)
        maybeStartNextRef.current()
      })
    }
  })

  useEffect(() => {
    const all = playQueue || []
    const newItems = all.slice(consumedCountRef.current)
    consumedCountRef.current = all.length
    if (newItems.length > 0) {
      queueRef.current.push(...newItems)
      console.log(`[avatar-debug][queue][t=${Math.round(performance.now() - timelineStartRef.current)}ms]`, 'playQueue에 새 항목 도착', {
        newItems: newItems.map((it) => ({ speakerId: it.speakerId, textLength: it.text?.length })),
        queueLengthNow: queueRef.current.length,
      })
    }
    maybeStartNextRef.current()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playQueue])

  // 에러 상태가 하나라도 있으면 카드 행 위에 작은 경고 배너 하나로 모아 보여준다(특정
  // 사진 위에 겹치지 않음). 위치/크기는 더 이상 누가 말하는지에 따라 바뀌지 않으므로
  // (위 ROLE_GRID_STYLE 참고) 여기서는 이 배너 표시 여부만 계산한다.
  const errorEntries = TILE_ORDER
    .map((id) => ({ id, statusText: statusMap[id] }))
    .filter((entry) => isErrorStatus(entry.statusText))

  // 재인/Claude(2026-07-26, dev #166 병합): 레이아웃(에러 배너 + 2열 그리드 +
  // ROLE_GRID_STYLE)은 용준님 버전을 그대로 쓴다. 내가 만들었던 468px 고정 폭
  // 레이아웃은 예전 IdeationConversationScreen의 gridTemplateColumns
  // ('1fr 80px 468px 80px 320px')에 맞춘 것이었는데, #166에서 그 그리드가
  // rb-ideation-side 카드 구조로 바뀌면서 기준 자체가 없어졌기 때문.
  // 단 speaking prop만 speakingMap으로 바꿨다 — 예전 speakingId(문자열 1개로
  // "지금 말하는 사람 1명"만 표현)는 화자 전환이 겹치는 구간에서 이전 화자 영상이
  // 사라지는 버그가 있어 화자별 boolean 맵으로 교체했고(위 541줄 참고),
  // speakingId는 이제 정의 자체가 없어서 그대로 두면 ReferenceError가 난다.
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
            speaking={!!speakingMap[speakerId]}
            statusText={statusMap[speakerId]}
            style={ROLE_GRID_STYLE[speakerId]}
          />
        ))}
      </div>
    </div>
  )
}
