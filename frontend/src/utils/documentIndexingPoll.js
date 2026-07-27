import { getDocumentStatus } from '../api/documentApi'

// 가은/Claude(2026-07-19, INF-007 / 2026-07-23 요청: 문서별 색인 로딩바):
// document_status가 "indexing"인 동안 짧은 간격으로 GET .../status(DOC-004)를 폴링한다.
// DocumentUploadPage.jsx와 ReviewBoardPrototype.jsx에 거의 같은 함수가 복사돼 있던 걸
// 하나로 합쳤다 — updateDoc(각 화면의 문서 행 갱신 함수)만 인자로 받으면 어느 화면에서든
// 쓸 수 있다.
//
// 색인이 최종적으로 성공(indexed)하면 이미 계산해둔 본문 품질 평가(contentStatus/
// contentMeta)를 그대로 적용하고, indexed_empty/실패/타임아웃/변환실패는 그걸 우선해서
// 각각의 안내 메시지로 덮어쓴다 — "본문은 괜찮았는데 색인이 실패했다"와 "본문 자체가
// 부실하다"는 서로 다른 문제라 구분해서 보여준다.
export const DOCUMENT_STATUS_POLL_INTERVAL_MS = 2000
export const DOCUMENT_STATUS_POLL_MAX_ATTEMPTS = 90 // 2s * 90 = 3분

// 폴링 중(진짜 진행률을 모르는 구간)에도 막대가 멈춰 보이지 않도록, 매 tick마다 조금씩
// 밀어올린다 — 92%를 넘기지 않고 기다리다가, 완료되는 순간에만 100%로 뛴다.
const _PROGRESS_CREEP_START = 70
const _PROGRESS_CREEP_STEP = 2
const _PROGRESS_CREEP_CAP = 92

export function pollDocumentIndexing(pid, documentId, rowId, contentStatus, contentMeta, updateDoc) {
  let attempts = 0
  const timer = setInterval(async () => {
    attempts += 1
    try {
      const statusResult = await getDocumentStatus(pid, documentId)
      if (statusResult.status === 'indexing') {
        if (attempts >= DOCUMENT_STATUS_POLL_MAX_ATTEMPTS) {
          clearInterval(timer)
          updateDoc(rowId, {
            status: 'error',
            meta: '색인 상태 확인이 너무 오래 걸리고 있어요 — 새로고침해서 다시 확인해주세요.',
          })
          return
        }
        updateDoc(rowId, {
          progress: Math.min(_PROGRESS_CREEP_CAP, _PROGRESS_CREEP_START + attempts * _PROGRESS_CREEP_STEP),
        })
        return
      }
      clearInterval(timer)
      if (statusResult.status === 'indexing_failed') {
        updateDoc(rowId, {
          status: 'error',
          progress: 100,
          meta: statusResult.indexing_error?.message || '문서 색인 중 오류가 발생했습니다.',
        })
      } else if (statusResult.status === 'indexing_timeout') {
        updateDoc(rowId, {
          status: 'error',
          progress: 100,
          meta: '색인이 시간 내에 끝나지 않았습니다 — 다시 시도해주세요.',
        })
      } else if (statusResult.status === 'conversion_failed') {
        updateDoc(rowId, {
          status: 'error',
          progress: 100,
          meta: statusResult.conversion_metadata?.conversion_error || '문서를 변환하지 못했습니다.',
        })
      } else if (statusResult.status === 'indexed_empty') {
        updateDoc(rowId, {
          status: 'warning',
          progress: 100,
          meta: '문서에서 읽을 수 있는 텍스트를 찾지 못했어요 — 파일을 확인해주세요.',
        })
      } else {
        updateDoc(rowId, { status: contentStatus, progress: 100, meta: contentMeta })
      }
    } catch (err) {
      clearInterval(timer)
      updateDoc(rowId, { status: 'error', meta: err.message })
    }
  }, DOCUMENT_STATUS_POLL_INTERVAL_MS)
}
