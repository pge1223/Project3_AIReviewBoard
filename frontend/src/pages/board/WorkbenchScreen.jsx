import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, AlertTriangle, ArrowRight, Lightbulb, MessageSquare, PenLine, Sparkles } from 'lucide-react'
import { getDocuments } from '../../api/documentApi'
import { getQuoteMatches, getContextCheck, getTypoCheck, getFormatCheck } from '../../api/workbenchApi'
import { getProjectReport } from '../../api/projectApi'
import { API_BASE_URL } from '../../api/client'
import PdfDocumentView from './PdfDocumentView'

// 재인/Claude(2026-07-21): docs/REVIEW_BOARD_서비스_방향성_정리_20260720.md의
// "5. 핵심 UI: 시각적 인터랙티브 워크벤치" 구현.
//
// 데이터 출처:
// - 기획서 원문: /documents/{project_id}/{document_id}/preview-pdf(backend/app/api/routes/
//   documents.py, LibreOffice로 원본을 PDF로 변환)를 pdf.js(PdfDocumentView.jsx)로 그린다.
//   HTML 재구성 방식을 먼저 시도했으나, docx 원본엔 "몇 페이지에서 어떻게 줄바꿈되는지"
//   정보가 없어(렌더러가 그릴 때 계산하는 값) 워드/한글 원본과 완전히 같은 페이지 모습을
//   못 만들었다 - 그래서 PDF 변환 + pdf.js로 바꿨다.
// - 위원 피드백(issues/suggestions): UploadAndAnalyzeScreen이 분석 끝나고
//   sessionStorage에 저장해둔 review_output(JSON)을 그대로 읽는다.
//
// 하이라이트 위치: 위원이 인용한 evidence.chunk_id로 벡터DB에서 청크 원문을 ID
// 직접 조회한다(backend/app/api/routes/workbench.py, AI 재호출 없이 항상 원문 그대로).
// 실제 하이라이트 그리기는 PdfDocumentView.jsx가 pdf.js의 textLayer 위에서 한다.
//
// 겹치는 하이라이트: 같은 청크를 가리키는 피드백이 여러 개면(evidence_ids가 issue/
// suggestion 문장 하나하나가 아니라 criterion 단위라 자주 그렇다) 전부 한 곳에 묶어서
// 보여준다 - "먼저 온 것만 남기고 버림"으로 나머지가 화면에서 사라지는 문제를 고쳤다.
//
// 아바타 연동은 아직 1차 버전이 아님 - 오른쪽엔 자리만 잡아두고, 실제 스트리밍은
// ai/media/musetalk/committee_video_streaming_architecture.md의 하위 배관을 재사용해서
// 다음 단계에서 붙인다.

function authHeader() {
  const token = localStorage.getItem('auth_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

// review_output.evidence는 evidence_id -> {chunk_id, source_type, ...} 배열이다.
// rubric_score.evidence_ids(criterion 단위)에서 "submission"(기획서 본문, 공고문/criteria
// 아님) 소스의 첫 evidence를 골라 그 chunk_id를 이 criterion의 모든 issue/suggestion에 붙인다.
function buildEvidenceChunkMap(reviewOutput) {
  const byId = new Map()
  for (const ev of reviewOutput?.evidence || []) {
    byId.set(ev.evidence_id, ev)
  }
  return byId
}

function firstSubmissionChunkId(evidenceIds, evidenceById) {
  for (const evId of evidenceIds || []) {
    const ev = evidenceById.get(evId)
    if (ev && (ev.source_type === 'submission' || !ev.source_type) && ev.chunk_id) {
      return ev.chunk_id
    }
  }
  return null
}

function extractFeedbackItems(reviewOutput) {
  if (!reviewOutput) return []
  const evidenceById = buildEvidenceChunkMap(reviewOutput)
  const items = []
  for (const reviewer of reviewOutput.reviewer_results || []) {
    for (const rs of reviewer.rubric_scores || []) {
      const chunkId = firstSubmissionChunkId(rs.evidence_ids, evidenceById)
      const kinds = [
        ['issue', rs.issues || []],
        ['suggestion', rs.suggestions || []],
      ]
      for (const [kind, texts] of kinds) {
        for (const text of texts) {
          items.push({
            id: `${reviewer.persona_id}-${rs.criterion_id}-${kind}-${items.length}`,
            kind,
            text,
            criterionName: rs.criterion_name,
            reviewerName: reviewer.persona_name || reviewer.role,
            personaId: reviewer.persona_id,
            judgment: rs.judgment,
            chunkId,
          })
        }
      }
    }
  }
  return items
}

// review_output.schema.json v2.3.0의 judgment(4종, rubricScore/unscoredCriterion.attempted_judgment
// 공용) -> 화면에 보여줄 한글 표기.
const JUDGMENT_LABEL = {
  strong: '우수',
  acceptable: '적당함',
  needs_improvement: '보완 필요',
  critical_risk: '심각한 위험',
}

// 재인/Claude(2026-07-21, ai/meeting/graph/transform.py에 unscored_criteria 추가 — 사용자
// 확인 하에 진행, 경이님 확인 필요): 위원이 근거 부족으로 채점하지 못한 항목을 화면에
// 보여준다. 지금까지는 이 항목들이 rubric_scores에서 그냥 사라져서, "왜 이 항목 피드백이
// 없지?"를 사용자가 알 방법이 없었다(실측 확인).
//
// reason별로 사용자한테 해줄 조언이 다르다:
// - insufficient_evidence/not_applicable: 위원이 원문 전체를 보고 직접 "이 관점에서 다룰
//   내용이 없다"고 판단한 것 - 문서에 관련 내용을 추가해야 한다는 신호.
// - evidence_gate_blocked: 위원은 실제 판단(attempted_judgment)까지 냈는데, 그걸 뒷받침할
//   근거를 저희 RAG 시스템이 명확히 못 찾아 연결해서 점수에 반영을 안 한 것 - 사용자
//   문서가 부족해서가 아니라 저희 검색 시스템 한계일 수 있어서, "당신 탓 아니다"를
//   명확히 하고 위원이 실제로 낸 판단을 그대로 보여준다.
//
// GPT가 review_items에 그 criterion_id 자체를 아예 안 넣은 경우(위원 응답 누락)는
// unscored_criteria에도 안 남으므로, 여기엔 나타나지 않는다 - 그건 사용자한테 보여줄
// 문제가 아니라 저희 쪽에서 고쳐야 할 시스템 이슈라 의도적으로 화면에서 제외한다.
function extractUnscoredItems(reviewOutput) {
  const items = []
  for (const reviewer of reviewOutput?.reviewer_results || []) {
    for (const u of reviewer.unscored_criteria || []) {
      const reviewerName = reviewer.persona_name || reviewer.role
      if (u.reason === 'evidence_gate_blocked') {
        const judgmentLabel = JUDGMENT_LABEL[u.attempted_judgment] || u.attempted_judgment
        items.push({
          id: `unscored-${reviewer.persona_id}-${u.criterion_id}`,
          criterionName: u.criterion_name,
          reviewerName,
          notUserFault: true,
          message: `${reviewerName}의 판단은 "${judgmentLabel}"이었지만, 이를 뒷받침할 근거를 ` +
            `저희 시스템이 문서에서 명확히 찾아 연결하지 못해 최종 점수에는 반영되지 않았습니다. ` +
            `문서 내용이 부족해서가 아니라 저희 검색 시스템의 한계일 수 있어요.`,
          strengths: u.strengths || [],
          weaknesses: u.weaknesses || [],
        })
      } else {
        items.push({
          id: `unscored-${reviewer.persona_id}-${u.criterion_id}`,
          criterionName: u.criterion_name,
          reviewerName,
          notUserFault: false,
          message: `${reviewerName}이(가) 문서 전체를 검토했지만, 이 항목을 평가할 근거를 ` +
            `찾지 못했습니다. 관련 내용을 문서에 보강해보세요.`,
        })
      }
    }
  }
  return items
}

// 재인/Claude(2026-07-21): "맥락 이상 감지"(backend/app/api/routes/workbench.py의
// /context-check) 결과를 위원 피드백과 똑같은 feedbackItems 모양으로 바꾼다 - 그래야
// 아래 하이라이트/우측 상세 패널 로직을 위원 피드백과 그대로 같이 쓸 수 있다.
function extractContextFeedbackItems(contextFindings) {
  return (contextFindings || []).map((f) => ({
    id: f.id,
    kind: 'context',
    text: f.message,
    criterionName: '맥락 이상',
    reviewerName: 'AI 검토',
  }))
}

// 재인/Claude(2026-07-22): "오탈자 검사" 결과도 맥락 이상 감지와 마찬가지로 PDF
// 하이라이트(클릭하면 아래 공용 상세 패널에 표시)까지는 같은 방식을 쓴다 - 대신
// 사용자 요청으로 화면 오른쪽엔 맥락 이상 감지와 섞이지 않는 별도의 "오탈자 의심"
// 요약 카드(평가되지 않은 항목 카드와 같은 패턴)를 하나 더 둔다.
function extractTypoFeedbackItems(typoFindings) {
  return (typoFindings || []).map((f) => ({
    id: f.id,
    kind: 'typo',
    text: `'${f.quote}' → '${f.corrected}'${f.message ? ` (${f.message})` : ''}`,
    criterionName: '오탈자',
    reviewerName: 'AI 검토',
  }))
}

const KIND_BADGE = {
  issue: { label: '지적사항', bg: 'var(--coral-dim)', fg: 'var(--coral)', icon: <AlertCircle size={12} /> },
  suggestion: { label: '제안', bg: 'var(--green-dim)', fg: 'var(--green)', icon: <Lightbulb size={12} /> },
  context: { label: '맥락 이상', bg: 'var(--amber-dim)', fg: 'var(--amber)', icon: <AlertTriangle size={12} /> },
  typo: { label: '오탈자', bg: 'var(--rose-dim)', fg: 'var(--rose)', icon: <PenLine size={12} /> },
}

export default function WorkbenchScreen({ projectId, onNext }) {
  const [pdfUrl, setPdfUrl] = useState(null)
  const [docError, setDocError] = useState('')
  const [reviewOutput, setReviewOutput] = useState(null)
  const [selectedFeedbackIds, setSelectedFeedbackIds] = useState([])

  const [quoteMatches, setQuoteMatches] = useState(null)
  const [matchingLoading, setMatchingLoading] = useState(false)
  const [matchingError, setMatchingError] = useState('')

  // 맥락 이상 감지는 위원 회의 결과와 무관하게 독립적으로 돌아간다(문서만 있으면 됨) -
  // 그래서 reviewOutput을 기다리지 않고 프로젝트가 정해지는 즉시 따로 불러온다. 실패해도
  // 위원 피드백 화면 자체는 정상 동작해야 하므로 조용히 콘솔에만 남긴다(치명적이지 않음).
  const [contextFindings, setContextFindings] = useState(null)
  // 오탈자 검사도 같은 이유로(문서만 있으면 됨, 위원 회의와 무관) 독립적으로 불러온다.
  const [typoFindings, setTypoFindings] = useState(null)
  // 분량·밀도 체크도 문서만 있으면 되므로 독립적으로 불러온다.
  const [formatCheck, setFormatCheck] = useState(null)

  useEffect(() => {
    if (!projectId) return
    let cancelled = false
    ;(async () => {
      try {
        const docs = await getDocuments(projectId)
        const target = docs.find((d) => (d.document_role || 'target') === 'target')
        if (!target) {
          if (!cancelled) setDocError('업로드된 기획서를 찾지 못했습니다.')
          return
        }
        if (!cancelled) setPdfUrl(`${API_BASE_URL}/documents/${projectId}/${target.id}/preview-pdf`)
      } catch (err) {
        if (!cancelled) setDocError(err.message)
      }
    })()
    return () => { cancelled = true }
  }, [projectId])

  // 가은/Claude(2026-07-21): 실측 제보 — "내 프로젝트"에서 분석이 이미 끝난 프로젝트로
  // 이어서 들어오면(sessionStorage가 이번 세션엔 없는 새 탭/재접속) 워크벤치가 빈 화면으로
  // 보였다. sessionStorage 캐시가 없으면 저장된 회의 결과(RPT-001 /report)를 대신 불러온다.
  useEffect(() => {
    if (!projectId) return
    const cached = sessionStorage.getItem(`analysis:${projectId}`)
    if (cached) {
      try {
        setReviewOutput(JSON.parse(cached))
        return
      } catch (e) {
        console.error('[WorkbenchScreen] 분석 결과 파싱 실패', e)
      }
    }
    let cancelled = false
    getProjectReport(projectId)
      .then((report) => { if (!cancelled) setReviewOutput(report) })
      .catch((err) => { if (!cancelled) console.error('[WorkbenchScreen] 결과 조회 실패', err) })
    return () => { cancelled = true }
  }, [projectId])

  useEffect(() => {
    if (!projectId) return
    let cancelled = false
    getContextCheck(projectId)
      .then((findings) => { if (!cancelled) setContextFindings(findings) })
      .catch((err) => { console.error('[WorkbenchScreen] 맥락 이상 감지 실패', err) })
    return () => { cancelled = true }
  }, [projectId])

  useEffect(() => {
    if (!projectId) return
    let cancelled = false
    getTypoCheck(projectId)
      .then((findings) => { if (!cancelled) setTypoFindings(findings) })
      .catch((err) => { console.error('[WorkbenchScreen] 오탈자 검사 실패', err) })
    return () => { cancelled = true }
  }, [projectId])

  useEffect(() => {
    if (!projectId) return
    let cancelled = false
    getFormatCheck(projectId)
      .then((data) => { if (!cancelled) setFormatCheck(data) })
      .catch((err) => { console.error('[WorkbenchScreen] 분량·밀도 검사 실패', err) })
    return () => { cancelled = true }
  }, [projectId])

  const feedbackItems = useMemo(() => extractFeedbackItems(reviewOutput), [reviewOutput])
  const contextFeedbackItems = useMemo(() => extractContextFeedbackItems(contextFindings), [contextFindings])
  const typoFeedbackItems = useMemo(() => extractTypoFeedbackItems(typoFindings), [typoFindings])
  const allFeedbackItems = useMemo(
    () => [...feedbackItems, ...contextFeedbackItems, ...typoFeedbackItems],
    [feedbackItems, contextFeedbackItems, typoFeedbackItems],
  )
  const feedbackById = useMemo(() => new Map(allFeedbackItems.map((f) => [f.id, f])), [allFeedbackItems])

  // 워크벤치 진입 시(원문+피드백이 둘 다 준비되면) 자동으로 인용 조회를 호출한다.
  useEffect(() => {
    if (pdfUrl === null || feedbackItems.length === 0) return
    const lookups = feedbackItems.filter((f) => f.chunkId)
    if (lookups.length === 0) return
    let cancelled = false
    setMatchingLoading(true)
    setMatchingError('')
    getQuoteMatches(projectId, lookups)
      .then((matches) => { if (!cancelled) setQuoteMatches(matches) })
      .catch((err) => { if (!cancelled) setMatchingError(err.message) })
      .finally(() => { if (!cancelled) setMatchingLoading(false) })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, pdfUrl, reviewOutput])

  // context-check는 자체 검증된 인용문(quote)을 이미 들고 있어 /quotes 왕복이 필요
  // 없다 - 위원 피드백의 quoteMatches와 그대로 합쳐서 PdfDocumentView에 하나로 넘긴다.
  // 둘 다 아직 준비 안 됐을 때만 null을 유지해 "아직 하이라이트 시도할 거 없음" 신호를
  // 보존한다(빈 배열은 truthy라 null과 구분해야 함).
  const combinedQuoteMatches = useMemo(() => {
    const contextQuoteMatches = (contextFindings || []).map((f) => ({ id: f.id, quote: f.quote, found: true }))
    const typoQuoteMatches = (typoFindings || []).map((f) => ({ id: f.id, quote: f.quote, found: true }))
    if (quoteMatches === null && contextQuoteMatches.length === 0 && typoQuoteMatches.length === 0) return null
    return [...(quoteMatches || []), ...contextQuoteMatches, ...typoQuoteMatches]
  }, [quoteMatches, contextFindings, typoFindings])

  const selectedFeedbackItems = selectedFeedbackIds
    .map((id) => feedbackById.get(id))
    .filter(Boolean)
  const unscoredItems = useMemo(() => extractUnscoredItems(reviewOutput), [reviewOutput])

  return (
    <div style={{ display: 'flex', gap: 20, height: 'calc(100vh - 64px)' }}>
      <style>{`
        .wb-pdf-page { position: relative; background: #fff; box-shadow: 0 2px 18px rgba(0,0,0,0.12); }
        .wb-pdf-page canvas { display: block; }
        /* 재인/Claude(2026-07-21): 하이라이트는 pdf.js textLayer의 개별 텍스트 조각(span)
           하나하나에 붙는다 - 조각마다 outline/border-radius를 따로 그리면 이어지는
           문장이어도 조각 경계마다 끊긴 것처럼 보인다("|"로 잘린 느낌). 배경색만 칠하고
           테두리·모서리를 없애면 인접한 조각들이 하나로 이어져 보인다. */
        .wb-pdf-highlight { cursor: pointer; pointer-events: auto; }
        .wb-pdf-highlight-issue { background: var(--coral-dim); }
        .wb-pdf-highlight-suggestion { background: var(--green-dim); }
        .wb-pdf-highlight-context { background: var(--amber-dim); }
        .wb-pdf-highlight-typo { background: var(--rose-dim); }
        /* "선택됨" 표시는 조각(span)마다 칠하지 않고, PdfDocumentView.jsx가 줄 단위로
           묶어 계산한 통짜 사각형을 이 레이어 위에 그린다(Google Docs/Notion 댓글
           표시 느낌) - pointer-events:none이라 클릭은 그대로 아래 textLayer가 받는다. */
        .wb-pdf-selection-layer { position: absolute; inset: 0; pointer-events: none; }
        .wb-pdf-selection-box {
          position: absolute;
          border: 1.5px solid var(--purple);
          background: rgba(124, 92, 234, 0.14);
          border-radius: 5px;
          box-shadow: 0 2px 10px rgba(124, 92, 234, 0.22);
          animation: wb-selection-in 0.16s ease-out;
        }
        @keyframes wb-selection-in {
          from { opacity: 0; transform: scale(0.97); }
          to { opacity: 1; transform: scale(1); }
        }
      `}</style>

      {/* 중앙: 기획서 원문(PDF, 워드/한글 원본과 같은 페이지 모습) + 하이라이트 */}
      <div
        className="card glass"
        style={{ flex: 2, minWidth: 0, overflowY: 'auto', position: 'relative', padding: '28px 0' }}
      >
        <div className="badge coral mono" style={{ marginBottom: 14, marginLeft: 28 }}>기획서 원문</div>
        {docError && <p style={{ color: 'var(--coral)', fontSize: 13, marginLeft: 28 }}>{docError}</p>}
        {matchingError && <p style={{ color: 'var(--coral)', fontSize: 13, marginLeft: 28 }}>인용 조회 실패: {matchingError}</p>}
        {!docError && pdfUrl === null && <p style={{ color: 'var(--text-2)', fontSize: 13, marginLeft: 28 }}>불러오는 중...</p>}

        {pdfUrl !== null && matchingLoading && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(250,248,244,0.85)', backdropFilter: 'blur(2px)', fontSize: 13, color: 'var(--text-1)',
            zIndex: 10,
          }}>
            AI 위원이 기획서를 짚어보는 중...
          </div>
        )}

        {pdfUrl !== null && (
          <PdfDocumentView
            pdfUrl={pdfUrl}
            authHeaders={authHeader()}
            quoteMatches={combinedQuoteMatches}
            feedbackById={feedbackById}
            selectedFeedbackIds={selectedFeedbackIds}
            onSelectFeedback={setSelectedFeedbackIds}
          />
        )}
      </div>

      {/* 오른쪽: 선택한 피드백 상세 (재인/Claude 2026-07-21: 아바타 D 연동 자리는
          팀 논의로 이 화면에서 빼기로 함 - committee_video_streaming_architecture.md
          쪽 스트리밍 연동은 이 워크벤치가 아닌 다른 화면에서 다룰 예정) */}
      <div style={{ flex: 1, minWidth: 280, display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* 재인/Claude(2026-07-22): 분량(공고문 요구 페이지 vs 실제) + 밀도(빽빽함) 요약.
            문서 전체 단위 체크라 맨 위에 둔다. 문제 있을 때만 눈에 띄게 색을 준다. */}
        {formatCheck && (formatCheck.page_message || formatCheck.density_message) && (
          <div
            className="card glass"
            style={{
              flex: '0 0 auto', padding: 14,
              border: `1px solid ${(formatCheck.page_verdict === '부족' || formatCheck.overall_verdict === '부족')
                ? 'var(--coral-dim)' : 'var(--green-dim)'}`,
            }}
          >
            <div className="badge mono" style={{ marginBottom: 10, background: 'var(--bg-2)', color: 'var(--text-1)' }}>
              📄 분량 · 밀도 체크
            </div>
            {formatCheck.page_message && (
              <div style={{ marginBottom: formatCheck.density_message ? 10 : 0 }}>
                <div style={{
                  fontSize: 12.5, fontWeight: 600, marginBottom: 3,
                  color: formatCheck.page_verdict === '부족' ? 'var(--coral)'
                    : formatCheck.page_verdict === '충족' ? 'var(--green)' : 'var(--text-1)',
                }}>
                  분량: {formatCheck.actual_pages}p
                  {formatCheck.required_max
                    ? ` / 기준 ${formatCheck.required_min !== formatCheck.required_max
                        ? `${formatCheck.required_min}~${formatCheck.required_max}` : formatCheck.required_max}p`
                    : ''}
                  {formatCheck.page_verdict ? ` (${formatCheck.page_verdict})` : ''}
                </div>
                <p style={{ fontSize: 12, lineHeight: 1.6, color: 'var(--text-2)' }}>{formatCheck.page_message}</p>
              </div>
            )}
            {formatCheck.density_message && (
              <div>
                <div style={{
                  fontSize: 12.5, fontWeight: 600, marginBottom: 3,
                  color: formatCheck.overall_verdict === '부족' ? 'var(--coral)' : 'var(--green)',
                }}>
                  밀도: 평균 {Math.round((formatCheck.overall_coverage || 0) * 100)}% 채움
                  {formatCheck.overall_verdict ? ` (${formatCheck.overall_verdict})` : ''}
                </div>
                <p style={{ fontSize: 12, lineHeight: 1.6, color: 'var(--text-2)' }}>{formatCheck.density_message}</p>
              </div>
            )}
          </div>
        )}

        {unscoredItems.length > 0 && (
          <div
            className="card glass"
            style={{ flex: '0 0 auto', maxHeight: '40%', overflowY: 'auto', padding: 14, border: '1px solid var(--amber-dim)' }}
          >
            <div className="badge mono" style={{ marginBottom: 10, background: 'var(--amber-dim)', color: 'var(--amber)' }}>
              <AlertTriangle size={12} /> 평가되지 않은 항목 {unscoredItems.length}개
            </div>
            {unscoredItems.map((u) => (
              <div key={u.id} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid var(--border, #eee)' }}>
                <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 4 }}>{u.criterionName}</div>
                <p style={{ fontSize: 12, lineHeight: 1.6, color: 'var(--text-1)', marginBottom: u.strengths?.length || u.weaknesses?.length ? 6 : 0 }}>
                  {u.message}
                </p>
                {(u.strengths?.length > 0 || u.weaknesses?.length > 0) && (
                  <div style={{ fontSize: 11.5, color: 'var(--text-2)', lineHeight: 1.6 }}>
                    {u.strengths?.map((s, i) => <div key={`s-${i}`}>+ {s}</div>)}
                    {u.weaknesses?.map((w, i) => <div key={`w-${i}`}>- {w}</div>)}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* 재인/Claude(2026-07-22): 사용자 요청으로 맥락 이상 감지와 섞이지 않는 별도
            "오탈자 의심" 카드 - 위 "평가되지 않은 항목" 카드와 같은 패턴(항상 노출되는
            요약 목록). 항목을 클릭하면 원문 하이라이트를 클릭한 것과 동일하게 선택되어
            아래 상세 패널과 PDF 하이라이트에 반영된다. */}
        {typoFindings && typoFindings.length > 0 && (
          <div
            className="card glass"
            style={{ flex: '0 0 auto', maxHeight: '40%', overflowY: 'auto', padding: 14, border: '1px solid var(--rose-dim)' }}
          >
            <div className="badge mono" style={{ marginBottom: 10, background: 'var(--rose-dim)', color: 'var(--rose)' }}>
              <PenLine size={12} /> 오탈자 의심 {typoFindings.length}건
            </div>
            {typoFindings.map((f) => (
              <div
                key={f.id}
                onClick={() => setSelectedFeedbackIds([f.id])}
                style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid var(--border, #eee)', cursor: 'pointer' }}
              >
                <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 4 }}>
                  '{f.quote}' → '{f.corrected}'
                </div>
                {f.message && (
                  <p style={{ fontSize: 12, lineHeight: 1.6, color: 'var(--text-1)' }}>{f.message}</p>
                )}
              </div>
            ))}
          </div>
        )}

        <div className="card glass" style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
          {selectedFeedbackItems.length === 0 && (
            <div style={{ color: 'var(--text-2)', fontSize: 13, display: 'flex', alignItems: 'center', gap: 8 }}>
              <MessageSquare size={16} /> 원문의 하이라이트를 클릭하면 상세 코멘트가 여기 표시됩니다.
            </div>
          )}
          {selectedFeedbackItems.map((item) => {
            const badge = KIND_BADGE[item.kind] || KIND_BADGE.issue
            return (
            <div key={item.id} style={{ paddingBottom: 14, borderBottom: '1px solid var(--border, #eee)' }}>
              <div className="badge mono" style={{ marginBottom: 10, background: badge.bg, color: badge.fg }}>
                {badge.icon}
                {/* criterionName이 배지 이름과 같으면(오탈자/맥락 이상처럼) 중복이라 생략하고,
                    지적사항/제안처럼 실제 평가기준명일 때만 뒤에 붙인다. */}
                {item.criterionName && item.criterionName !== badge.label
                  ? `${badge.label} · ${item.criterionName}`
                  : badge.label}
              </div>
              <p style={{ fontSize: 13.5, lineHeight: 1.6, marginBottom: 12 }}>{item.text}</p>
              <div style={{ fontSize: 11.5, color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Sparkles size={12} /> {item.reviewerName}
              </div>
            </div>
            )
          })}
        </div>
      </div>

      {/* 경이/Claude(2026-07-24): AI 피드백 → 완성 리포트 이동 버튼. WorkbenchScreen에 다음
          버튼이 없어 사이드바로만 넘어갈 수 있었다("안 넘어간다" 제보). 스크롤과 무관하게
          항상 보이도록 우하단 고정 배치. onNext(=goNext)로 report 단계로 진행한다. */}
      {onNext && (
        <button
          type="button"
          className="btn-primary"
          onClick={onNext}
          style={{
            position: 'fixed', right: 28, bottom: 24, zIndex: 50,
            display: 'inline-flex', alignItems: 'center', gap: 8,
            boxShadow: '0 6px 20px rgba(124,92,234,0.35)',
          }}
        >
          완성 리포트로 <ArrowRight size={16} />
        </button>
      )}
    </div>
  )
}
