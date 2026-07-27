// 작성자: 경이 (테스트 전용 — 별도 섹션. 가은님 새 디자인/board 프로토타입은 건드리지 않음)
// 목적: "1회성 답변이 아닌 버전 추적형 User RAG = 개인 맞춤형 피드백 루프"를 프론트에서 먼저
//   눈으로 검증하는 실험 화면. 도메인은 IT(기술 기반) 공모전, 심사위원은 기획 위원 + 개발
//   위원 2인. v1.0만 먼저 보이고 "다음 수정본 제출"로 v1.1 -> v1.2 -> v1.3 을 하나씩 쌓아가며,
//   위원 탭으로 각 위원의 점수·피드백만 골라 보고, 항목별 이전 vs 현재 막대 비교로 점수
//   상승세와 해결 과정을 한눈에 본다.
//   ★ 디자인: 가은님의 새 웜 화이트/글래스 톤(ReviewBoardPrototype .rb-root)에 맞춘
//   VersionTrackerTest.css(.vt-root)를 그대로 쓴다 — /board 플로우에 이어붙이기 쉽게.
//   ★ 실데이터 매핑: 각 버전의 위원별 피드백 = review_output.reviewer_results,
//   버전 간 점수 증감·해결/잔존/신규 = ai/meeting/scoring/comparison.py 의
//   build_revision_comparison() 출력. 둘 다 내 백엔드 산출물이라 mock -> 실데이터 교체만 하면 됨.
// import: react, react-router-dom, lucide-react(가은 새 디자인과 동일 아이콘), 스타일 CSS.

import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ArrowLeft, TrendingUp, TrendingDown, CheckCircle2, AlertCircle, Plus,
  Lightbulb, Compass, Cpu, FlaskConical,
  AlertTriangle, Zap, ChevronDown, FileText, Info,
} from 'lucide-react'
import { getMyProfile } from '../api/profileApi'
import { getProjectReport, getProjectComparison, analyzeProject, getAnalyzeProgress } from '../api/projectApi'
import { uploadDocument, getDocuments, deleteDocument, getDocumentStatus } from '../api/documentApi'
import { getTypoCheck, getContextCheck, getFormatCheck } from '../api/workbenchApi'
import './VersionTrackerTest.css'

const CRITERION_MAX = 25 // 4항목 × 25 = 100점

// --- 개인화 입력: 제출 프로필 (TEST 2종) -----------------------------------
// "버전 추적형 User RAG = 개인 맞춤형 피드백 루프"를 살리기 위해, 수정본 제출(기본)에 더해
// GitHub·이력/교육수준을 함께 제출한다. 개발 위원 피드백은 이 프로필에 따라 구현 난이도와
// 설명 상세도를 다르게 보여준다.
const PROFILES = {
  nonmajor: {
    key: 'nonmajor',
    label: '비전공자',
    difficulty: 'hard',
    education: '경영학 학사 졸업',
    github: 'github.com/user · 커밋 32, HTML/CSS 위주 (백엔드 이력 없음)',
    experience: '공모전·IT 인턴 경험 없음',
  },
  major: {
    key: 'major',
    label: '컴퓨터공학 전공자',
    difficulty: 'easy',
    education: '컴퓨터공학 학사 졸업',
    github: 'github.com/user · 커밋 480, Python·FastAPI·React, RAG 토이프로젝트 2건',
    experience: 'IT 스타트업 백엔드 인턴 6개월 · 교내 해커톤 수상 1회',
  },
}

// 백엔드 scoring.classify_impl_difficulty 와 동일한 3단계 → 색/아이콘.
const DIFFICULTY = {
  hard: { color: '#e0603d', bg: 'rgba(224,96,61,0.1)', Icon: AlertTriangle },
  moderate: { color: '#b8830b', bg: 'rgba(184,131,11,0.12)', Icon: AlertTriangle },
  easy: { color: '#16a37a', bg: 'rgba(22,163,122,0.1)', Icon: Zap },
}
// personalization.py 의 _LABEL_BY_LEVEL / _VERBOSITY_BY_LEVEL 과 1:1.
const DIFFICULTY_LABEL = { hard: '구현 난이도 · 어려울 수 있음', moderate: '구현 난이도 · 보통', easy: '구현 난이도 · 쉬움' }
const VERBOSITY_BY_LEVEL = { hard: 'detailed', moderate: 'standard', easy: 'brief' }

// 개발 위원 지적(미해결/신규)에 대한 구현 가이드 산문(prose). profile별 상세도가 다르다.
// 실서비스에선 이 산문을 scoring.build_impl_guide 의 llm_call 이 생성한다 — 여기선 mock.
// nonmajor = 길고 친절한 단계별(자세히 보기), major = 짧고 간결한 한 줄.
const IMPL_GUIDE = {
  'f-stack': {
    nonmajor: '모델·DB 선택은 비전공자에게 가장 막막한 부분이에요. ① 임베딩은 한국어에 강한 KURE-v1을 그대로 쓰세요(직접 학습 X, HuggingFace에서 이름만 지정). ② 벡터DB는 설치가 쉬운 Chroma를 로컬 폴더에 저장(persistent)하도록 한 줄 설정. ③ "top-k=5"는 "질문과 가장 비슷한 문단 5개를 가져온다"는 뜻이니 5로 두면 됩니다. ④ LLM은 요약엔 저렴한 모델, 최종 답변엔 좋은 모델로 나누면 비용이 절반 이하로 줄어요. 각 단계는 공식 문서 예제 코드를 복사해 이름만 바꾸면 동작합니다.',
    major: 'KURE-v1 + Chroma(persistent, top-k=5) + LLM 2티어 라우팅. 3장 다이어그램에 스택·파라미터만 표기하면 됩니다.',
  },
  'f-eval': {
    nonmajor: '"검색이 잘 되는지"를 숫자로 보여주는 부분이에요. ① 정답이 있는 질문 30개를 미리 만드세요(질문 ↔ 답이 나와야 하는 문단). ② 시스템이 가져온 문단 5개 안에 정답 문단이 있으면 성공 → 30개 중 성공 개수가 "Recall@5"입니다(예: 27/30=0.9). ③ 답변이 그 근거를 실제로 인용했는지는 몇 개만 눈으로 확인하면 됩니다. 엑셀 표 하나로도 충분해요.',
    major: '골든셋 30건으로 Recall@5·근거 인용 정확도만 4장에 표로. pytest 파라미터라이즈로 자동화하면 30분.',
  },
  'a-parse': {
    nonmajor: '업로드한 파일에서 글자를 뽑아 잘게 나누는 과정이에요. ① PyMuPDF는 PDF에서 텍스트를 뽑는 무료 라이브러리로 예제 5줄이면 됩니다. ② "청킹"은 긴 글을 800토큰(대략 한글 1,000자)씩 자르는 것, "overlap 100"은 자를 때 앞뒤를 조금 겹쳐 문맥이 끊기지 않게 하는 안전장치예요. ③ 각 조각에 "몇 쪽에서 왔는지"만 같이 저장하면 나중에 출처를 보여줄 수 있어요. 도식은 네모 4개(파일→텍스트→조각→저장)를 화살표로 이으면 됩니다.',
    major: 'PyMuPDF → RecursiveCharacterTextSplitter(800/100) → page 메타 보존. 3장에 4-스텝 다이어그램만.',
  },
  'a-privacy': {
    nonmajor: '개인정보를 안전하게 다루는 규칙을 적는 부분이에요(코딩보다 정책 서술에 가깝습니다). ① 파일을 어디에 저장하는지, ② 저장 시 암호화 옵션을 켜는지, ③ 며칠 뒤 자동 삭제하는지(예: 30일), ④ "AI 학습에는 절대 사용하지 않음"을 5장에 문장으로 명시하면 됩니다. 기술 구현은 클라우드 설정 몇 개면 되고, 핵심은 "문서에 약속을 적는 것"이에요.',
    major: '저장 시 암호화(KMS)·TTL 30일·학습 배제 조항. 인프라 설정 + 5장 정책 문구. 반나절.',
  },
  'f-scale': {
    nonmajor: '"사람이 몰리면 돈이 얼마나 드나"를 추정하는 부분이에요. ① 동시에 몇 명이 쓸지 가정(예: 최대 20명), ② 회의 1건에 토큰이 대략 얼마 드는지 API 가격표로 계산(예: 회의 1건 ≈ 100원), ③ 같은 문서를 또 물으면 캐시로 재사용해 비용을 아끼는 전략만 한 단락 적으면 됩니다. 실제 부하 테스트까진 필요 없고 표 하나면 충분해요.',
    major: '동시요청 가정 → 월 토큰·비용 산정표 + 응답 캐싱/요청 큐잉 한 단락. 6장에 추정치로.',
  },
  'a-fallback': {
    nonmajor: 'AI나 외부 서비스가 잠깐 죽었을 때 앱이 멈추지 않게 하는 안전장치예요. ① 응답이 너무 늦으면 몇 초 후 포기(타임아웃), ② 실패하면 한두 번 다시 시도(재시도), ③ 좋은 모델이 안 되면 저렴한 모델로 대신(폴백), ④ 그래도 안 되면 "잠시 후 다시 시도해주세요" 안내. 이 4가지를 순서도로 3장에 그리면 되고, 대부분 라이브러리 옵션으로 처리돼요.',
    major: 'timeout → retry(backoff) → 모델 폴백(고성능→경량) → user-facing 에러. 3장에 시퀀스만. tenacity로 구현.',
  },
}

// 위원 탭 색 톤(경이 요청, 2026-07-22): 기획 위원 = 연보라 그라데이션, 개발 위원 = 연분홍
// 그라데이션. gradient는 활성 탭 배경/강조에, color/soft는 텍스트·아이콘·테두리 등 단색이
// 필요한 곳에 쓴다. (구현 난이도 hard/moderate/easy 색(빨강/노랑/초록)은 위원 색과 별개의
// 의미축이라 그대로 둔다.)
const COMMITTEES = {
  planning: {
    name: '기획 위원', Icon: Compass, desc: '문제 정의 · 사용자 가치 · 차별성',
    color: '#7c5cea', soft: '#a78bfa', dim: 'rgba(124,92,234,0.12)',
    gradient: 'linear-gradient(135deg, #b7a3f4 0%, #8b6ff0 100%)',
    // 항목 막대(이전 vs 현재)·현재 점수 숫자 색 — 위원 톤을 따라간다.
    bar: { grad: 'linear-gradient(90deg,#9b82f0,#7c5cea)', faint: '#d8cff0', num: '#7c5cea' },
  },
  dev: {
    name: '개발 위원', Icon: Cpu, desc: '기술 구현 · 아키텍처 · 데이터',
    color: '#d65a9c', soft: '#f0a6c9', dim: 'rgba(214,90,156,0.14)',
    gradient: 'linear-gradient(135deg, #f7abcc 0%, #e06aa6 100%)',
    bar: { grad: 'linear-gradient(90deg,#f0a6c9,#d65a9c)', faint: '#f3d5e6', num: '#d65a9c' },
  },
}

// 색 정리(경이 요청, 2026-07-22): 배경마다 색이 달라 지저분해서, 배경은 전부 연한 베이지
// (#f5f0e3)로 통일하고 선(border)·글씨·아이콘만 의미색으로 남긴다. 신규 지적은 코랄(붉은
// 기) 대신 부드러운 주황(#e2882e)으로.
const _BEIGE = '#f5f0e3'
const STATUS_META = {
  open: { Icon: AlertCircle, label: '보완 필요', color: '#b8830b', bg: _BEIGE, border: 'rgba(184,131,11,0.45)' },
  new: { Icon: Plus, label: '신규 지적', color: '#e2882e', bg: _BEIGE, border: 'rgba(226,136,46,0.5)' },
  resolved: { Icon: CheckCircle2, label: '해결됨', color: '#16a37a', bg: _BEIGE, border: 'rgba(22,163,122,0.45)' },
}

// --- Mock 데이터 (버전 추적 스토리: 50 -> 72 -> 86 -> 95) ------------------
const ALL_VERSIONS = [
  {
    version: 'v1.0', label: '최초 제출', submitted_at: '2026-07-15 14:20', total_score: 50,
    criteria: [
      {
        id: 'problem', name: '문제 정의 · 차별성', committee: 'planning', score: 14, judgment: 'needs_improvement',
        feedback: [
          { id: 'p-persona', status: 'open', text: '타깃 사용자가 "예비창업 대학생"인지 "일반 소상공인"인지 문서 전반에서 혼재됩니다.', suggestion: '1장 도입부에 페르소나 1명(예: 예비창업 대학생 김OO, 25세)으로 좁혀 정의하고, 그 사용자의 핵심 문제를 한 문장으로 제시하세요.' },
          { id: 'p-diff', status: 'open', text: '경쟁 서비스 대비 차별점이 "더 똑똑함" 수준의 정성적 서술뿐입니다.', suggestion: '기능 4개 × 경쟁사 2곳 비교표를 넣어 정량적 차별점(무엇이 얼마나 다른지)을 보이세요.' },
        ],
      },
      {
        id: 'impact', name: '기대효과 · 사용자 가치', committee: 'planning', score: 12, judgment: 'needs_improvement',
        feedback: [
          { id: 'i-metric', status: 'open', text: '기대효과가 "효율 향상" 같은 추상적 표현에 그칩니다.', suggestion: '"피드백 1회 소요 3시간 → 20분", "재수정률 30% 감소"처럼 측정 가능한 지표 2개로 바꾸세요.' },
        ],
      },
      {
        id: 'feasibility', name: '기술 실현 가능성', committee: 'dev', score: 13, judgment: 'critical_risk',
        feedback: [
          { id: 'f-stack', status: 'open', text: '사용할 LLM · 임베딩 모델 · 벡터DB가 특정되어 있지 않습니다.', suggestion: '3장 아키텍처에 "임베딩=KURE-v1, 벡터DB=Chroma(persistent), 검색 top-k=5, LLM=경량/고성능 2단계 분리"처럼 스택과 파라미터를 명시하세요.' },
          { id: 'f-eval', status: 'open', text: 'RAG 검색 정확도를 어떻게 검증할지 계획이 없습니다.', suggestion: '평가셋 30건 + Recall@5 · 근거 인용 정확도 지표로 측정하는 검증 절차를 4장에 추가하세요.' },
        ],
      },
      {
        id: 'architecture', name: '아키텍처 · 데이터 처리', committee: 'dev', score: 11, judgment: 'critical_risk',
        feedback: [
          { id: 'a-parse', status: 'open', text: '업로드 문서(PDF/DOCX/PPT)의 파싱 · 청킹 전략이 빠져 있습니다.', suggestion: '"PyMuPDF 파싱 → 800토큰/overlap 100 청킹 → 출처 페이지 메타 보존" 파이프라인을 도식으로 3장에 넣으세요.' },
          { id: 'a-privacy', status: 'open', text: '사업계획서 내 민감정보 처리 방침이 없습니다.', suggestion: '저장 위치 · 암호화 · 보관기간(예: 30일)과 "모델 학습 미사용"을 5장 보안 절에 명시하세요.' },
        ],
      },
    ],
  },
  {
    version: 'v1.1', label: '1차 수정본', submitted_at: '2026-07-18 10:15', total_score: 72,
    criteria: [
      {
        id: 'problem', name: '문제 정의 · 차별성', committee: 'planning', score: 20, judgment: 'acceptable',
        feedback: [
          { id: 'p-persona', status: 'resolved', text: '타깃 사용자 혼재', note: '1장에 페르소나(예비창업 대학생 김OO, 25세)로 좁히고 핵심 문제를 한 문장으로 제시함.' },
          { id: 'p-diff', status: 'resolved', text: '차별점이 정성적', note: '경쟁사 2곳 × 기능 4개 비교표를 2장에 추가함.' },
        ],
      },
      {
        id: 'impact', name: '기대효과 · 사용자 가치', committee: 'planning', score: 17, judgment: 'acceptable',
        feedback: [
          { id: 'i-metric', status: 'resolved', text: '기대효과가 추상적', note: '"소요 3시간→20분", "재수정률 30%↓" 정량 지표로 교체함.' },
          { id: 'i-evidence', status: 'new', text: '제시한 지표의 근거(측정 방법 · 표본)가 없어 신뢰도가 낮습니다.', suggestion: '파일럿 5명 대상 사전측정값을 각주로 붙여 지표의 출처를 밝히세요.' },
        ],
      },
      {
        id: 'feasibility', name: '기술 실현 가능성', committee: 'dev', score: 19, judgment: 'acceptable',
        feedback: [
          { id: 'f-stack', status: 'resolved', text: '모델 · 벡터DB 미명시', note: '3장에 KURE-v1 + Chroma(top-k=5), LLM 2단계 분리를 명시함.' },
          { id: 'f-eval', status: 'open', text: 'RAG 검색 정확도 검증 절차가 아직 없습니다.', suggestion: '평가셋 30건 + Recall@5 · 근거 인용 정확도 지표로 4장에 검증 절차를 추가하세요.' },
        ],
      },
      {
        id: 'architecture', name: '아키텍처 · 데이터 처리', committee: 'dev', score: 16, judgment: 'needs_improvement',
        feedback: [
          { id: 'a-parse', status: 'resolved', text: '파싱 · 청킹 전략 부재', note: 'PyMuPDF 파싱 → 800토큰/overlap 100 청킹 파이프라인 도식을 3장에 추가함.' },
          { id: 'a-privacy', status: 'open', text: '민감정보 처리 방침이 아직 없습니다.', suggestion: '저장 암호화 · 보관 30일 · 학습 미사용을 5장 보안 절에 명시하세요.' },
        ],
      },
    ],
  },
  {
    version: 'v1.2', label: '2차 수정본', submitted_at: '2026-07-21 09:40', total_score: 86,
    criteria: [
      { id: 'problem', name: '문제 정의 · 차별성', committee: 'planning', score: 23, judgment: 'strong', feedback: [] },
      {
        id: 'impact', name: '기대효과 · 사용자 가치', committee: 'planning', score: 21, judgment: 'strong',
        feedback: [
          { id: 'i-evidence', status: 'resolved', text: '지표 근거 부재', note: '파일럿 5명 사전측정값을 각주로 추가해 지표 신뢰도를 확보함.' },
        ],
      },
      {
        id: 'feasibility', name: '기술 실현 가능성', committee: 'dev', score: 22, judgment: 'strong',
        feedback: [
          { id: 'f-eval', status: 'resolved', text: 'RAG 검증 절차 부재', note: '평가셋 30건 + Recall@5 · 근거 인용 정확도 검증 절차를 4장에 추가함.' },
          { id: 'f-scale', status: 'new', text: '동시 사용자 부하와 LLM 호출 비용 추정이 없습니다.', suggestion: '예상 동시요청 수 기준 월 토큰 사용량 · 비용과 캐싱/큐잉 전략을 6장에 추정치로 넣으세요.' },
        ],
      },
      {
        id: 'architecture', name: '아키텍처 · 데이터 처리', committee: 'dev', score: 20, judgment: 'strong',
        feedback: [
          { id: 'a-privacy', status: 'resolved', text: '민감정보 처리 방침 부재', note: '5장에 저장 암호화 · 보관 30일 · 학습 미사용을 명시함.' },
          { id: 'a-fallback', status: 'new', text: 'LLM · 외부 API 장애 시 폴백 전략이 없습니다.', suggestion: '타임아웃 · 재시도 · 모델 폴백(고성능→경량) 순서와 실패 시 사용자 안내를 3장 아키텍처에 추가하세요.' },
        ],
      },
    ],
  },
  {
    version: 'v1.3', label: '3차 수정본', submitted_at: '2026-07-24 16:05', total_score: 95,
    criteria: [
      { id: 'problem', name: '문제 정의 · 차별성', committee: 'planning', score: 24, judgment: 'strong', feedback: [] },
      { id: 'impact', name: '기대효과 · 사용자 가치', committee: 'planning', score: 23, judgment: 'strong', feedback: [] },
      {
        id: 'feasibility', name: '기술 실현 가능성', committee: 'dev', score: 24, judgment: 'strong',
        feedback: [
          { id: 'f-scale', status: 'resolved', text: '부하 · 비용 추정 부재', note: '6장에 월 토큰 사용량 · 비용과 캐싱/큐잉 전략을 추정치로 추가함.' },
        ],
      },
      {
        id: 'architecture', name: '아키텍처 · 데이터 처리', committee: 'dev', score: 24, judgment: 'strong',
        feedback: [
          { id: 'a-fallback', status: 'resolved', text: '장애 폴백 전략 부재', note: '3장에 타임아웃 · 재시도 · 모델 폴백 순서와 사용자 안내를 추가함.' },
        ],
      },
    ],
  },
]

// --- 유틸 ------------------------------------------------------------------
function criterionBefore(versions, versionIndex, criterionId) {
  if (versionIndex === 0) return null
  const prev = versions[versionIndex - 1].criteria.find((c) => c.id === criterionId)
  return prev ? prev.score : null
}
function committeeScore(version, committee) {
  // 소수 배점 합산 시 부동소수점 잔여 오차 방지 — 소수 1자리 반올림.
  const sum = version.criteria.filter((c) => c.committee === committee).reduce((s, c) => s + c.score, 0)
  return Math.round(sum * 10) / 10
}

function useCountUp(target, duration = 850) {
  const [val, setVal] = useState(target)
  const fromRef = useRef(target)
  useEffect(() => {
    const from = fromRef.current
    if (from === target) return
    const start = performance.now()
    let raf
    const tick = (now) => {
      const p = Math.min(1, (now - start) / duration)
      const eased = 1 - Math.pow(1 - p, 3)
      setVal(Math.round(from + (target - from) * eased))
      if (p < 1) raf = requestAnimationFrame(tick)
      else fromRef.current = target
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, duration])
  return val
}
function CountUp({ value, className, style }) {
  return <span className={className} style={style}>{useCountUp(value)}</span>
}

// --- 표시 컴포넌트 ---------------------------------------------------------
function DeltaPill({ value, size = 'md' }) {
  // 부동소수점 잔여 오차(예: 24.700000000000003) 방지 — 표시 직전 소수 1자리로 반올림.
  const v = Math.round(value * 10) / 10
  const up = v > 0
  const flat = v === 0
  const color = flat ? '#918d9f' : up ? '#16a37a' : '#e0603d'
  const bg = flat ? 'rgba(145,141,159,0.12)' : up ? 'rgba(22,163,122,0.12)' : 'rgba(224,96,61,0.12)'
  const Icon = up ? TrendingUp : TrendingDown
  const pad = size === 'lg' ? '6px 14px' : '3px 9px'
  const fs = size === 'lg' ? 15 : 12
  return (
    <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 5, borderRadius: 99, fontWeight: 700, whiteSpace: 'nowrap', color, background: bg, padding: pad, fontSize: fs }}>
      {!flat && <Icon size={size === 'lg' ? 15 : 12} />}
      {flat ? '±0' : `${up ? '+' : ''}${v}점`}
    </span>
  )
}

// 이전 vs 현재 막대 비교 (핵심 시각화). accent={grad,faint,num}로 위원 톤(기획=보라/
// 개발=분홍)을 따라간다.
const _PLANNING_BAR = { grad: 'linear-gradient(90deg,#9b82f0,#7c5cea)', faint: '#d8cff0', num: '#7c5cea' }
function CompareBars({ before, after, max, animKey, accent = _PLANNING_BAR }) {
  const rows = before == null
    ? [{ label: '출발점', value: after, fill: accent.grad, strong: true, delay: 0 }]
    : [
        { label: '이전', value: before, fill: accent.faint, strong: false, delay: 0 },
        { label: '현재', value: after, fill: accent.grad, strong: true, delay: 0.15 },
      ]
  return (
    <div key={animKey} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {rows.map((r) => (
        <div key={r.label} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span className="mono" style={{ fontSize: 11, width: 40, flexShrink: 0, color: r.strong ? '#5b5770' : '#918d9f', fontWeight: r.strong ? 700 : 500 }}>{r.label}</span>
          <div style={{ flex: 1, height: 14, borderRadius: 999, background: 'var(--bg-2)', overflow: 'hidden', minWidth: 120 }}>
            <div className="vt-bar-seg" style={{ height: '100%', width: `${(r.value / max) * 100}%`, background: r.fill, animationDelay: `${r.delay}s` }} />
          </div>
          {/* 소수점 점수(6.67 등)가 고정폭을 넘쳐 "/10"과 붙어 보이던 문제 — 폭을 내용에 맞추고 간격을 띄운다 */}
          <span className="mono" style={{ fontWeight: 800, minWidth: 30, textAlign: 'right', whiteSpace: 'nowrap', flexShrink: 0, color: r.strong ? accent.num : '#918d9f', fontSize: r.strong ? 17 : 14 }}>{r.value}</span>
          <span className="mono" style={{ fontSize: 11, color: '#b6b1c2', whiteSpace: 'nowrap', flexShrink: 0 }}>/ {max}</span>
        </div>
      ))}
    </div>
  )
}

// "자세히 보기" 산문을 ①②③ 마커로 쪼개 [도입문 + 단계 배열]로 만든다 — 한 문단으로
// 뭉쳐 초보자가 읽기 힘든 걸 번호 단계 카드로 나눠 한눈에 보이게 한다(경이 요청, 2026-07-22).
// 마커가 없으면(전공자 간결 산문 등) steps=[]로 두고 호출부가 그대로 한 줄로 렌더한다.
const _CIRCLED_MARKERS = /[①②③④⑤⑥⑦⑧⑨]/
function parseGuideSteps(prose) {
  if (!prose) return { intro: '', steps: [] }
  // 1) 원문자 ①②③ 형식
  if (_CIRCLED_MARKERS.test(prose)) {
    const parts = prose.split(_CIRCLED_MARKERS)
    return { intro: (parts[0] || '').trim(), steps: parts.slice(1).map((s) => s.trim()).filter(Boolean) }
  }
  // 2) "1. 2. 3." 아라비아 숫자 리스트(실제 LLM 산문이 자주 쓰는 형식). 문장 중간 숫자
  //    (예: "top-k=5", "30일", "5명") 오탐을 막기 위해, 숫자 앞은 시작/공백/괄호이고 뒤는
  //    ".)" + 공백이며, 1부터 순차 증가(1,2,3…)하는 마커만 단계로 인정한다.
  const re = /(?:^|[\s(])([1-9])[.)]\s+/g
  const seq = []
  let m
  while ((m = re.exec(prose)) !== null) {
    const num = Number(m[1])
    if (num === seq.length + 1) {
      seq.push({ num, start: m.index + m[0].lastIndexOf(m[1]), end: re.lastIndex })
    }
  }
  if (seq.length >= 2) {
    const intro = prose.slice(0, seq[0].start).trim()
    const steps = seq
      .map((mk, i) => prose.slice(mk.end, i + 1 < seq.length ? seq[i + 1].start : prose.length).trim())
      .filter(Boolean)
    return { intro, steps }
  }
  return { intro: '', steps: [] }
}

// 구현 가이드 단계 뷰: 도입문 + 번호 단계 카드(왼쪽에서 하나씩 슬라이드-인). diff는
// DIFFICULTY[level]({color,bg,Icon}) — 단계 번호 배지 색을 난이도 색과 맞춘다.
function GuideSteps({ prose, diff }) {
  const { intro, steps } = parseGuideSteps(prose)
  if (steps.length === 0) {
    return (
      <div style={{ marginTop: 8, fontSize: 12.5, lineHeight: 1.75, color: '#4a4660', background: diff.bg, border: `1px solid ${diff.color}22`, padding: '11px 13px', borderRadius: 10 }}>
        {prose}
      </div>
    )
  }
  return (
    <div style={{ marginTop: 9 }}>
      {intro && (
        <div className="vt-step" style={{ animationDelay: '0ms', fontSize: 12.5, lineHeight: 1.65, color: '#5b5770', marginBottom: 9 }}>{intro}</div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
        {steps.map((st, i) => (
          <div key={i} className="vt-step" style={{ animationDelay: `${80 + i * 75}ms`, display: 'flex', gap: 10, alignItems: 'flex-start', background: '#fff', border: '1px solid rgba(28,26,46,0.07)', borderRadius: 11, padding: '9px 12px', boxShadow: '0 1px 4px rgba(28,26,46,0.04)' }}>
            <span className="mono" style={{ flexShrink: 0, width: 22, height: 22, borderRadius: '50%', background: diff.color, color: '#fff', fontSize: 12, fontWeight: 800, display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: 1 }}>{i + 1}</span>
            <span style={{ fontSize: 12.5, lineHeight: 1.6, color: '#3a3750', flex: 1 }}>{st}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// "왜 이 점수·이 피드백인가" — 지적(피드백) 단위 4단계 근거 플로우(경이 요청 2026-07-25 개편).
// 모든 인용은 색인된 실제 문서의 RAG evidence에서만 온다(지어낸 문장 절대 금지):
//   STEP 1 제출 문서에서 확인한 내용(파일명·p.N 인용) → STEP 2 왜 문제인가(공고문 기준표
//   파일명 + 배점 + 보조 자료 근거) → STEP 3 근거 종합 판정·점수 논리 → STEP 4 피드백
//   (개발 위원×비전공자는 난이도 '어려움' + '구체적 해결방안' 단계 애니메이션 토글).
// 인용 원문을 읽기 좋게 불릿 줄로 분해한다(경이 확정 2026-07-26 가독성 형식).
// PDF 표가 한 줄로 풀리며 생긴 잡음("※ <작성 요령>", "○ -")을 제거하고, ㅇ/○/숫자절 단위로
// 줄을 나눈다 — 원문 문장 자체는 바꾸지 않는다(재배열·잡음 제거만).
function splitQuoteLines(quote) {
  let t = ` ${quote || ''} `
  t = t.replace(/※\s*<\s*작성\s*요령\s*>/g, '\n')
  t = t.replace(/[○◦•▪]/g, '\n')
  t = t.replace(/\sㅇ\s/g, '\nㅇ ')
  t = t.replace(/\s(\d{1,2}\.\s)/g, '\n$1') // 절 제목(예: "5. 목표 달성도 및 성과")
  let lines = t.split('\n').map((s) => s.trim().replace(/^[-–—:*]\s*/, '')).filter((s) => s && !/^[-<>*※.…]+$/.test(s))
  // 불릿 앞에 남은 짧은 머리 조각(표 헤더 잔재, 예: "성과 지표 (KPI) 방안")은 버린다
  if (lines.length > 1 && !/^(ㅇ|\d{1,2}\.)/.test(lines[0]) && lines[0].length < 26) lines = lines.slice(1)
  return lines.map((s) => (/^(ㅇ|\d{1,2}\.)/.test(s) ? s : `ㅇ ${s}`))
}

// 보조 자료 인용 블록 — 파일명·페이지 번호를 헤더로, 내용은 불릿 줄로(마지막 사진 형식).
function SupportQuoteBlock({ q }) {
  const lines = splitQuoteLines(q.quote)
  return (
    <div style={{ marginTop: 8 }}>
      <div className="mono" style={{ fontSize: 11, color: '#918d9f', lineHeight: 1.6 }}>
        파일명: <span style={{ color: '#5b5770' }}>{q.source}</span>
        {q.page != null && <>{'  ·  '}페이지 번호: <span style={{ color: '#5b5770' }}>{q.page}p</span></>}
      </div>
      {q.section && <div style={{ fontSize: 12.5, fontWeight: 800, color: '#3a3750', marginTop: 3 }}>{q.section}</div>}
      <div style={{ marginTop: 2 }}>
        {lines.map((ln, i) => (
          <div key={i} style={{ fontSize: 12.5, color: '#3a3750', lineHeight: 1.7 }}>{ln}</div>
        ))}
      </div>
    </div>
  )
}

function WhyFeedbackFlow({ f, crit, rubricInfo, citations, statusColor, noticeName, guide }) {
  // 플로우 컬러는 위원 accent가 아니라 이 지적의 상태색(보완 필요=황토, 신규=주황)을 따른다
  // (경이 요청 2026-07-26 — 아코디언 프레임과 톤 통일).
  const flowColor = statusColor || '#7c5cea'
  const [guideOpen, setGuideOpen] = useState(false)
  // 보조 자료는 기본 접힘 — 헤더 탭을 클릭해야 세부 인용이 펼쳐진다(경이 요청 2026-07-26)
  const [supportOpen, setSupportOpen] = useState(false)
  // STEP 1 인용 통합·중복 제거(경이 확정 2026-07-25): 지적 전용 인용(f.ref)과 채점에 쓰인
  // 제출 문서 인용을 한 목록으로 합치되, 어떤 인용이 더 긴 인용에 통째로 포함되면(부분 문장)
  // 가장 완전한(긴) 인용만 남긴다 — 같은 문장이 3번씩 보이던 문제의 일반 해법.
  const normQ = (s) => (s || '').replace(/\s+/g, '')
  const pool = []
  if (f.ref?.quote) pool.push({ quote: f.ref.quote, page: f.ref.page ?? null, source: '' })
  for (const q of (citations || []).filter((q) => q.role === 'submission')) pool.push(q)
  const seenQ = new Set()
  const deduped = pool.filter((a, i) => {
    const ka = normQ(a.quote)
    if (!ka || seenQ.has(ka)) return false
    // 다른(더 긴) 인용에 포함되는 부분 문장이면 제거
    if (pool.some((b, j) => j !== i && normQ(b.quote) !== ka && normQ(b.quote).includes(ka))) return false
    seenQ.add(ka)
    return true
  })
  // 지적-근거 정합(경이 확인 2026-07-27): 제출 문서 인용은 "항목 단위"로 모이기 때문에, 이
  // 지적과 무관한 문단이 붙을 수 있다(예: '법적 제약' 지적 밑에 '구현 서비스 혁신성' 문단).
  // 지적의 핵심 주제 토큰이 하나도 없는 인용은 이 지적의 근거로 보여주지 않는다. 지적 전용
  // 검증 인용(f.ref)도 동일하게 검사한다(경이 확인 2026-07-27) — 위원 LLM이 다른 항목의
  // 문장을 지적에 붙이는 사례 실측('법적 제약·예산' 지적에 '수치 목표' 문장). 원문 게이트는
  // 존재만 검증하지 주제는 못 거르므로 여기서 거른다. 걸러서 하나도 안 남으면 "내용 부재"로
  // 정직하게 표기(그 부재 자체가 지적의 사유인 경우가 대부분).
  const issueTopics = issueTopicTokens(f.text)
  const subs = issueTopics.size
    ? deduped.filter((q) => [...issueTopics].some((t) => normQ(`${q.section || ''}${q.quote}`).includes(t)))
    : deduped
  // 보조 자료(그 외 공고 자료) 인용 — STEP 2 접힘 탭에서 표시. 중심 자료(공고문) 원문 청크
  // 인용은 배점표가 한 줄로 풀린 표 덤프·심사 절차 안내 같은 노이즈라 보여주지 않는다
  // (경이 X 표시 2026-07-27) — 중심 자료 블록은 파일명·배점·세부 기준만.
  const normC = (s) => (s || '').replace(/[^\p{L}\p{N}]/gu, '')
  let supports = (citations || []).filter((q) => q.role === 'support')
  // 소제목만 있는 청크(작성 요령 불릿이 이웃 청크로 잘려 내용이 없는 경우)는 정보가 없어 숨긴다
  supports = supports.filter((q) => {
    const sec = normC(q.section || '')
    if (!sec) return true
    const residue = normC(q.quote).split(sec).join('').replace(/\d+/g, '')
    return residue.length >= 8
  })
  // 같은 파일·페이지에서 한 블록이 다른 블록에 통째로 포함되면 더 완전한 쪽만 남긴다(중복 제거)
  supports = supports.filter((a, i) => !supports.some((b, j) => {
    if (j === i || a.source !== b.source || a.page !== b.page) return false
    const na = normC(a.quote)
    const nb = normC(b.quote)
    if (!na || !nb.includes(na)) return false
    return nb.length > na.length || j < i
  }))
  const diff = guide ? DIFFICULTY[guide.level] : null
  const quoteLine = (q, i) => (
    <div key={`${q.quote}-${i}`} style={{ fontSize: 12, color: '#5b5770', lineHeight: 1.65, marginTop: 3 }}>
      “{q.quote}”{q.page != null && <span className="mono" style={{ color: '#a8a4b2' }}> (p.{q.page})</span>}
      {q.source && <span style={{ color: '#a8a4b2' }}> — {q.source}</span>}
    </div>
  )
  const steps = [
    {
      icon: '📄', title: '제출 문서 근거',
      body: (
        <>
          {subs.length > 0 ? (
            <div>
              {subs.map(quoteLine)}
            </div>
          ) : (
            <span>내용 부재: 제출 문서에서 이 항목과 관련한 구체적 서술이 제시되지 않았습니다.</span>
          )}
        </>
      ),
    },
    {
      icon: '📋', title: '왜 문제인가 — 평가 기준 근거',
      body: (
        <>
          {/* 첫째, 중심 자료(파일명에 '공고문'이 들어간 파일) — 채점의 기준점 */}
          <div className="vt-step" style={{ animationDelay: '0.15s', padding: '8px 11px', borderRadius: 8, background: 'rgba(28,26,46,0.035)' }}>
            <span className="mono" style={{ fontSize: 11, fontWeight: 800, color: '#1c1a2e' }}>중심 자료{noticeName ? ` — ${noticeName}` : ''}</span>
            <div style={{ fontSize: 12.5, lineHeight: 1.65, marginTop: 2 }}>
              「{crit.name}」 <b>배점 {crit.max}점</b>{rubricInfo?.description ? ` — ${rubricInfo.description}` : ''}
            </div>
          </div>
          {/* 둘째, 보조 자료('공고문'이 아닌 공고 자료) — 중심 자료를 보완하는 세부 근거.
              의미 유사도(KURE score) 높은 순으로 표시하며, 파일명·페이지·불릿 구조로 가독성 있게. */}
          {supports.length > 0 && (
            <div className="vt-step" style={{ animationDelay: '0.45s', padding: '8px 11px', borderRadius: 8, background: 'rgba(28,26,46,0.025)', marginTop: 6 }}>
              {/* 접힘 탭 — 클릭하면 아래 세부 인용(파일명·페이지·불릿)이 펼쳐진다 */}
              <button type="button" onClick={() => setSupportOpen((v) => !v)}
                style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 6, background: 'none', border: 'none', cursor: 'pointer', padding: 0, textAlign: 'left' }}>
                <span className="mono" style={{ fontSize: 11, fontWeight: 800, color: '#1c1a2e', flex: 1 }}>보조 자료 — 중심 자료(공고문 기준)를 보완하는 세부 근거</span>
                <ChevronDown size={14} style={{ flexShrink: 0, color: '#918d9f', transform: supportOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.18s ease' }} />
              </button>
              {supportOpen && supports.map((q, i) => <SupportQuoteBlock key={`${q.source}-${q.page}-${i}`} q={q} />)}
            </div>
          )}
        </>
      ),
    },
    {
      icon: '⚖️', title: '근거를 종합한 판정과 점수',
      body: `위 제출 문서 근거와 공고문 기준을 함께 보면, 이 항목의 판정은 「${WHY_JUDGMENT_LABEL[crit.judgment] || crit.judgment}」 — 배점 ${crit.max}점 중 ${crit.score}점입니다.`
        + (crit.calibration ? ` 또한 근거 신호 부족으로 결정론적 상한 ${crit.calibration.cap_score}점이 적용되었습니다.` : ''),
    },
    {
      icon: '✏️', title: '이렇게 고치면 점수가 오릅니다',
      body: (
        <>
          <div>{f.suggestion || f.text}</div>
          {guide && diff && (
            <div style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11.5, fontWeight: 700, padding: '4px 10px', borderRadius: 99, color: diff.color, background: diff.bg }}>
                  <diff.Icon size={12} /> {guide.label}
                </span>
                {guide.verbosity === 'detailed' ? (
                  <button type="button" className="vt-tab" onClick={() => setGuideOpen((v) => !v)}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 12, fontWeight: 800, color: diff.color, background: 'transparent', border: `1.5px solid ${diff.color}44`, borderRadius: 99, cursor: 'pointer', padding: '4px 12px' }}>
                    구체적 해결방안 <ChevronDown size={13} style={{ transform: guideOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.18s ease' }} />
                  </button>
                ) : (
                  <span style={{ fontSize: 12.5, color: '#5b5770' }}>{guide.prose}</span>
                )}
              </div>
              {guide.verbosity === 'detailed' && guideOpen && <GuideSteps prose={guide.prose} diff={diff} />}
            </div>
          )}
        </>
      ),
    },
  ]
  return (
    <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
      {steps.map((s, i) => (
        <div key={i} className="vt-step" style={{ animationDelay: `${i * 0.4}s`, display: 'flex', gap: 10, padding: '11px 14px', background: 'rgba(255,255,255,0.85)', border: `1px solid ${flowColor}22`, borderLeft: `3px solid ${flowColor}`, borderRadius: 10 }}>
          <span style={{ fontSize: 16, flexShrink: 0 }}>{s.icon}</span>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 3 }}>
              <span className="mono" style={{ fontSize: 10, fontWeight: 800, color: '#1c1a2e' }}>STEP {i + 1}</span>
              <span style={{ fontSize: 12.5, fontWeight: 800, color: '#1c1a2e' }}>{s.title}</span>
            </div>
            <div style={{ fontSize: 12.5, color: '#5b5770', lineHeight: 1.65 }}>{s.body}</div>
          </div>
        </div>
      ))}
    </div>
  )
}

// 지적 항목 — 아코디언(기본 접힘): 헤더(상태 배지 + 지적 한 줄)를 클릭하면 내용(제안·가이드·
// "왜 이 점수·피드백인가요?")이 펼쳐진다(경이 요청 2026-07-25 — 길게 늘어놓지 않고 깔끔하게).
function FeedbackItem({ f, guide, crit, rubricInfo, citations, accent, noticeName }) {
  const [open, setOpen] = useState(false)
  const [whyOpen, setWhyOpen] = useState(false)
  const s = STATUS_META[f.status]
  const resolved = f.status === 'resolved'
  // 해결됨 행(경이 요청 2026-07-26): "직전 버전 지적 그대로(당시 상태 라벨 포함)"를 회색
  // 취소선으로 보여주고 맨 끝에 해결됨을 붙인다 — 이전 지적을 잘 고쳤다는 게 한눈에 보이게.
  const prevMeta = resolved ? STATUS_META[f.prevStatus === 'new' ? 'new' : 'open'] : null
  const Icon = (prevMeta || s).Icon
  // 헤더에는 지적 요지만 — 인용이 검증된 지적(f.ref)이면 "(p.N) '...' ..." 인용 문장 부분을
  // 잘라낸다(같은 인용이 STEP 1 근거에 다시 나와 중복되던 문제, 경이 확정 2026-07-25).
  const headText = (() => {
    if (!f.ref?.quote) return f.text
    const cut = f.text.split(/\(\s*p\.?\s*\d+\s*\)/)[0].trim().replace(/[,·;]+$/, '')
    return cut.length >= 4 ? cut : f.text
  })()
  return (
    <div style={{ borderRadius: 11, background: s.bg, border: `1px solid ${s.border}` }}>
      {/* 헤더 — 클릭으로 펼침/접힘(해결됨 행은 펼칠 내용이 없어 클릭·화살표 없음) */}
      <button type="button" onClick={() => { if (!resolved) setOpen((v) => !v) }}
        style={{ width: '100%', display: 'flex', alignItems: 'flex-start', gap: 10, padding: '11px 13px', background: 'none', border: 'none', cursor: resolved ? 'default' : 'pointer', textAlign: 'left' }}>
        <span style={{ flexShrink: 0, width: 20, height: 20, borderRadius: '50%', background: (prevMeta || s).color, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: 1 }}>
          <Icon size={12} strokeWidth={3} />
        </span>
        <span style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'baseline', gap: 7, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 11.5, fontWeight: 800, flexShrink: 0, color: resolved ? '#a8a4b2' : s.color, textDecoration: resolved ? 'line-through' : 'none' }}>
            {resolved ? `${f.prevVersion ? `${f.prevVersion} ` : ''}${prevMeta.label}` : s.label}
          </span>
          <span style={{ fontSize: 13.5, lineHeight: 1.5, fontWeight: 600, textDecoration: resolved ? 'line-through' : 'none', color: resolved ? '#a8a4b2' : '#3a3750' }}>{headText}</span>
          {resolved && (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, flexShrink: 0, fontSize: 12.5, fontWeight: 800, color: '#16a37a' }}>
              <CheckCircle2 size={14} strokeWidth={3} /> 해결됨
            </span>
          )}
        </span>
        {!resolved && <ChevronDown size={15} style={{ flexShrink: 0, marginTop: 3, color: s.color, transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.18s ease' }} />}
      </button>

      {open && !resolved && (
        <div className="vt-fade" style={{ padding: '0 13px 12px 43px' }}>
          {/* "이렇게 고치세요" 박스는 STEP 4와 중복이라 완전 제거(경이 확정 2026-07-25) —
              개선 제안은 "왜 이 점수·피드백인가요?"의 STEP 4에서만 보여준다 */}
          {/* 지적별 "Why?" — 근거 인용 포함 4단계 (검정 글씨 + 상태색 프레임, 경이 요청 2026-07-26) */}
          <button type="button" onClick={() => setWhyOpen((v) => !v)}
            style={{ marginTop: 8, display: 'inline-flex', alignItems: 'center', gap: 6, padding: '6px 12px', borderRadius: 99, border: `1.5px solid ${s.color}55`, background: whyOpen ? `${s.color}1a` : 'transparent', color: '#1c1a2e', fontSize: 12, fontWeight: 800, cursor: 'pointer' }}>
            <Lightbulb size={13} /> Why?
            <ChevronDown size={13} style={{ transition: 'transform 0.2s', transform: whyOpen ? 'rotate(180deg)' : 'none' }} />
          </button>
          {whyOpen && <WhyFeedbackFlow key={`why-${f.id}`} f={f} crit={crit} rubricInfo={rubricInfo} citations={citations} statusColor={s.color} noticeName={noticeName} guide={guide} />}
        </div>
      )}
    </div>
  )
}

// ★ E2E 교체 지점 ★
// 이 함수 하나가 백엔드 scoring.attach_impl_guides(dev_feedback, profile, llm_call) 의 응답으로
// 대체된다. 반환 형태를 백엔드와 1:1로 맞춰둠: { feedback_id, level, verbosity, label, prose }.
// 연동 시 여기 내부(mock)만 fetch 결과로 바꾸면 되고, 렌더링(FeedbackItem)은 그대로 동작한다.
// level 은 백엔드 classify_impl_difficulty 결과와 동일(비전공자=hard, 전공자=easy).
function personalizeGuide(feedback, profile) {
  if (!profile) return null
  if (feedback.status === 'resolved') return null // 해결된 지적은 구현할 게 없음
  const prose = IMPL_GUIDE[feedback.id]?.[profile.key]
  if (!prose) return null
  const level = profile.difficulty
  return { feedback_id: feedback.id, level, verbosity: VERBOSITY_BY_LEVEL[level], label: DIFFICULTY_LABEL[level], prose }
}

const WHY_JUDGMENT_LABEL = {
  strong: '우수', acceptable: '적정', needs_improvement: '보완 필요',
  critical_risk: '중대 리스크', insufficient_evidence: '근거 부족', not_applicable: '해당 없음',
}

function CriterionCard({ c, before, index, animKey, isDev, profile, accent, realGuides, citations, priority, rubricInfo, noticeName }) {
  // 우선순위 팝업 — "왜 이 순위인가"(판정·감점·근거 기반 점수 상한)를 배지 클릭 시 보여준다
  // (경이 요청 2026-07-25: 상한 배너를 카드에 늘어놓지 않고 우선순위 근거로 접어 넣기).
  const [prioOpen, setPrioOpen] = useState(false)
  // 점수 변화(+N점) 배지 클릭 팝업 — "이전 → 현재 점수가 왜·어떻게 변했는지"를 이 항목의
  // 실제 데이터(해결/신규/보완 필요 건수·판정·상한)로만 설명한다(경이 요청 2026-07-26).
  const [deltaOpen, setDeltaOpen] = useState(false)
  const delta = before == null ? null : c.score - before
  // 상태 필터가 걸려 있어도 건수는 전체 기준으로 센다(allFeedback = 필터 전 원본)
  const allFb = c.allFeedback || c.feedback || []
  const fCounts = allFb.reduce((a, f) => { a[f.status] = (a[f.status] || 0) + 1; return a }, {})
  // 실데이터 모드(realGuides): impl_guide는 criterion 단위 1개라, 개발 위원 항목의 "첫 미해결
  // 지적"에만 붙인다(criterion_id로 매칭). mock 모드: 미해결/신규 지적마다 프로필 기반 가이드.
  const firstOpenIdx = c.feedback.findIndex((f) => f.status === 'open' || f.status === 'new')
  const guideFor = (f, fi) => {
    if (realGuides) {
      if (!isDev || fi !== firstOpenIdx) return null
      return realGuides.get(c.id) || null
    }
    if (!isDev || (f.status !== 'open' && f.status !== 'new')) return null
    return personalizeGuide(f, profile)
  }
  return (
    // 우선순위 팝업이 열리면 이 카드를 형제 카드들 위로 올린다 — 각 카드가 vt-fade 애니메이션으로
    // 자체 스태킹 컨텍스트를 갖기 때문에, 팝업의 z-index만으로는 다음 카드에 가려진다(실측 2026-07-25).
    <div className="vt-fade card glass" style={{ animationDelay: `${index * 90}ms`, position: 'relative', zIndex: prioOpen || deltaOpen ? 40 : 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 9, minWidth: 0 }}>
          {priority != null && (
            <span style={{ position: 'relative', flexShrink: 0 }}>
              <button type="button" className="mono" onClick={() => setPrioOpen((v) => !v)}
                title="왜 이 우선순위인지 보기"
                style={{ fontSize: 10.5, fontWeight: 800, padding: '3px 10px', borderRadius: 99, border: '1.5px solid transparent', cursor: 'pointer', background: priority === 1 ? 'rgba(224,96,61,0.14)' : 'rgba(28,26,46,0.07)', color: priority === 1 ? '#e0603d' : '#5b5770', outline: prioOpen ? `1.5px solid ${priority === 1 ? '#e0603d' : '#918d9f'}` : 'none' }}>
                우선순위 {priority} ⓘ
              </button>
              {/* 우선순위 근거 팝업 — 판정·감점 + 근거 기반 점수 상한(있을 때) */}
              {prioOpen && (
                <div className="vt-fade" style={{ position: 'absolute', top: 'calc(100% + 8px)', left: 0, zIndex: 30, width: 320, padding: '13px 15px', borderRadius: 12, background: '#fffdf9', border: '1px solid rgba(28,26,46,0.14)', boxShadow: '0 12px 30px rgba(28,26,46,0.18)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 7 }}>
                    <span style={{ fontSize: 12, fontWeight: 800 }}>우선순위 {priority} — 산정 근거</span>
                    <button type="button" onClick={() => setPrioOpen(false)} style={{ border: 'none', background: 'none', cursor: 'pointer', color: '#918d9f', fontSize: 13, fontWeight: 700 }}>✕</button>
                  </div>
                  <div style={{ fontSize: 12, color: '#5b5770', lineHeight: 1.7 }}>
                    · 판정 <b>「{WHY_JUDGMENT_LABEL[c.judgment] || c.judgment}」</b> · 감점 <b className="mono">{Math.round((c.max - c.score) * 10) / 10}점</b> (배점 {c.max}점 중 {c.score}점)
                    <div style={{ marginTop: 3, color: '#918d9f' }}>우선순위는 판정 심각도 → 감점 비율 순으로 정해집니다.</div>
                    {c.calibration && (
                      <div style={{ marginTop: 7, padding: '8px 10px', borderRadius: 9, background: 'rgba(224,96,61,0.08)', border: '1px solid rgba(224,96,61,0.2)', color: '#7a442f', fontSize: 11.5, lineHeight: 1.6 }}>
                        <b>근거 기반 점수 상한 적용:</b> 위원 제안 {c.calibration.original_score}점 → 상한 {c.calibration.cap_score}점
                        {(c.calibration.signals || []).length > 0 && <span> · {(c.calibration.signals || []).map((s) => s.reason).join(' · ')}</span>}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </span>
          )}
          <div style={{ fontSize: 15.5, fontWeight: 700 }}>{c.name}</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          {delta != null && (
            <span style={{ position: 'relative', flexShrink: 0 }}>
              <button type="button" onClick={() => setDeltaOpen((v) => !v)} title="이전 → 현재 점수 변화 근거 보기"
                style={{ border: 'none', background: 'none', padding: 0, cursor: 'pointer', display: 'inline-flex', outline: deltaOpen ? '1.5px solid #918d9f' : 'none', borderRadius: 99 }}>
                <DeltaPill value={delta} />
              </button>
              {/* 점수 변화 근거 팝업 — 해결/신규/잔여 건수 + 판정·상한 + 점수 체계 설명 */}
              {deltaOpen && (
                <div className="vt-fade" style={{ position: 'absolute', top: 'calc(100% + 8px)', right: 0, zIndex: 30, width: 340, padding: '13px 15px', borderRadius: 12, background: '#fffdf9', border: '1px solid rgba(28,26,46,0.14)', boxShadow: '0 12px 30px rgba(28,26,46,0.18)', textAlign: 'left' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 7 }}>
                    <span style={{ fontSize: 12, fontWeight: 800 }}>점수 변화 근거 — <span className="mono">{before} → {c.score}</span> (배점 {c.max}점)</span>
                    <button type="button" onClick={() => setDeltaOpen(false)} style={{ border: 'none', background: 'none', cursor: 'pointer', color: '#918d9f', fontSize: 13, fontWeight: 700 }}>✕</button>
                  </div>
                  <div style={{ fontSize: 12, color: '#5b5770', lineHeight: 1.7 }}>
                    {(fCounts.resolved || 0) > 0 && <div>· 이전 버전 지적 <b>{fCounts.resolved}건이 해결</b>되어 해당 감점 요인이 사라졌습니다.</div>}
                    {(fCounts.new || 0) > 0 && <div>· 이번 버전에서 <b>신규 지적 {fCounts.new}건</b>이 나와 상승 폭을 제한했습니다.</div>}
                    {(fCounts.open || 0) > 0 && <div>· <b>보완 필요 {fCounts.open}건</b>이 남아 있어, 반영하면 추가 상승 여지가 있습니다.</div>}
                    <div>· 이번 버전 판정은 <b>「{WHY_JUDGMENT_LABEL[c.judgment] || c.judgment}」</b> — 배점 <b className="mono">{c.max}점</b>에 판정별 점수 밴드 비율을 적용해 산정됩니다.</div>
                    {c.calibration && (
                      <div>· 근거 신호 부족으로 <b>결정론적 상한 {c.calibration.cap_score}점</b>이 적용되었습니다{c.calibration.original_score != null ? ` (위원 제안 ${c.calibration.original_score}점)` : ''}.</div>
                    )}
                    <div style={{ marginTop: 7, paddingTop: 7, borderTop: '1px dashed rgba(28,26,46,0.14)', color: '#918d9f' }}>
                      점수 체계: 위원이 문서 근거를 들어 판정(우수·적정·보완 필요·중대 리스크)을 내리면 판정별 점수 밴드로 환산되고, 지적이 해결될수록 판정이 올라가 점수가 상승합니다.
                    </div>
                  </div>
                </div>
              )}
            </span>
          )}
        </div>
      </div>

      <CompareBars before={before} after={c.score} max={c.max ?? CRITERION_MAX} animKey={animKey} accent={accent?.bar} />
      {/* 근거 기반 점수 상한 배너는 카드에 늘어놓지 않고 "우선순위 N ⓘ" 클릭 팝업으로 이동(2026-07-25) */}

      {c.feedback.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 14 }}>
          {c.feedback.map((f, fi) => (
            <FeedbackItem key={f.id} f={f} guide={guideFor(f, fi)} crit={c} rubricInfo={rubricInfo} citations={citations} accent={accent} noticeName={noticeName} />
          ))}
        </div>
      ) : (
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 13, color: '#16a37a', marginTop: 14, background: 'rgba(22,163,122,0.08)', padding: '10px 13px', borderRadius: 10, fontWeight: 600 }}>
          <CheckCircle2 size={15} /> 남은 지적 없음 — 이 항목은 깔끔합니다
        </div>
      )}

    </div>
  )
}

// 버전별 총점 라인 차트(SVG).
function ScoreTrendChart({ versions, selectedIndex, onSelect }) {
  const W = 640, H = 220, padX = 46, padTop = 50, padBottom = 44
  const innerW = W - padX * 2
  const innerH = H - padTop - padBottom
  const scores = versions.map((v) => v.total_score)
  const yMin = Math.max(0, Math.min(...scores) - 12)
  const yMax = Math.min(100, Math.max(...scores) + 10)
  const n = versions.length
  const xOf = (i) => (n === 1 ? W / 2 : padX + (innerW * i) / (n - 1))
  const yOf = (s) => padTop + innerH * (1 - (s - yMin) / (yMax - yMin || 1))
  const pts = versions.map((v, i) => ({ x: xOf(i), y: yOf(v.total_score), v, i }))
  const linePath = pts.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
  const baseY = H - padBottom
  const areaPath = `${linePath} L ${pts[pts.length - 1].x.toFixed(1)} ${baseY} L ${pts[0].x.toFixed(1)} ${baseY} Z`

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} key={n}>
      <defs>
        <linearGradient id="vtArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#7c5cea" stopOpacity="0.24" />
          <stop offset="100%" stopColor="#7c5cea" stopOpacity="0" />
        </linearGradient>
      </defs>
      <line x1={padX} y1={baseY} x2={W - padX} y2={baseY} stroke="rgba(28,26,46,0.08)" strokeWidth="1" />
      {n > 1 && <path className="vt-area" d={areaPath} fill="url(#vtArea)" />}
      {n > 1 && <path className="vt-line" pathLength="1" d={linePath} fill="none" stroke="#7c5cea" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />}
      {pts.slice(1).map((p, i) => {
        const prev = pts[i]
        // 부동소수점 잔여 오차 방지 — 소수 1자리 반올림 후 표시.
        const d = Math.round((p.v.total_score - prev.v.total_score) * 10) / 10
        const mx = (prev.x + p.x) / 2
        const my = (prev.y + p.y) / 2 - 13
        return (
          <text key={`d-${p.v.version}`} className="vt-dot-label mono" x={mx} y={my} textAnchor="middle" style={{ animationDelay: '0.9s' }} fontSize="12.5" fontWeight="800" fill={d >= 0 ? '#16a37a' : '#e0603d'}>
            {d >= 0 ? '+' : ''}{d}
          </text>
        )
      })}
      {pts.map((p, i) => {
        const active = i === selectedIndex
        return (
          <g key={p.v.version} className="vt-dotg" onClick={() => onSelect(i)}>
            <circle cx={p.x} cy={p.y} r="20" fill="transparent" />
            <circle className="vt-dot" cx={p.x} cy={p.y} r={active ? 8 : 5.5} fill={active ? '#7c5cea' : '#faf8f4'} stroke="#7c5cea" strokeWidth="3" style={{ animationDelay: `${0.4 + i * 0.12}s` }} />
            <text className="vt-dot-label mono" x={p.x} y={p.y - 17} textAnchor="middle" style={{ animationDelay: `${0.5 + i * 0.12}s` }} fontSize="14" fontWeight="800" fill="#1c1a2e">{p.v.total_score}</text>
            <text className="vt-dot-label mono" x={p.x} y={baseY + 21} textAnchor="middle" style={{ animationDelay: `${0.5 + i * 0.12}s` }} fontSize="11.5" fontWeight={active ? 800 : 600} fill={active ? '#7c5cea' : '#918d9f'}>{p.v.version}</text>
          </g>
        )
      })}
    </svg>
  )
}

// 프로필 표시 — "제출 정보 카드"는 MyPage(마이페이지)로 이동함(가은 요청, 2026-07-21).
// locked=false(단독 /version-test 데모): 비전공자/전공자 토글로 개인화 차이를 직접 확인.
// locked=true(/board 흐름 임베드): 실제 로그인 사용자는 프로필이 하나로 고정이므로 토글을
//   숨기고 "내 프로필 · OOO"만 읽기전용으로 보여준다(전공자가 비전공자 선택지를 보는 등의
//   혼동 방지 — 경이 요청, 2026-07-22). 변경은 마이페이지에서만.
function ProfileToggle({ profileKey, onChange, locked = false }) {
  if (locked) {
    const me = PROFILES[profileKey]
    return (
      <div className="card glass" style={{ marginBottom: 18, padding: '14px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span className="badge purple mono"><FlaskConical size={11} /> 프로필</span>
          <span style={{ fontSize: 12, color: '#918d9f' }}>프로필 기준으로 개발 위원 피드백의 구현 난이도 · 상세도가 개인별 맞춤 제공됩니다</span>
        </div>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 800, color: '#1c1a2e' }}>
            {me.label}
          </span>
          <span style={{ fontSize: 11, color: '#918d9f' }}>· 마이페이지에서 변경</span>
        </div>
      </div>
    )
  }
  return (
    <div className="card glass" style={{ marginBottom: 18, padding: '14px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <span className="badge amber mono"><FlaskConical size={11} /> TEST 프로필</span>
        <span style={{ fontSize: 12, color: '#918d9f' }}>제출자 프로필에 따라 개발 위원 피드백의 구현 난이도 · 상세도가 달라집니다</span>
      </div>
      <div style={{ display: 'inline-flex', background: 'var(--bg-2)', borderRadius: 10, padding: 3, gap: 3 }}>
        {['nonmajor', 'major'].map((k) => {
          const active = profileKey === k
          return (
            <button key={k} className="vt-tab" onClick={() => onChange(k)}
              style={{ padding: '6px 12px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 12.5, fontWeight: 700, background: active ? '#fff' : 'transparent', color: active ? '#1c1a2e' : '#918d9f', boxShadow: active ? '0 1px 4px rgba(28,26,46,0.1)' : 'none' }}>
              {PROFILES[k].label} 제출
            </button>
          )
        })}
      </div>
    </div>
  )
}

// --- 페이지 ----------------------------------------------------------------
// 개발 위원(구현 난이도 가이드가 붙는) persona — 백엔드 is_technical_persona와 동일.
const TECHNICAL_PERSONA_IDS = new Set(['technical_feasibility', 'dev_expert'])

// 실데이터 배선(B): 백엔드 GET /projects/{id}/report 응답을 이 화면의 버전 구조로 변환한다.
// 지금은 회의 1건 = v1.0 한 버전(수정본 재분석으로 v1.1+를 쌓는 건 C 단계). score_result.breakdown을
// 기준 항목으로 삼고, reviewer_results에서 criterion별 이름/판정/지적(issues·suggestions)을 채운다.
// 개발 위원(technical_feasibility)이 채점한 항목만 dev 탭, 나머지는 planning 탭.
// impl_guides(개인화 구현 가이드)는 여기서 안 붙이고, feedback_id(=criterion_id)로 렌더 시 매칭한다.
// 종합 위원(완성도·전달력)은 동적 rubric에서 전 항목을 겹쳐 채점하는 경향이 있어, "첫 채점자
// 우선"으로 담당을 정하면 개발 항목(예: 실현가능성)이 기획으로 잘못 분류된다. 그래서 한 항목을
// 여러 위원이 채점하면 우선순위로 실제 담당을 고른다: 기술 위원(개발) > 전문 위원(창의/사업) >
// 종합 위원(완성도). 이 우선순위가 곧 그 항목의 소속 탭(dev/planning)을 결정한다.
const GENERALIST_PERSONA_IDS = new Set(['presentation_completeness'])
function personaRank(pid) {
  if (TECHNICAL_PERSONA_IDS.has(pid)) return 2
  if (GENERALIST_PERSONA_IDS.has(pid)) return 0
  return 1
}

// 판정 표시 일관성(경이 2026-07-27): 위원 LLM의 원 판정은 상한(calibration) 적용 "전" 제안
// 기준이라, 상한으로 점수가 깎이면 "판정 적정인데 15점 중 6점" 같은 모순이 화면에 남고,
// 재분석마다 LLM 판정이 흔들리면 같은 점수인데 접속마다 판정이 달라 보인다. 그래서 화면
// 판정은 **최종(보정 후) 점수 비율**에서 결정론적으로 도출한다 — 경계는 reviewer_prompt의
// 점수 구간 기준과 동일(0~40% 중대 리스크 / 40~64% 보완 필요 / 65~84% 적정 / 85%~ 우수).
function judgmentFromScore(score, max) {
  if (!max || max <= 0) return 'acceptable'
  const r = score / max
  if (r >= 0.85) return 'strong'
  if (r >= 0.65) return 'acceptable'
  if (r >= 0.4) return 'needs_improvement'
  return 'critical_risk'
}

function reportToVersions(report) {
  const sr = report.score_result || {}
  const detail = new Map() // criterion_id -> {name, judgment, issues, suggestions, personaId}
  for (const r of report.reviewer_results || []) {
    for (const rs of r.rubric_scores || []) {
      const cur = detail.get(rs.criterion_id)
      if (!cur || personaRank(r.persona_id) > personaRank(cur.personaId)) {
        detail.set(rs.criterion_id, {
          name: rs.criterion_name,
          judgment: rs.judgment,
          issues: rs.issues || [],
          suggestions: rs.suggestions || [],
          issueRefs: rs.issue_refs || [], // 지적별 인용(원문 검증 verified 포함, issues와 정렬)
          personaId: r.persona_id,
        })
      }
    }
  }
  const criteria = (sr.breakdown || []).map((b) => {
    const d = detail.get(b.criterion_id) || {}
    const committee = TECHNICAL_PERSONA_IDS.has(d.personaId) ? 'dev' : 'planning'
    const issues = d.issues || []
    const suggestions = d.suggestions || []
    const issueRefs = d.issueRefs || []
    const feedback = []
    const n = Math.max(issues.length, suggestions.length)
    for (let i = 0; i < n; i++) {
      const issue = issues[i] || ''
      const sug = suggestions[i] || ''
      if (!issue && !sug) continue
      feedback.push({
        id: `${b.criterion_id}-${i}`,
        status: 'open', // 회의 1건(v1.0) 시점엔 모두 미해결. 해결/신규는 버전 비교(C)에서 계산.
        text: issue || sug,
        suggestion: issue ? sug : '',
        // 이 지적이 근거한 원문 문장(백엔드가 제출 문서 원문 존재를 검증한 것만 사용)
        ref: issue && issueRefs[i]?.verified ? issueRefs[i] : null,
      })
    }
    return {
      id: b.criterion_id,
      name: d.name || b.criterion_id,
      committee,
      score: b.raw_score ?? 0,
      max: b.max_score ?? CRITERION_MAX,
      calibration: b.calibration || null,
      judgment: judgmentFromScore(b.raw_score ?? 0, b.max_score ?? CRITERION_MAX),
      feedback,
    }
  })
  return [
    {
      version: 'v1.0',
      label: '현재 제출',
      submitted_at: report.created_at,
      total_score: sr.total_score ?? 0,
      max_total: sr.max_score ?? 100, // 측정 가능 항목 배점 합(주관 항목 제외 시 100 미만일 수 있음)
      criteria,
    },
  ]
}

// C-3: "다음 수정본 제출"로 회의가 쌓일 때마다 백엔드 GET /comparison 의 versions 배열에
// v1.0 → v1.1 → v1.2 … 가 하나씩 누적된다(build_version_history). 제출한 만큼 버전이 늘어나며
// 오래된 버전이 사라지거나 라벨이 밀리지 않는다. 각 버전을 이 화면의 버전 구조로 변환한다:
//   · 항목별 issues/suggestions를 짝지어 feedback으로 만들고,
//   · 직전 버전 대비 new_issues는 '신규', resolved_issues는 '해결'로 표시한다.
// impl_guides(개인화 구현 가이드)는 최신 버전에만 붙으므로 여기서 안 붙이고 criterion_id로
// 렌더 시 매칭한다(realGuides).

// 해결됨 근거 매칭(경이 요청 2026-07-26): "(p.N) '인용'" 꼬리를 뗀 뒤 문장부호·공백을 제거하고,
// 동일/포함 또는 2-gram 유사도로 같은 지적인지 판정한다 — 회의마다 LLM이 같은 지적을 조금씩
// 다르게 표현(어미·조사·인용 차이)해 resolved_issues에 변형이 여러 건 실려 오던 문제의 해법.
const stripIssueCitation = (s) => {
  const cut = (s || '').split(/\(\s*p\.?\s*\d+\s*\)/)[0].trim().replace(/[,·;]+$/, '')
  return cut.length >= 4 ? cut : (s || '').trim()
}
const _normIssueText = (s) => stripIssueCitation(s).replace(/[‘’“”'"()[\]{}<>.,·;:!?~\s-]/g, '')

// 지적문의 "핵심 주제 토큰"만 남긴다(경이 확인 2026-07-27) — 평가 지적에 공통으로 나오는
// 일반어(구체적/설명/부족/계획/근거 등)를 빼면 남는 단어가 지적의 실제 주제다
// (법·제도/예산/모델·알고리즘/데이터·품질/확산/성과지표 …). 재표현 잔존 매칭과
// STEP 1 지적-근거 정합 필터가 함께 쓴다.
const _ISSUE_TOPIC_STOP = new Set([
  '구체', '설명', '부족', '필요', '방안', '언급', '제시', '고려', '검토', '분석', '계획',
  '내용', '정보', '수준', '정도', '부분', '관련', '문제', '문서', '명확', '명시', '미흡',
  '보완', '근거', '항목', '대하', '대한', '위한', '없음', '있음', '전반', '여부', '해결',
])
const issueTopicTokens = (s) => {
  const out = new Set()
  // 벗긴 결과가 2자 미만이 되는 제거는 하지 않는다 — '제도'의 '도'까지 조사로 벗겨
  // 주제 토큰 자체가 사라지는 과잉 제거 방지.
  const strip = (w, re) => {
    const m = w.match(re)
    return m && w.length - m[0].length >= 2 ? w.slice(0, w.length - m[0].length) : w
  }
  for (let w of stripIssueCitation(s).split(/[^\p{L}\p{N}]+/u)) {
    // 접미가 겹쳐 붙은 형태는 변화가 없을 때까지 반복해서 벗긴다(경이 확인 2026-07-27) —
    // 1회만 벗기면 '구체적인'이 '구체적'에서 멈춰 불용어('구체')에 못 닿고, 무관한 지적과
    // 인용이 '구체적' 같은 일반어로 이어지는 오매칭이 생긴다(실측: '법적 제약·예산' 지적에
    // '수치 목표' 인용이 통과).
    for (let prev = ''; prev !== w; ) {
      prev = w
      w = strip(w, /(에서|으로|이나|이라|하다|되다|하여|되어|했다|됐다|하고)$/u)
      w = strip(w, /(은|는|이|가|을|를|에|의|도|와|과|만|로|들)$/u)
      w = strip(w, /(적|성|인|된|한|함|됨)$/u)
    }
    if (w.length >= 2 && !_ISSUE_TOPIC_STOP.has(w)) out.add(w)
  }
  return out
}
// 두 지적이 같은 주제인가 — 핵심 주제 토큰이 (작은 쪽 기준) 절반 이상 겹치면 같은 지적의
// 재표현으로 본다. 예: "법·제도적 제약 수준 설명 부족" ↔ "법적 제약·제도적 문제 고려 부족"
// = {제도,제약} 겹침 → 같음. "확산 계획 부족" ↔ "성과 지표(KPI) 계획 부족" = 겹침 0 → 다름.
function sameIssueTopic(a, b) {
  const A = issueTopicTokens(a)
  const B = issueTopicTokens(b)
  if (!A.size || !B.size) return false
  let inter = 0
  for (const t of A) if (B.has(t)) inter += 1
  return inter / Math.min(A.size, B.size) >= 0.5
}

function sameIssueText(a, b) {
  const na = _normIssueText(a)
  const nb = _normIssueText(b)
  if (!na || !nb) return false
  if (na === nb || na.includes(nb) || nb.includes(na)) return true
  const grams = (s) => { const g = new Set(); for (let i = 0; i < s.length - 1; i++) g.add(s.slice(i, i + 2)); return g }
  const A = grams(na)
  const B = grams(nb)
  if (A.size === 0 || B.size === 0) return false
  let inter = 0
  for (const g of A) if (B.has(g)) inter += 1
  return (2 * inter) / (A.size + B.size) >= 0.66
}

function buildVersionsFromHistory(versions) {
  return versions.map((v, vi) => ({
    version: v.version,
    label: v.label,
    submitted_at: v.submitted_at,
    total_score: v.total_score ?? 0,
    max_total: v.max_score ?? 100, // 측정 가능 항목 배점 합(주관 항목 제외 시 100 미만)
    // 버전별 AI 피드백 스냅샷(오탈자·맥락·분량밀도) — 그 버전 문서를 검사한 기록.
    // 없으면 null(스냅샷 도입 전 회의) → 렌더에서 정직하게 "기록 없음" 처리.
    ai_feedback: v.ai_feedback || null,
    criteria: (v.criteria || []).map((c) => {
      const newSet = new Set(c.new_issues || [])
      const issues = c.issues || []
      const suggestions = c.suggestions || []
      const issueRefs = c.issue_refs || []
      // 직전 버전의 같은 항목 지적 — "잔존(재표현)"과 "해결됨" 판정의 기준 원본
      const prevC = vi > 0 ? (versions[vi - 1].criteria || []).find((p) => p.criterion_id === c.criterion_id) : null
      const prevIssues = (prevC?.issues || []).filter(Boolean)
      const prevNewSet = new Set(prevC?.new_issues || [])
      // 재표현 잔존 매칭(경이 확인 2026-07-27): 재분석마다 LLM이 같은 지적을 조금 다른 문장으로
      // 다시 내면, 백엔드 집합 비교(문자열 일치)는 "이전 것 해결됨 + 사실상 같은 지적 신규"로
      // 잘못 갈라 모순이 보인다("법·제도적 제약 수준 설명 부족" 해결됨과 "법적 제약·제도적 문제
      // 고려 부족" 신규가 동시 표시). 핵심 주제 토큰이 겹치면 같은 지적의 재표현(잔존)으로 보고
      // '보완 필요'로 표시하며, 매칭된 직전 지적은 해결됨 후보에서 제외한다.
      const carriedPrevIdx = new Set()
      const feedback = []
      const n = Math.max(issues.length, suggestions.length)
      for (let i = 0; i < n; i++) {
        const issue = issues[i] || ''
        const sug = suggestions[i] || ''
        if (!issue && !sug) continue
        let status = 'open'
        if (issue && newSet.has(issue)) {
          const pi = prevIssues.findIndex((p, j) => !carriedPrevIdx.has(j) && (sameIssueText(issue, p) || sameIssueTopic(issue, p)))
          if (pi !== -1) carriedPrevIdx.add(pi)
          status = pi !== -1 ? 'open' : 'new'
        }
        feedback.push({
          id: `${c.criterion_id}-${i}`,
          status,
          text: issue || sug,
          suggestion: issue ? sug : '',
          // 이 지적이 근거한 원문 문장(백엔드 원문 검증 통과분만)
          ref: issue && issueRefs[i]?.verified ? issueRefs[i] : null,
        })
      }
      // 해결됨 표시 규칙(경이 요청 2026-07-26): ① 직전 버전의 같은 항목에 "실제로 있던 지적"과
      // 매칭되는 것만 보여주고(직전 버전 어디에도 없던 해결됨 = 근거 없는 생성이라 감춤),
      // ② 같은 지적의 표현 변형이 여러 건 오면 직전 지적 1건당 해결됨 1건으로 합치며,
      // ③ 문구는 재표현본이 아니라 직전 버전 지적 원문을 그대로 쓴다(취소선 + 해결됨 표시용).
      // ④ 이번 버전에 재표현으로 잔존한 지적(carriedPrevIdx)은 해결된 게 아니므로 제외한다.
      const resolvedPrevIdx = new Set()
      for (const t of c.resolved_issues || []) {
        const pi = prevIssues.findIndex((p, j) => !resolvedPrevIdx.has(j) && !carriedPrevIdx.has(j) && sameIssueText(t, p))
        if (pi !== -1) resolvedPrevIdx.add(pi)
      }
      for (const pi of [...resolvedPrevIdx].sort((a, b) => a - b)) {
        feedback.push({
          id: `${c.criterion_id}-resolved-${pi}`,
          status: 'resolved',
          text: stripIssueCitation(prevIssues[pi]),
          prevVersion: versions[vi - 1]?.version || '',
          prevStatus: prevNewSet.has(prevIssues[pi]) ? 'new' : 'open',
          note: '이번 수정본에서 반영되어 더 이상 지적되지 않습니다',
        })
      }
      return {
        id: c.criterion_id,
        name: c.criterion_name || c.criterion_id,
        committee: c.committee || 'planning',
        score: c.score ?? 0,
        max: c.max ?? CRITERION_MAX,
        calibration: c.calibration || null,
        judgment: judgmentFromScore(c.score ?? 0, c.max ?? CRITERION_MAX),
        feedback,
      }
    }),
  }))
}

// AI 피드백 탭(3번째) — 위원 채점과 별개로 자동 검사한 문서 품질(점수 미반영, 버전마다
// "수정 필요/해결" 추적). 3축을 본다(경이 요청, 2026-07-23 → 분량·밀도 추가 2026-07-23):
//   · 분량·밀도(getFormatCheck) — 공고문 요구 페이지 수 충족 + 페이지 채움률(빈 공간).
//     같은 내용이라도 여백이 많은 문서(A)와 꽉 채운 문서(B)를 가르는 축이라, 위원 채점(내용)이
//     비슷해도 여기서 B가 A보다 낫다는 게 드러난다.
//   · 오탈자(getTypoCheck) / 맥락 이상(getContextCheck) — 현재 버전 문서 기준 라이브 검사.
const AI_FEEDBACK = {
  name: 'AI 피드백', Icon: FileText, desc: '분량·밀도 · 오탈자 (점수 미반영)',
  color: '#16a37a', dim: 'rgba(22,163,122,0.12)',
  gradient: 'linear-gradient(135deg, #7fd8b8 0%, #16a37a 100%)',
}

// 분량·밀도 요약 카드 — A(빈 공간 많음)와 B(꽉 참)의 차이가 정확히 여기서 보인다.
function FormatSummary({ format }) {
  if (!format) return null
  const hasReq = format.required_min != null || format.required_max != null
  // 표기: "20p / 30p"(단일 기준), "20p / 10~30p"(범위), 기준을 못 찾으면 "20p / 기준 없음"
  // ("기준 기준 없음" 중복 표기 버그 수정, 경이 2026-07-26)
  const req = !hasReq ? '기준 없음'
    : format.required_min === format.required_max ? `${format.required_max}p`
    : `${format.required_min ?? ''}~${format.required_max ?? ''}p`
  const cov = format.overall_coverage != null ? Math.round(format.overall_coverage * 100) : null
  const pageOk = format.page_verdict == null || format.page_verdict === '충족'
  const densOk = format.overall_verdict == null || format.overall_verdict === '양호'

  const Metric = ({ label, ok, verdict, big, msg }) => (
    <div style={{ flex: 1, minWidth: 230, background: '#faf7f1', borderRadius: 12, padding: '14px 16px', border: `1px solid ${ok ? 'rgba(22,163,122,0.3)' : 'rgba(224,96,61,0.32)'}` }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: 12.5, fontWeight: 800, color: '#5b5770' }}>{label}</span>
        <span className="mono" style={{ fontSize: 11, fontWeight: 800, padding: '2px 10px', borderRadius: 99, color: ok ? '#16a37a' : '#e0603d', border: `1px solid ${ok ? '#16a37a' : '#e0603d'}` }}>
          {ok ? <>✓ {verdict}</> : <>! {verdict}</>}
        </span>
      </div>
      {big && <div className="mono" style={{ fontSize: 20, fontWeight: 800, color: '#1c1a2e', marginBottom: 4 }}>{big}</div>}
      {msg && <div style={{ fontSize: 12, color: '#5b5770', lineHeight: 1.6 }}>{msg}</div>}
    </div>
  )

  return (
    <div className="vt-fade card glass" style={{ padding: '16px 20px', marginBottom: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 14, fontWeight: 800, color: '#16a37a' }}>분량 · 밀도</span>
        <span style={{ fontSize: 11.5, color: '#918d9f' }}>공고문 기준 분량 충족과 페이지 채움 정도 — 같은 내용이라도 여백이 많으면 여기서 드러납니다</span>
      </div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <Metric label="분량 (페이지 수)" ok={pageOk} verdict={format.page_verdict || '기준 없음'}
          big={`${format.actual_pages ?? '?'}p / ${req}`} msg={format.page_message} />
        <Metric label="밀도 (채움률)" ok={densOk} verdict={format.overall_verdict || '기준 없음'}
          big={cov != null ? `${cov}%` : '—'} msg={format.density_message} />
      </div>
    </div>
  )
}

// AI 피드백 안내 탭 — '중요한 정보'(기본 접힘): 채움률만 보고 글자로만 채우는 오해를 막는
// 실무 팁(경이 요청 2026-07-26). 클릭하면 펼쳐진다.
function ImportantInfoTab() {
  const [open, setOpen] = useState(false)
  return (
    <div className="card glass" style={{ marginBottom: 14, padding: '4px 20px' }}>
      <button type="button" onClick={() => setOpen((v) => !v)}
        style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 8, background: 'none', border: 'none', cursor: 'pointer', padding: '13px 0', textAlign: 'left' }}>
        <AlertTriangle size={15} style={{ color: '#b8830b', flexShrink: 0 }} />
        <span style={{ fontSize: 13.5, fontWeight: 800, color: '#1c1a2e', flex: 1 }}>중요한 정보</span>
        <ChevronDown size={15} style={{ flexShrink: 0, color: '#918d9f', transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.18s ease' }} />
      </button>
      {open && (
        <div className="vt-fade" style={{ padding: '0 0 13px', fontSize: 13, lineHeight: 1.7, color: '#1c1a2e' }}>
          글씨만 가득 채우기보다 <b>대시보드(도표, 시각화 자료 등)</b> 관련 자료들도 넣어야 <b>서류 심사에 통과합니다.</b>
        </div>
      )}
    </div>
  )
}

function AiFeedbackPanel({ findings, format, missingVersion }) {
  const list = findings || []
  // 스냅샷 없는 옛 버전 — 최신 문서 검사 결과를 대신 보여주면 그 버전의 기록인 것처럼
  // 오해되므로(오귀속), 기록이 없다는 사실을 그대로 알린다.
  if (missingVersion) {
    return (
      <div className="card glass" style={{ padding: '22px 24px', textAlign: 'center', color: '#918d9f', fontSize: 13, lineHeight: 1.7 }}>
        <b style={{ color: '#5b5770' }}>{missingVersion} 제출 시점의 AI 피드백 기록이 없습니다.</b><br />
        버전별 검사 기록(분량·밀도·오탈자·맥락) 저장은 이후 제출되는 수정본부터 적용됩니다 —
        다른 버전의 검사 결과를 이 버전의 것처럼 보여주지 않습니다.
      </div>
    )
  }
  return (
    <>
      {/* '중요한 정보'는 패널 맨 위(AI 피드백 탭 버튼 바로 아래)에 — 경이 위치 지정 2026-07-27 */}
      <ImportantInfoTab />
      <FormatSummary format={format} />
      <div className="card glass" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', borderLeft: '4px solid #16a37a', marginBottom: 16, padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ width: 42, height: 42, borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(22,163,122,0.12)', color: '#16a37a' }}>
            <FileText size={21} />
          </span>
          <div>
            <div style={{ fontSize: 15.5, fontWeight: 700, color: '#16a37a' }}>오탈자 · 맥락</div>
            <div style={{ fontSize: 12, color: '#918d9f', marginTop: 2 }}>점수 미반영 — 버전마다 수정/해결 추적</div>
          </div>
        </div>
        {list.length > 0
          ? <span className="badge amber mono">! 수정 필요 {list.length}</span>
          : <span className="mono" style={{ fontSize: 11.5, color: '#16a37a', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 5 }}><CheckCircle2 size={14} /> 이슈 없음</span>}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {list.map((f, i) => (
          <div key={f.id || i} className="vt-fade card glass" style={{ padding: 16, animationDelay: `${i * 70}ms` }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span className="mono" style={{ fontSize: 11, fontWeight: 800, padding: '3px 10px', borderRadius: 99, background: 'rgba(184,131,11,0.14)', color: '#b8830b' }}>
                {f.kind === 'typo' ? '오탈자' : '문자서식·맥락'}
              </span>
              <span className="mono" style={{ fontSize: 11, color: '#e0603d', fontWeight: 700 }}>수정 필요</span>
            </div>
            {f.corrected ? (
              <div style={{ fontSize: 13.5, marginBottom: 6 }}>
                <span style={{ color: '#e0603d', textDecoration: 'line-through' }}>{f.quote}</span>
                <span style={{ margin: '0 8px', color: '#918d9f' }}>→</span>
                <span style={{ color: '#16a37a', fontWeight: 700 }}>{f.corrected}</span>
              </div>
            ) : (
              f.quote && <div style={{ fontSize: 13, color: '#3a3750', marginBottom: 6, lineHeight: 1.6 }}>“…{f.quote}…”</div>
            )}
            {f.message && <div style={{ fontSize: 12.5, color: '#5b5770', lineHeight: 1.6 }}>{f.message}</div>}
          </div>
        ))}
      </div>
      <p style={{ fontSize: 11.5, color: '#a8a4b2', lineHeight: 1.7, marginTop: 18 }}>
        ※ AI 피드백(오탈자·문자서식)은 위원 채점과 별개로 점수에 반영되지 않으며, 현재 버전 문서를 자동 검사한 결과입니다.
      </p>
    </>
  )
}

// 경이/Claude(2026-07-25): 점수 체계표 — 공고문에서 동적 추출한 rubric(하드코딩 없음, 다른
// 공모전 배점표가 와도 그대로 반영)을 기준으로 ① 채점 항목(배점·채점 기준) ② 측정 불가로
// 제외된 항목(사유) ③ 가점 요소(항상 제외)를 한 표로 보여준다. "왜 이렇게 채점할 수밖에
// 없었는지"를 리포트 안에서 설명하는 역할.
function ScoringSchemeCard({ rubric, open, onToggle }) {
  if (!rubric) return null
  const criteria = rubric.criteria || []
  const excluded = rubric.excluded_criteria || []
  const bonusMax = rubric.bonus_max_score || 0
  // 정직한 출처 표기(2026-07-25 사고 재발 방지): 백엔드가 "공고문에서 실제 추출"이라고 명시한
  // 경우(true)에만 그렇게 말한다. 정적 템플릿 폴백/출처 불명이면 빨간 경고로 사실대로 알린다 —
  // 폴백을 "공고문 배점표에서 자동 추출"로 보여주는 것은 사용자를 속이는 것.
  const extracted = rubric.extracted_from_notice === true
  const cell = { padding: '9px 12px', fontSize: 12.5, lineHeight: 1.6, verticalAlign: 'top', borderTop: '1px solid rgba(28,26,46,0.07)' }
  const tag = (bg, color, text) => (
    <span className="mono" style={{ fontSize: 10.5, fontWeight: 800, padding: '2px 9px', borderRadius: 99, background: bg, color, whiteSpace: 'nowrap' }}>{text}</span>
  )
  return (
    <div className="card glass" style={{ marginBottom: 18, padding: '4px 20px 4px', position: 'relative', border: extracted ? undefined : '1.5px solid rgba(224,96,61,0.45)' }}>
      {/* 접힘/펼침 헤더 — 클릭하면 표가 열린다(기본 접힘, 경이 요청 2026-07-25) */}
      <button type="button" onClick={onToggle}
        style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', background: 'none', border: 'none', cursor: 'pointer', padding: '14px 0', textAlign: 'left' }}>
        <span style={{ fontSize: 14.5, fontWeight: 800, color: '#1c1a2e' }}>점수 체계표</span>
        {extracted ? (
          <span style={{ fontSize: 11.5, color: '#918d9f', flex: 1 }}>측정 가능 항목만 채점</span>
        ) : (
          <span style={{ fontSize: 11.5, color: '#e0603d', fontWeight: 700, flex: 1 }}>⚠️ 공고문 기준이 아님 — 기본 템플릿으로 채점됨 (만점 {rubric.total_max_score}점)</span>
        )}
        <ChevronDown size={16} style={{ color: '#918d9f', transition: 'transform 0.2s', transform: open ? 'rotate(180deg)' : 'none', flexShrink: 0 }} />
      </button>
      {!extracted && (
        <div style={{ margin: '0 0 12px', padding: '10px 14px', borderRadius: 10, background: 'rgba(224,96,61,0.08)', border: '1px solid rgba(224,96,61,0.3)', fontSize: 12.5, lineHeight: 1.65, color: '#8a4a30' }}>
          공고문에서 <b>평가기준·배점을 추출하지 못해</b> 서비스 기본 템플릿으로 채점되었습니다. 이 표와 총점은 <b>공고문 기준이 아니므로 참고하지 마세요.</b>{' '}
          평가기준·배점표가 담긴 <b>공고문 원문(파일 또는 공지 URL)</b>을 공고 자료로 올리고 재분석하면 공고문 기준으로 다시 채점됩니다.
        </div>
      )}
      {open && (
      <div className="vt-fade" style={{ overflowX: 'auto', paddingBottom: 14 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
          <thead>
            <tr>
              {['평가 항목', '배점', '채점 여부', '채점 기준 · 사유'].map((h) => (
                <th key={h} style={{ ...cell, borderTop: 'none', fontSize: 11, fontWeight: 800, color: '#918d9f', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {criteria.map((c) => (
              <tr key={c.criterion_id}>
                <td style={{ ...cell, fontWeight: 700, whiteSpace: 'nowrap' }}>{c.criterion_name}</td>
                <td className="mono" style={{ ...cell, whiteSpace: 'nowrap' }}>{c.max_score}점</td>
                <td style={cell}>{tag('rgba(22,163,122,0.12)', '#16a37a', '채점')}</td>
                <td style={{ ...cell, color: '#5b5770' }}>{c.description || '공고문 배점표 기준으로 문서 근거를 들어 채점합니다.'}</td>
              </tr>
            ))}
            {excluded.map((c) => (
              <tr key={c.criterion_id}>
                <td style={{ ...cell, fontWeight: 700, whiteSpace: 'nowrap', color: '#918d9f' }}>{c.criterion_name}</td>
                <td className="mono" style={{ ...cell, whiteSpace: 'nowrap', color: '#918d9f' }}>{c.max_score}점</td>
                <td style={cell}>{tag('rgba(224,96,61,0.1)', '#e0603d', '제외')}</td>
                <td style={{ ...cell, color: '#918d9f' }}>{c.reason}</td>
              </tr>
            ))}
            {bonusMax > 0 && (
              <tr>
                <td style={{ ...cell, fontWeight: 700, whiteSpace: 'nowrap', color: '#918d9f' }}>가점 요소</td>
                <td className="mono" style={{ ...cell, whiteSpace: 'nowrap', color: '#918d9f' }}>최대 {bonusMax}점</td>
                <td style={cell}>{tag('rgba(224,96,61,0.1)', '#e0603d', '제외')}</td>
                <td style={{ ...cell, color: '#918d9f' }}>가점 기준은 공모전마다 변동이 크고 별도 증빙 확인이 필요해 자동 채점에서 제외합니다.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      )}
    </div>
  )
}

// embedded: true면 /board 플로우("종합 리포트" 단계) 안에 끼워 넣는 모드 — 상단 나가기/
// 실험 배지 바를 숨긴다(사이드바가 이미 단계 이동을 제공하므로). 기본(false)은 /version-test
// 단독 페이지로 동작. projectId가 오면(embedded) 그 프로젝트의 실제 /report를 렌더한다.
export default function VersionTrackerTestPage({ embedded = false, projectId = null }) {
  const navigate = useNavigate()
  const [revealed, setRevealed] = useState(1)
  const [selectedIndex, setSelectedIndex] = useState(0)
  // 탭 순서·기본값: AI 피드백 → 기획 위원 → 개발 위원 (경이 요청 2026-07-25)
  const [committee, setCommittee] = useState('ai_feedback')
  // 상세(탭 영역)는 처음에 숨기고, 점수 추이의 버전 점(v1.0…)을 클릭해야 열린다.
  const [detailOpen, setDetailOpen] = useState(false)
  const [schemeOpen, setSchemeOpen] = useState(false) // 점수 체계표 접기/펼치기
  const [noticeOpen, setNoticeOpen] = useState(false) // 총점 참고용 안내 — 총점 옆 ⓘ 클릭 팝업
  const [statusFilter, setStatusFilter] = useState('all') // 전체/신규/보완필요(남음)/해결 필터
  const detailRef = useRef(null)
  const [profileKey, setProfileKey] = useState('nonmajor')
  const profile = PROFILES[profileKey]
  const [report, setReport] = useState(null)      // 실제 /report (embedded)
  const [versionPayload, setVersionPayload] = useState(null) // /comparison 응답(versions 히스토리)
  const [aiFindings, setAiFindings] = useState([])    // AI 피드백(오탈자·맥락) — 점수 미반영
  const [formatCheck, setFormatCheck] = useState(null) // 분량·밀도(빈 공간) 요약 — A vs B 변별 축
  const [projectDocs, setProjectDocs] = useState([])  // 문서 역할 메타(제출/공고문/보조) — 인용 분류용
  const [reportLoaded, setReportLoaded] = useState(false)
  const [submitting, setSubmitting] = useState(false) // 수정본 업로드+재분석 중
  const [submitStage, setSubmitStage] = useState('')  // 진행 상태 문구
  const [submitError, setSubmitError] = useState('')
  const fileInputRef = useRef(null)

  // /board 흐름 임베드 모드: 실제 로그인 사용자의 프로필로 고정한다(토글 대신). 백엔드
  // classify_impl_difficulty의 핵심 신호인 education.is_technical_major로 2단계(전공/비전공)를
  // 가른다 — 전공자면 major(쉬움/간결), 아니면 nonmajor(어려움/자세히). 프로필 미제출/조회
  // 실패면 안전 폴백으로 nonmajor(어려움) 유지(백엔드 폴백과 동일 방향).
  useEffect(() => {
    if (!embedded) return
    let cancelled = false
    getMyProfile()
      .then((p) => { if (!cancelled) setProfileKey(p?.education?.is_technical_major ? 'major' : 'nonmajor') })
      .catch(() => { /* 미제출/비로그인 → nonmajor 폴백 유지 */ })
    return () => { cancelled = true }
  }, [embedded])

  // 임베드 모드: 실제 회의 결과(/report)와 버전 비교(/comparison)를 함께 불러온다. 재분석 후
  // 다시 부르기 위해 함수로 분리. 실패 시 mock 유지(단독 데모와 동일).
  const loadReportAndComparison = useCallback(async () => {
    if (!embedded || !projectId) return
    const [r, c, docs] = await Promise.all([
      getProjectReport(projectId).catch(() => null),
      getProjectComparison(projectId).catch(() => null),
      getDocuments(projectId).catch(() => []), // 인용 출처를 제출문서/공고문/보조자료로 분류하기 위한 역할 메타
    ])
    setReport(r)
    setVersionPayload(c) // {versions:[v1.0,v1.1,...], comparison, available, meeting_count}
    setProjectDocs(docs || [])
  }, [embedded, projectId])

  // AI 피드백(오탈자·맥락)은 LLM 호출이라 느릴 수 있어 메인 리포트 로딩과 분리해 비동기로 받는다.
  const loadAiFindings = useCallback(async () => {
    if (!embedded || !projectId) return
    const [typos, ctx, fmt] = await Promise.all([
      getTypoCheck(projectId).catch(() => []),
      getContextCheck(projectId).catch(() => []),
      getFormatCheck(projectId).catch(() => null),
    ])
    setAiFindings([
      ...(typos || []).map((f) => ({ ...f, kind: 'typo' })),
      ...(ctx || []).map((f) => ({ ...f, kind: 'context' })),
    ])
    setFormatCheck(fmt) // 분량(페이지 수)·밀도(채움률·빈 페이지) — A(빈 공간 많음) vs B(꽉 참) 차이가 여기서 드러난다
  }, [embedded, projectId])

  useEffect(() => {
    if (!embedded || !projectId) { setReportLoaded(true); return }
    let cancelled = false
    ;(async () => {
      await loadReportAndComparison()
      if (!cancelled) setReportLoaded(true)
    })()
    loadAiFindings() // 비동기 별도 로딩(리포트 표시를 막지 않음)
    return () => { cancelled = true }
  }, [embedded, projectId, loadReportAndComparison, loadAiFindings])

  // 버전 히스토리(/comparison.versions)가 있으면 v1.0, v1.1, v1.2 … 전체를 그리고, 없으면
  // (조회 실패 등) /report 단일 버전으로 폴백, 그것도 없으면 mock. realGuides는 최신 회의
  // impl_guides(criterion_id 키)로 개발 위원 항목 개인화 가이드를 렌더 시 매칭한다.
  const realVersions = useMemo(() => {
    const hist = versionPayload?.versions
    if (Array.isArray(hist) && hist.length) return buildVersionsFromHistory(hist)
    if (report) return reportToVersions(report)
    return null
  }, [versionPayload, report])
  const usingReal = Boolean(realVersions)
  const ALL = usingReal ? realVersions : ALL_VERSIONS
  // 총점 참고용 안내(ⓘ)는 공고문에서 실제 추출한 rubric으로 채점했을 때만 의미가 있다
  const noticeAvailable = usingReal && report?.rubric?.extracted_from_notice === true
  const realGuides = useMemo(() => {
    if (!usingReal) return null
    const m = new Map()
    for (const g of report.impl_guides || []) m.set(g.feedback_id, g)
    return m
  }, [usingReal, report])

  // 문서 역할 메타 — 파일명 → 역할(submission/notice/support). 중심 자료(notice)의 기준은
  // **파일명에 '공고문'이 포함된 파일**(경이 확정, 2026-07-25). 없으면 rubric이 실제 추출된
  // source 문서 → 첫 criteria 문서 순으로 폴백. 그 외 공고 자료는 전부 보조 자료(support) —
  // 보조 자료는 중심 자료(공고문 기준)를 보완하는 근거로만 쓰인다.
  const docRoles = useMemo(() => {
    const byName = new Map()
    const noticeIds = new Set(report?.rubric?.source_document_ids || [])
    const criteriaDocs = projectDocs.filter((d) => (d.document_role || 'target') === 'criteria')
    const nameOf = (d) => d.original_filename || d.source_url || ''
    const noticeDoc =
      criteriaDocs.find((d) => nameOf(d).includes('공고문')) ||
      criteriaDocs.find((d) => noticeIds.has(d.id)) ||
      criteriaDocs[0] ||
      null
    const noticeName = noticeDoc ? nameOf(noticeDoc) : null
    for (const d of projectDocs) {
      const name = nameOf(d)
      if (!name) continue
      const role = d.document_role || 'target'
      if (role === 'target') byName.set(name, 'submission')
      else if (role === 'criteria') byName.set(name, d === noticeDoc ? 'notice' : 'support')
      else byName.set(name, 'support')
    }
    return { byName, noticeName }
  }, [projectDocs, report])

  // 근거 인용(criterion_id → RAG evidence quote, 역할 분류 포함) — 위원 rubric_scores.evidence_ids를
  // report.evidence(quote·page·문서명)와 조인. 인용은 색인된 실제 문서 원문에서만 온다(지어내기 없음).
  const realCitations = useMemo(() => {
    if (!usingReal || !report) return null
    const evById = new Map((report.evidence || []).map((e) => [e.evidence_id, e]))
    const m = new Map()
    const seenByCid = new Map() // criterion_id → Set(중복 인용 제거용 정규화 quote)
    for (const r of report.reviewer_results || []) {
      for (const rs of r.rubric_scores || []) {
        const arr = m.get(rs.criterion_id) || []
        const seen = seenByCid.get(rs.criterion_id) || new Set()
        for (const id of rs.evidence_ids || []) {
          const e = evById.get(id)
          if (!e) continue
          // 원문 검증(경이 요청 2026-07-26): 파일명·페이지가 붙는 인용은 색인된 청크 원문(text)에
          // 실제로 존재하는 문장만 허용한다. 위원 LLM이 적은 quote가 청크 원문에 없으면(지어낸
          // 문장을 실제 파일·페이지에 붙이는 할루시네이션) quote 대신 청크 원문을 그대로 보여주고,
          // 청크를 못 찾은 인용(chunk_id 미상)은 파일·페이지를 보증할 수 없어 표시하지 않는다.
          if (!e.chunk_id || e.chunk_id === 'unknown') continue
          const chunkText = (e.text || '').trim().replace(/\s+/g, ' ')
          if (!chunkText) continue
          const claimed = (e.quote || '').trim().replace(/\s+/g, ' ')
          const normEq = (s) => s.replace(/[\s‘’“”'"()[\]·,.\-—–…]/g, '')
          const quote = claimed && normEq(chunkText).includes(normEq(claimed)) ? claimed : chunkText
          const key = quote.replace(/\s+/g, '')
          if (seen.has(key)) continue // 위원 여러 명이 같은 청크를 인용하면 한 번만 표시
          seen.add(key)
          const source = e.document_name || ''
          arr.push({
            quote: quote.length > 280 ? quote.slice(0, 280) + '…' : quote,
            page: e.page ?? null,
            source,
            section: e.section || null, // 원문 섹션 제목(가독성 헤더용)
            score: typeof e.score === 'number' ? e.score : null, // KURE 의미 유사도
            role: docRoles.byName.get(source) || 'support',
          })
        }
        if (arr.length) {
          // 단어 매칭 순서가 아니라 **의미 유사도(KURE score) 높은 순**으로 상위 4개만 표시.
          arr.sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
          m.set(rs.criterion_id, arr.slice(0, 4))
          seenByCid.set(rs.criterion_id, seen)
        }
      }
    }
    // 보조 자료 배타 배정(경이 확정 2026-07-27): 위원이 다른 항목 채점에 인용했더라도, 보조
    // 청크(섹션 제목+본문)는 "의미가 가장 잘 맞는 평가 항목"에서만 보여준다 — 신청 서식의
    // 'AI 모델·알고리즘 설계 적정성' 섹션이 실현 가능성 밑에 뜨는 어긋남 방지. 매칭은 항목명+
    // 설명의 2-gram이 청크 텍스트에 얼마나 나타나는지(결정론적 커버리지)로 판정하고, 어느
    // 항목과도 뚜렷이 안 맞으면(신호 부족) 그대로 둔다. 제외 항목(안전성·윤리성)도 배정 후보에
    // 넣어, 그런 섹션이 채점 항목에 잘못 붙는 것까지 막는다.
    const normGram = (s) => (s || '').replace(/[^\p{L}\p{N}]/gu, '')
    const gramSet = (s) => {
      const t = normGram(s)
      const g = new Set()
      for (let i = 0; i < t.length - 1; i++) g.add(t.slice(i, i + 2))
      return g
    }
    const pool = [
      ...(report.rubric?.criteria || []),
      ...(report.rubric?.excluded_criteria || []),
    ]
      .map((c) => ({ id: c.criterion_id, grams: gramSet(`${c.criterion_name || ''} ${c.description || c.reason || ''}`) }))
      .filter((c) => c.grams.size > 0)
    if (pool.length) {
      const coverage = (text, grams) => {
        let hit = 0
        for (const g of grams) if (text.includes(g)) hit += 1
        return hit / grams.size
      }
      for (const [cid, arr] of m) {
        m.set(cid, arr.filter((q) => {
          if (q.role !== 'support') return true
          const text = normGram(`${q.section || ''} ${q.quote}`)
          let best = 0
          let bestIds = []
          for (const c of pool) {
            const cov = coverage(text, c.grams)
            if (cov > best + 1e-9) { best = cov; bestIds = [c.id] }
            else if (cov > 0 && Math.abs(cov - best) <= 1e-9) bestIds.push(c.id)
          }
          if (best < 0.1) return true // 어느 항목과도 뚜렷이 안 맞으면 배정 판단 보류
          return bestIds.includes(cid)
        }))
      }
    }
    return m
  }, [usingReal, report, docRoles])

  // 공고문 배점표 메타(criterion_id → {description, max_score}) — "왜 이 점수인가" 단계 1(공고
  // 기준)에서 인용한다. 동적 rubric이라 공모전이 바뀌어도 그대로 반영된다.
  const rubricInfoById = useMemo(() => {
    const m = new Map()
    for (const c of report?.rubric?.criteria || []) m.set(c.criterion_id, c)
    return m
  }, [report])

  // 버전 점 클릭으로 상세가 열리면 상세 영역으로 부드럽게 스크롤한다.
  useEffect(() => {
    if (detailOpen) detailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [detailOpen, selectedIndex])

  // 실데이터 버전 수가 바뀌면(1→2) 모두 펼치고 최신 버전을 선택한다(mock의 단계 공개와 분리).
  useEffect(() => {
    if (realVersions) {
      setRevealed(realVersions.length)
      setSelectedIndex(realVersions.length - 1)
    }
  }, [realVersions])

  const versions = ALL.slice(0, revealed)
  const selected = versions[selectedIndex] || ALL[0]
  const prev = selectedIndex > 0 ? versions[selectedIndex - 1] : null
  const totalDelta = prev ? selected.total_score - prev.total_score : null
  const nextVersion = revealed < ALL.length ? ALL[revealed].version : null
  const heroScore = useCountUp(selected.total_score)

  const cm = COMMITTEES[committee] || AI_FEEDBACK // ai_feedback 탭은 점수 영역을 안 그리지만 참조 안전용
  // 우선순위 정렬 — "무엇부터 고쳐야 하나"가 위에 오도록: 판정 심각도 → 깎인 점수 비율 순.
  const SEVERITY = { critical_risk: 3, needs_improvement: 2, insufficient_evidence: 1.5, acceptable: 1, strong: 0 }
  const cmItems = selected.criteria
    .filter((c) => c.committee === committee)
    .slice()
    .sort((a, b) => {
      const sa = SEVERITY[a.judgment] ?? 1, sb = SEVERITY[b.judgment] ?? 1
      if (sb !== sa) return sb - sa
      const la = (a.max - a.score) / (a.max || 1), lb = (b.max - b.score) / (b.max || 1)
      return lb - la
    })
  // 상태 필터(전체/신규/남음/해결) — 지적 단위로 거르고, 걸러진 뒤 지적이 없는 항목은 숨긴다.
  const visibleItems = statusFilter === 'all'
    ? cmItems
    : cmItems
        // allFeedback: 필터 전 원본 — 점수 변화 팝업의 해결/신규/잔여 건수가 필터에 흔들리지 않게
        .map((c) => ({ ...c, allFeedback: c.feedback, feedback: c.feedback.filter((f) => f.status === statusFilter) }))
        .filter((c) => c.feedback.length > 0)
  const cmScore = committeeScore(selected, committee)
  const cmBefore = prev ? committeeScore(prev, committee) : null
  const cmMax = cmItems.reduce((s, ci) => s + (ci.max ?? CRITERION_MAX), 0)
  const cmDelta = cmBefore == null ? null : cmScore - cmBefore
  const counts = cmItems.reduce((a, c) => { c.feedback.forEach((f) => { a[f.status] += 1 }); return a }, { open: 0, new: 0, resolved: 0 })

  function handleNext() {
    if (usingReal) {
      if (!submitting) fileInputRef.current?.click() // 실모드: 진짜 수정본 파일 선택
      return
    }
    if (!nextVersion) return
    setRevealed((r) => { setSelectedIndex(r); return r + 1 })
  }

  // C-2: 실제 수정본 업로드 → 재분석 → 재조회. analyze가 target 첫 문서를 쓰므로 기존 target을
  // 지워 항상 "최신 수정본 1개"만 남긴다(이전 버전 데이터는 meeting 스냅샷에 보존돼 비교 가능).
  async function handleRevisionFile(e) {
    const file = e.target.files?.[0]
    if (e.target) e.target.value = ''
    if (!file || !projectId) return
    setSubmitting(true); setSubmitError(''); setSubmitStage('수정본 업로드 중...')
    try {
      const docs = await getDocuments(projectId).catch(() => [])
      for (const d of docs) {
        if ((d.document_role || 'target') === 'target') await deleteDocument(projectId, d.id).catch(() => {})
      }
      // source_type은 백엔드가 파일로 추론하므로 기존 업로드와 동일하게 'pdf' 고정으로 넘긴다.
      const uploaded = await uploadDocument(projectId, file, 'pdf', 'target')
      setSubmitStage('문서 색인 중...')
      const docId = uploaded?.id || uploaded?.document_id
      for (let i = 0; i < 40 && docId; i++) {
        const st = await getDocumentStatus(projectId, docId).catch(() => null)
        const s = st?.status
        if (s === 'indexed' || s === 'indexed_empty') break
        if (s === 'indexing_failed' || s === 'conversion_failed' || s === 'indexing_timeout') {
          throw new Error('업로드한 수정본을 색인하지 못했습니다.')
        }
        await new Promise((r) => setTimeout(r, 1500))
      }
      setSubmitStage('AI 위원 재검토 중...')
      const token = window.crypto?.randomUUID?.() || String(Date.now())
      let polling = true
      ;(async () => {
        while (polling) {
          const p = await getAnalyzeProgress(projectId, token).catch(() => null)
          if (p) {
            const done = p.reviews_done || 0, total = p.reviews_total || 0
            setSubmitStage(total ? `AI 위원 재검토 중... (${done}/${total})` : 'AI 위원 재검토 중...')
          }
          await new Promise((r) => setTimeout(r, 1500))
        }
      })()
      await analyzeProject(projectId, undefined, token)
      polling = false
      setSubmitStage('결과 정리 중...')
      await loadReportAndComparison()
      loadAiFindings() // 새 버전 문서의 오탈자·서식 재검사(비동기)
    } catch (err) {
      setSubmitError(err?.message || '수정본 분석에 실패했습니다.')
    } finally {
      setSubmitting(false); setSubmitStage('')
    }
  }

  const animKey = `${selected.version}-${committee}-${profileKey}`

  // 임베드 모드에서 실제 /report를 아직 불러오는 중이면 mock을 깜빡 보여주지 않고 로딩 표시.
  if (embedded && projectId && !reportLoaded) {
    return (
      <div className="vt-root">
        <div style={{ maxWidth: 920, margin: '0 auto', padding: '48px 24px', textAlign: 'center' }}>
          <span className="badge purple mono"><FlaskConical size={12} /> 종합 리포트</span>
          <h1 style={{ fontSize: 22, fontWeight: 700, marginTop: 14, color: '#1c1a2e' }}>회의 결과를 불러오는 중...</h1>
        </div>
      </div>
    )
  }

  return (
    <div className="vt-root">
      <div style={{ maxWidth: 920, margin: '0 auto', padding: '28px 24px 64px' }}>
        {/* 상단 — 단독 페이지일 때만. 플로우 임베드 시엔 사이드바가 이동을 담당. */}
        {!embedded && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <button className="btn-ghost" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }} onClick={() => navigate('/board')}>
              <ArrowLeft size={15} /> 나가기
            </button>
            <span className="badge purple mono"><FlaskConical size={12} /> User RAG · 실험 화면</span>
          </div>
        )}

        {/* ===== 개요 화면(점수 추이) — 버전 상세가 열리면 통째로 숨긴다(별도 화면 전환, 경이 요청 2026-07-25) ===== */}
        {!detailOpen && (<>
        {/* 히어로 */}
        <div className="card glass" style={{ padding: '26px 28px', marginBottom: 18 }}>
          <h1 style={{ fontSize: 32, fontWeight: 800, lineHeight: 1.3, textAlign: 'center', margin: '2px 0 18px' }}>종합 리포트</h1>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 24, flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: 300 }}>
              <div style={{ maxWidth: 500, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 9, fontSize: 13, lineHeight: 1.6, color: '#1c1a2e' }}>
                  <span style={{ flexShrink: 0, fontSize: 11, fontWeight: 800, color: '#16a37a', background: 'rgba(22,163,122,0.1)', padding: '3px 10px', borderRadius: 99, whiteSpace: 'nowrap' }}>AI 피드백</span>
                  <span><b>오탈자 · 분량 · 밀도 · 맥락</b>을 점검합니다.</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 9, fontSize: 13, lineHeight: 1.6, color: '#1c1a2e' }}>
                  <span style={{ flexShrink: 0, fontSize: 11, fontWeight: 800, color: '#7c5cea', background: 'rgba(124,92,234,0.1)', padding: '3px 10px', borderRadius: 99, whiteSpace: 'nowrap' }}>기획 · 개발 위원</span>
                  <span>공고문 <b>평가기준 · 배점</b>을 근거로 채점합니다.</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginTop: 3, padding: '10px 13px', background: '#fbf3ec', border: '1px solid rgba(224,96,61,0.28)', borderRadius: 10 }}>
                  <AlertTriangle size={15} style={{ color: '#e0603d', flexShrink: 0, marginTop: 1 }} />
                  <span style={{ fontSize: 12.5, lineHeight: 1.55, color: '#1c1a2e' }}>
                    <b>총점이 높아도</b> AI 피드백을 반영하지 않으면 <b>서류 심사를 통과하지 못합니다.</b>
                  </span>
                </div>
              </div>
            </div>
            <div style={{ textAlign: 'center', minWidth: 140, padding: '4px 8px', position: 'relative' }}>
              <div className="mono" style={{ fontSize: 11, color: '#918d9f', marginBottom: 6, fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                {selected.version} 총점
                {noticeAvailable && (
                  <span role="button" tabIndex={0} aria-label="총점 참고용 안내"
                    onClick={() => setNoticeOpen((v) => !v)}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setNoticeOpen((v) => !v) } }}
                    style={{ display: 'inline-flex', alignItems: 'center', cursor: 'pointer', padding: 2, borderRadius: 6 }}>
                    <Info size={14} style={{ color: '#b8830b', flexShrink: 0 }} />
                  </span>
                )}
              </div>
              <div className="mono" style={{ fontSize: 46, fontWeight: 800, lineHeight: 1, marginBottom: 10, color: '#1c1a2e' }}>
                {heroScore}<span style={{ fontSize: 15, color: '#918d9f' }}>/{selected.max_total ?? 100}</span>
              </div>
              {totalDelta != null ? <DeltaPill value={totalDelta} size="lg" /> : <span className="badge amber mono">출발점</span>}
              {/* ⓘ 클릭 팝업 — 총점 참고용 안내 (점수 체계표 헤더에서 총점 옆으로 이동) */}
              {noticeAvailable && noticeOpen && (
                <div className="vt-fade" style={{ position: 'absolute', top: 26, right: 0, zIndex: 60, width: 'min(430px, 78vw)', textAlign: 'left', background: '#fff', border: '1px solid rgba(184,131,11,0.35)', borderLeft: '4px solid #b8830b', borderRadius: 12, boxShadow: '0 14px 34px rgba(28,26,46,0.16)', padding: '12px 16px', display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <Info size={16} style={{ color: '#b8830b', flexShrink: 0, marginTop: 2 }} />
                  <div style={{ fontSize: 12.5, lineHeight: 1.7, color: '#5b5770' }}>
                    <b style={{ color: '#8a6508' }}>제시된 총점은 참고용입니다.</b>{' '}
                    공고문 평가 항목 중 <b>문서 내용으로 측정 가능한 항목만</b> 근거를 들어 채점하며,
                    정성 판단이 필요한 <b>주관적 항목</b>과 공모전마다 기준이 달라지는 <b>가점 요소</b>는
                    총점에서 제외됩니다. 항목별 채점·제외 사유는 아래 <b>점수 체계표</b>에서 확인할 수
                    있으며, 실제 심사 결과와는 다를 수 있습니다.
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 점수 체계표(접힘 탭 — 클릭해서 펼침, 총점 참고용 안내는 히어로 총점 옆 ⓘ 팝업) — 실데이터 모드에서만 */}
        {usingReal && report?.rubric && (
          <ScoringSchemeCard rubric={report.rubric} open={schemeOpen} onToggle={() => setSchemeOpen((v) => !v)} />
        )}

        {/* TEST 프로필 토글 — 제출 정보 카드는 MyPage로 이동함 */}
        <ProfileToggle profileKey={profileKey} onChange={setProfileKey} locked={embedded} />

        {/* 점수 추이 그래프 — 상세(위원 탭)는 여기서 버전 점을 클릭해야 아래에 열린다 */}
        <div className="card glass" style={{ padding: '20px 22px 10px', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', marginBottom: 4 }}>
            <div>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 7 }}>
                <TrendingUp size={17} color="#7c5cea" /> 버전별 점수 추이
              </h2>
              <span style={{ fontSize: 12, color: '#918d9f' }}>버전 점(v1.0 …)을 클릭하면 그 버전의 상세 리포트 화면으로 이동합니다.</span>
            </div>
          </div>
          {/* 업로드+재분석 진행/에러 배너 */}
          {submitting && (
            <div className="card glass" style={{ margin: '4px 0 8px', padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: '#5b5770' }}>
              <FlaskConical size={14} className="vt-spin" style={{ color: '#7c5cea' }} /> {submitStage || '수정본 분석 중...'}
            </div>
          )}
          {submitError && (
            <div className="card glass" style={{ margin: '4px 0 8px', padding: '10px 14px', fontSize: 13, color: '#e0603d', borderLeft: '3px solid #e0603d' }}>
              {submitError}
            </div>
          )}
          <ScoreTrendChart versions={versions} selectedIndex={selectedIndex}
            onSelect={(i) => { setSelectedIndex(i); setDetailOpen(true); setStatusFilter('all') }} />
        </div>

        {/* 다음 수정본 제출 — 점수 추이 카드 "바깥 아래"(경이 요청 2026-07-25) */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 22 }}>
          <button className="btn-primary" onClick={handleNext} disabled={usingReal ? submitting : !nextVersion} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            {usingReal
              ? (submitting ? <>분석 중...</> : <><Plus size={14} /> 다음 수정본 제출</>)
              : nextVersion
                ? <><Plus size={14} /> 다음 수정본 제출 ({nextVersion})</>
                : <><CheckCircle2 size={14} /> 모든 버전 반영됨</>}
          </button>
          {/* 실제 수정본 파일 입력(숨김) — 버튼이 이걸 트리거한다(C-2) */}
          <input ref={fileInputRef} type="file" accept=".pdf,.docx,.pptx,.hwp,.hwpx" style={{ display: 'none' }} onChange={handleRevisionFile} />
        </div>

        </>)}

        {/* ===== 버전 상세 화면 — 개요를 대체하는 별도 화면. 뒤로가기/버전 칩으로 이동 ===== */}
        {detailOpen && (
        <div ref={detailRef} key={`detail-${selected.version}`} className="vt-fade">
        {/* 상세 헤더 — 뒤로가기 + 버전 전환 칩 + 이 버전 총점 */}
        <div className="card glass" style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', margin: '0 0 14px', padding: '12px 16px' }}>
          <button className="btn-ghost" onClick={() => setDetailOpen(false)} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 13, fontWeight: 700 }}>
            <ArrowLeft size={15} /> 점수 추적 그래프
          </button>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {versions.map((v, i) => {
              const active = i === selectedIndex
              return (
                <button key={v.version} type="button" className="mono vt-tab" onClick={() => { setSelectedIndex(i); setStatusFilter('all') }}
                  style={{ padding: '6px 13px', borderRadius: 99, border: `1.5px solid ${active ? '#7c5cea' : 'rgba(28,26,46,0.12)'}`, background: active ? '#7c5cea' : 'rgba(255,255,255,0.8)', color: active ? '#fff' : '#5b5770', fontSize: 12, fontWeight: 800, cursor: 'pointer' }}>
                  {v.version}
                </button>
              )
            })}
          </div>
          <div className="mono" style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 12, color: '#918d9f', fontWeight: 700 }}>{selected.label || '상세 리포트'}</span>
            <span style={{ fontSize: 19, fontWeight: 800 }}>{selected.total_score}<span style={{ fontSize: 12, color: '#918d9f' }}>/{selected.max_total ?? 100}</span></span>
            {totalDelta != null && <DeltaPill value={totalDelta} />}
          </div>
        </div>
        {/* 탭 순서: AI 피드백 → 기획 위원 → 개발 위원(경이 요청 2026-07-25) */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
          {['ai_feedback', 'planning', 'dev'].map((cid) => {
            const t = cid === 'ai_feedback' ? AI_FEEDBACK : COMMITTEES[cid]
            const active = committee === cid
            const Icon = t.Icon
            return (
              <button key={cid} className="vt-tab" onClick={() => { setCommittee(cid); setStatusFilter('all') }}
                style={{ flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8, padding: '12px 16px', borderRadius: 12, border: `1.5px solid ${active ? 'transparent' : 'rgba(28,26,46,0.1)'}`, background: active ? t.gradient : 'rgba(255,255,255,0.72)', color: active ? '#fff' : '#5b5770', fontSize: 14, fontWeight: 700, cursor: 'pointer', boxShadow: active ? `0 10px 22px ${t.dim}` : 'none' }}>
                <Icon size={16} /> {t.name}
              </button>
            )
          })}
        </div>

        {/* AI 피드백 탭: 점수 영역 대신 오탈자·맥락·분량밀도 — **선택한 버전의** 검사 기록을
            보여준다(경이 요청 2026-07-26). 그 버전 회의에 저장된 스냅샷 우선, 최신 버전은
            라이브 검사로 폴백, 스냅샷 없는 옛 버전은 최신 데이터를 대신 보여주지 않고(오귀속)
            정직하게 기록 없음을 알린다. */}
        {committee === 'ai_feedback' && (() => {
          const snap = selected.ai_feedback
          if (snap) {
            const snapFindings = [
              ...((snap.typo?.findings) || []).map((f) => ({ ...f, kind: 'typo' })),
              ...((snap.context?.findings) || []).map((f) => ({ ...f, kind: 'context' })),
            ]
            return <AiFeedbackPanel findings={snapFindings} format={snap.format || null} />
          }
          if (selectedIndex === ALL.length - 1) {
            return <AiFeedbackPanel findings={aiFindings} format={formatCheck} />
          }
          return <AiFeedbackPanel missingVersion={selected.version} />
        })()}

        {committee !== 'ai_feedback' && (<>
        {/* 위원 소계 요약 */}
        <div className="card glass" key={`sum-${animKey}`} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', borderLeft: `4px solid ${cm.color}`, marginBottom: 16, padding: '16px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ width: 42, height: 42, borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', background: cm.dim, color: cm.color }}>
              <cm.Icon size={21} />
            </span>
            <div>
              <div style={{ fontSize: 15.5, fontWeight: 700, color: cm.color }}>{cm.name}</div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <span className="mono" style={{ fontSize: 10.5, color: '#918d9f', fontWeight: 600 }}>점수</span>
              <span className="mono" style={{ fontSize: 21, fontWeight: 800 }}>
                {cmBefore != null && <span style={{ fontSize: 14, color: '#918d9f', fontWeight: 700 }}>{cmBefore} → </span>}
                <CountUp value={cmScore} style={{ color: cm.color }} /><span style={{ fontSize: 12, color: '#918d9f' }}> / {cmMax}</span>
              </span>
            </div>
            {cmDelta != null && <DeltaPill value={cmDelta} />}
            {/* 상태 필터 탭 — 색상 없이 회색 배경 · 검정 글씨로 통일(경이 요청 2026-07-26) */}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {[
                { k: 'all', label: `전체` },
                { k: 'new', label: `+ 신규 ${counts.new}` },
                { k: 'open', label: `! 보완 필요 ${counts.open}` },
                { k: 'resolved', label: `✓ 해결 ${counts.resolved}` },
              ].map((f) => (
                <button key={f.k} type="button" className="badge mono" onClick={() => setStatusFilter(f.k)}
                  style={{ cursor: 'pointer', background: 'rgba(28,26,46,0.07)', color: '#1c1a2e', border: statusFilter === f.k ? '1.5px solid #1c1a2e' : '1.5px solid transparent', opacity: statusFilter === f.k ? 1 : 0.62 }}>
                  {f.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* 위원 항목 카드 — 우선순위(심각도·감점 비율) 순 정렬 + 상태 필터 적용 */}
        <div key={`body-${animKey}-${statusFilter}`} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {visibleItems.length === 0 && (
            <div className="card glass" style={{ padding: 18, textAlign: 'center', color: '#918d9f', fontSize: 13 }}>
              이 필터에 해당하는 {statusFilter === 'resolved' ? '해결' : '지적'}이 없습니다.
            </div>
          )}
          {visibleItems.map((c, i) => {
            // 인용 전달 규칙(경이 요청 2026-07-26): 중심·보조 자료(공고문·붙임)는 프로젝트 공통이라
            // 모든 버전의 STEP 2에 동일하게 보여준다. 제출 문서 인용(submission)은 버전마다 파일이
            // 달라(evidence는 최신 회의 기준) 최신 버전에서만 보여준다 — 옛 버전에 최신 문서 인용을
            // 붙이면 그것도 오귀속이므로.
            const isLatest = selectedIndex === ALL.length - 1
            const cited = realCitations ? realCitations.get(c.id) || [] : null
            const citations = cited == null ? null : (isLatest ? cited : cited.filter((q) => q.role !== 'submission'))
            return (
              <CriterionCard key={c.id} c={c} before={criterionBefore(versions, selectedIndex, c.id)} index={i} priority={i + 1} animKey={`${animKey}-${statusFilter}`} isDev={committee === 'dev'} profile={profile} accent={cm} rubricInfo={rubricInfoById.get(c.id)} noticeName={docRoles.noticeName} realGuides={isLatest ? realGuides : null} citations={citations} />
            )
          })}
        </div>

        </>)}
        </div>
        )}
      </div>
    </div>
  )
}

