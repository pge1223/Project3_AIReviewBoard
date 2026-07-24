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
  AlertTriangle, Zap, ChevronDown, FileText,
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

const JUDGMENT_LABEL = {
  strong: { text: '우수', color: '#16a37a', bg: 'rgba(22,163,122,0.12)' },
  acceptable: { text: '적정', color: '#7c5cea', bg: 'rgba(124,92,234,0.12)' },
  needs_improvement: { text: '보완 필요', color: '#b8830b', bg: 'rgba(184,131,11,0.14)' },
  critical_risk: { text: '중대 리스크', color: '#e0603d', bg: 'rgba(224,96,61,0.12)' },
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
  return version.criteria.filter((c) => c.committee === committee).reduce((s, c) => s + c.score, 0)
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
  const up = value > 0
  const flat = value === 0
  const color = flat ? '#918d9f' : up ? '#16a37a' : '#e0603d'
  const bg = flat ? 'rgba(145,141,159,0.12)' : up ? 'rgba(22,163,122,0.12)' : 'rgba(224,96,61,0.12)'
  const Icon = up ? TrendingUp : TrendingDown
  const pad = size === 'lg' ? '6px 14px' : '3px 9px'
  const fs = size === 'lg' ? 15 : 12
  return (
    <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 5, borderRadius: 99, fontWeight: 700, whiteSpace: 'nowrap', color, background: bg, padding: pad, fontSize: fs }}>
      {!flat && <Icon size={size === 'lg' ? 15 : 12} />}
      {flat ? '±0' : `${up ? '+' : ''}${value}점`}
    </span>
  )
}

// accentColor/accentBg: 'acceptable'(적정)은 기본색이 보라라, 개발 위원 섹션에선 위원
// accent(분홍)로 맞춘다(기획 위원은 accent가 보라라 그대로). 나머지 판정색은 의미축이라 유지.
function JudgmentChange({ before, after, accentColor, accentBg }) {
  const a = JUDGMENT_LABEL[after]
  const b = JUDGMENT_LABEL[before]
  if (!a) return null
  const changed = before && before !== after
  const col = (key, base) => (key === 'acceptable' && accentColor ? { color: accentColor, background: accentBg } : { color: base.color, background: base.bg })
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      {changed && b && (
        <>
          <span style={{ ...chip, ...col(before, b), opacity: 0.5 }}>{b.text}</span>
          <span style={{ color: '#c7c3d0', fontSize: 12 }}>→</span>
        </>
      )}
      <span style={{ ...chip, ...col(after, a) }}>{a.text}</span>
    </span>
  )
}
const chip = { fontSize: 11.5, fontWeight: 700, padding: '3px 10px', borderRadius: 8 }

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
          <span className="mono" style={{ fontWeight: 800, width: 24, textAlign: 'right', flexShrink: 0, color: r.strong ? accent.num : '#918d9f', fontSize: r.strong ? 17 : 14 }}>{r.value}</span>
          <span className="mono" style={{ fontSize: 11, color: '#b6b1c2', width: 24, flexShrink: 0 }}>/{max}</span>
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

function FeedbackItem({ f, guide }) {
  const [open, setOpen] = useState(false)
  const s = STATUS_META[f.status]
  const Icon = s.Icon
  // guide: { feedback_id, level, verbosity, label, prose } — 백엔드 attach_impl_guides 출력 형태.
  // verbosity==='detailed'(비전공/입문)면 '자세히 보기'로 접고, 그 외(standard/brief)면 인라인.
  const diff = guide ? DIFFICULTY[guide.level] : null
  const detailed = guide?.verbosity === 'detailed'
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '11px 13px', borderRadius: 11, background: s.bg, border: `1px solid ${s.border}` }}>
      <span style={{ flexShrink: 0, width: 20, height: 20, borderRadius: '50%', background: s.color, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: 1 }}>
        <Icon size={12} strokeWidth={3} />
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 7, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 11.5, fontWeight: 800, flexShrink: 0, color: s.color }}>{s.label}</span>
          <span style={{ fontSize: 13.5, lineHeight: 1.5, fontWeight: 600, textDecoration: f.status === 'resolved' ? 'line-through' : 'none', color: f.status === 'resolved' ? '#a8a4b2' : '#3a3750' }}>{f.text}</span>
        </div>
        {f.suggestion && (
          <div style={{ display: 'flex', gap: 7, marginTop: 8, fontSize: 13, lineHeight: 1.6, color: '#5b5770', background: 'rgba(255,255,255,0.6)', padding: '8px 11px', borderRadius: 9 }}>
            <Lightbulb size={15} color="#b8830b" style={{ flexShrink: 0, marginTop: 1 }} />
            <span><b style={{ color: '#3a3750' }}>이렇게 고치세요:</b> {f.suggestion}</span>
          </div>
        )}

        {/* 개인화: 개발 위원 구현 난이도 (프로필 기반) */}
        {guide && diff && (
          <div style={{ marginTop: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11.5, fontWeight: 700, padding: '4px 10px', borderRadius: 99, color: diff.color, background: diff.bg }}>
                <diff.Icon size={12} /> {guide.label}
              </span>
              {detailed && (
                <button className="vt-tab" onClick={() => setOpen((v) => !v)}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 12, fontWeight: 700, color: '#1c1a2e', background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px 4px' }}>
                  자세히 보기 <ChevronDown size={13} style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.18s ease' }} />
                </button>
              )}
            </div>
            {detailed ? (
              open && <GuideSteps prose={guide.prose} diff={diff} />
            ) : (
              <div style={{ marginTop: 7, fontSize: 12.5, lineHeight: 1.6, color: '#5b5770' }}>{guide.prose}</div>
            )}
          </div>
        )}

        {f.note && (
          <div style={{ display: 'flex', gap: 7, marginTop: 8, fontSize: 13, lineHeight: 1.6, color: '#12876a' }}>
            <CheckCircle2 size={15} style={{ flexShrink: 0, marginTop: 1 }} />
            <span><b>반영됨:</b> {f.note}</span>
          </div>
        )}
      </div>
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

function CriterionCard({ c, before, index, animKey, isDev, profile, accent, realGuides }) {
  const delta = before == null ? null : c.score - before
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
    <div className="vt-fade card glass" style={{ animationDelay: `${index * 90}ms` }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 15.5, fontWeight: 700 }}>{c.name}</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          {delta != null && <DeltaPill value={delta} />}
          <JudgmentChange before={null} after={c.judgment} accentColor={accent?.color} accentBg={accent?.dim} />
        </div>
      </div>

      <CompareBars before={before} after={c.score} max={c.max ?? CRITERION_MAX} animKey={animKey} accent={accent?.bar} />

      {c.calibration && (
        <div style={{ marginTop: 12, padding: '10px 12px', borderRadius: 10, background: 'rgba(224,96,61,0.08)', border: '1px solid rgba(224,96,61,0.2)', color: '#7a442f', fontSize: 12.5, lineHeight: 1.55 }}>
          <b>근거 기반 점수 상한 적용:</b>{' '}
          위원 제안 {c.calibration.original_score}점 → 상한 {c.calibration.cap_score}점
          {(c.calibration.signals || []).length > 0 && (
            <span> · {(c.calibration.signals || []).map((s) => s.reason).join(' · ')}</span>
          )}
        </div>
      )}

      {c.feedback.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 14 }}>
          {c.feedback.map((f, fi) => <FeedbackItem key={f.id} f={f} guide={guideFor(f, fi)} />)}
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
        const d = p.v.total_score - prev.v.total_score
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
          <span className="badge purple mono"><FlaskConical size={11} /> 내 프로필</span>
          <span style={{ fontSize: 12, color: '#918d9f' }}>내 프로필 기준으로 개발 위원 피드백의 구현 난이도 · 상세도가 맞춤 제공됩니다</span>
        </div>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 800, color: me.difficulty === 'easy' ? '#16a37a' : '#e0603d' }}>
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
      })
    }
    return {
      id: b.criterion_id,
      name: d.name || b.criterion_id,
      committee,
      score: b.raw_score ?? 0,
      max: b.max_score ?? CRITERION_MAX,
      calibration: b.calibration || null,
      judgment: d.judgment || 'acceptable',
      feedback,
    }
  })
  return [
    {
      version: 'v1.0',
      label: '현재 제출',
      submitted_at: report.created_at,
      total_score: sr.total_score ?? 0,
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
function buildVersionsFromHistory(versions) {
  return versions.map((v) => ({
    version: v.version,
    label: v.label,
    submitted_at: v.submitted_at,
    total_score: v.total_score ?? 0,
    criteria: (v.criteria || []).map((c) => {
      const newSet = new Set(c.new_issues || [])
      const issues = c.issues || []
      const suggestions = c.suggestions || []
      const feedback = []
      const n = Math.max(issues.length, suggestions.length)
      for (let i = 0; i < n; i++) {
        const issue = issues[i] || ''
        const sug = suggestions[i] || ''
        if (!issue && !sug) continue
        feedback.push({
          id: `${c.criterion_id}-${i}`,
          status: issue && newSet.has(issue) ? 'new' : 'open',
          text: issue || sug,
          suggestion: issue ? sug : '',
        })
      }
      for (const t of c.resolved_issues || []) {
        feedback.push({ id: `${c.criterion_id}-resolved-${feedback.length}`, status: 'resolved', text: t, note: '이번 수정본에서 반영되어 더 이상 지적되지 않습니다' })
      }
      return {
        id: c.criterion_id,
        name: c.criterion_name || c.criterion_id,
        committee: c.committee || 'planning',
        score: c.score ?? 0,
        max: c.max ?? CRITERION_MAX,
        calibration: c.calibration || null,
        judgment: c.judgment || 'acceptable',
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
  const req = !hasReq ? '기준 없음'
    : format.required_min === format.required_max ? `${format.required_min}p`
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
          big={`${format.actual_pages ?? '?'}p / 기준 ${req}`} msg={format.page_message} />
        <Metric label="밀도 (채움률)" ok={densOk} verdict={format.overall_verdict || '기준 없음'}
          big={cov != null ? `${cov}%` : '—'} msg={format.density_message} />
      </div>
    </div>
  )
}

function AiFeedbackPanel({ findings, format }) {
  const list = findings || []
  return (
    <>
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

// embedded: true면 /board 플로우("완성 리포트" 단계) 안에 끼워 넣는 모드 — 상단 나가기/
// 실험 배지 바를 숨긴다(사이드바가 이미 단계 이동을 제공하므로). 기본(false)은 /version-test
// 단독 페이지로 동작. projectId가 오면(embedded) 그 프로젝트의 실제 /report를 렌더한다.
export default function VersionTrackerTestPage({ embedded = false, projectId = null }) {
  const navigate = useNavigate()
  const [revealed, setRevealed] = useState(1)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [committee, setCommittee] = useState('planning')
  const [profileKey, setProfileKey] = useState('nonmajor')
  const profile = PROFILES[profileKey]
  const [report, setReport] = useState(null)      // 실제 /report (embedded)
  const [versionPayload, setVersionPayload] = useState(null) // /comparison 응답(versions 히스토리)
  const [aiFindings, setAiFindings] = useState([])    // AI 피드백(오탈자·맥락) — 점수 미반영
  const [formatCheck, setFormatCheck] = useState(null) // 분량·밀도(빈 공간) 요약 — A vs B 변별 축
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
    const [r, c] = await Promise.all([
      getProjectReport(projectId).catch(() => null),
      getProjectComparison(projectId).catch(() => null),
    ])
    setReport(r)
    setVersionPayload(c) // {versions:[v1.0,v1.1,...], comparison, available, meeting_count}
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
  const realGuides = useMemo(() => {
    if (!usingReal) return null
    const m = new Map()
    for (const g of report.impl_guides || []) m.set(g.feedback_id, g)
    return m
  }, [usingReal, report])

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
  const cmItems = selected.criteria.filter((c) => c.committee === committee)
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
          <span className="badge purple mono"><FlaskConical size={12} /> 완성 리포트</span>
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

        {/* 히어로 */}
        <div className="card glass" style={{ padding: '26px 28px', marginBottom: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 24, flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: 300 }}>
              <span className="badge purple mono" style={{ marginBottom: 12 }}>버전 추적형 USER RAG</span>
              <h1 style={{ fontSize: 27, fontWeight: 700, lineHeight: 1.3, margin: '10px 0 10px' }}>IT 공모전 · 개인 맞춤형 피드백 루프</h1>
              <div style={{ maxWidth: 500, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 9, fontSize: 13, lineHeight: 1.6, color: '#5b5770' }}>
                  <span style={{ flexShrink: 0, fontSize: 11, fontWeight: 800, color: '#7c5cea', background: 'rgba(124,92,234,0.1)', padding: '3px 10px', borderRadius: 99, whiteSpace: 'nowrap' }}>기획 · 개발 위원</span>
                  <span>공고문 <b>평가기준 · 배점</b>을 근거로 채점합니다.</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 9, fontSize: 13, lineHeight: 1.6, color: '#5b5770' }}>
                  <span style={{ flexShrink: 0, fontSize: 11, fontWeight: 800, color: '#16a37a', background: 'rgba(22,163,122,0.1)', padding: '3px 10px', borderRadius: 99, whiteSpace: 'nowrap' }}>AI 피드백</span>
                  <span><b>오탈자 · 분량 · 밀도 · 맥락</b>을 점검합니다.</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginTop: 3, padding: '10px 13px', background: '#fbf3ec', border: '1px solid rgba(224,96,61,0.28)', borderRadius: 10 }}>
                  <AlertTriangle size={15} style={{ color: '#e0603d', flexShrink: 0, marginTop: 1 }} />
                  <span style={{ fontSize: 12.5, lineHeight: 1.55, color: '#8a4a30' }}>
                    위원 <b>총점이 높아도</b> AI 피드백을 반영하지 않으면 <b>서류 심사를 통과하지 못합니다.</b>
                  </span>
                </div>
              </div>
            </div>
            <div style={{ textAlign: 'center', minWidth: 140, padding: '4px 8px' }}>
              <div className="mono" style={{ fontSize: 11, color: '#918d9f', marginBottom: 6, fontWeight: 600 }}>{selected.version} 총점</div>
              <div className="mono" style={{ fontSize: 46, fontWeight: 800, lineHeight: 1, marginBottom: 10, color: '#1c1a2e' }}>
                {heroScore}<span style={{ fontSize: 15, color: '#918d9f' }}>/100</span>
              </div>
              {totalDelta != null ? <DeltaPill value={totalDelta} size="lg" /> : <span className="badge amber mono">출발점</span>}
            </div>
          </div>
        </div>

        {/* TEST 프로필 토글 — 제출 정보 카드는 MyPage로 이동함 */}
        <ProfileToggle profileKey={profileKey} onChange={setProfileKey} locked={embedded} />

        {/* 점수 추이 그래프 */}
        <div className="card glass" style={{ padding: '20px 22px 10px', marginBottom: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', marginBottom: 4 }}>
            <div>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 7 }}>
                <TrendingUp size={17} color="#7c5cea" /> 버전별 점수 추이
              </h2>
              <span style={{ fontSize: 12, color: '#918d9f' }}>점을 클릭하면 해당 버전의 피드백을 볼 수 있어요</span>
            </div>
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
          <ScoreTrendChart versions={versions} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
        </div>

        {/* 위원 탭 + AI 피드백 탭(점수 없는 문자서식·오탈자) */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
          {['planning', 'dev', 'ai_feedback'].map((cid) => {
            const t = cid === 'ai_feedback' ? AI_FEEDBACK : COMMITTEES[cid]
            const active = committee === cid
            const Icon = t.Icon
            return (
              <button key={cid} className="vt-tab" onClick={() => setCommittee(cid)}
                style={{ flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8, padding: '12px 16px', borderRadius: 12, border: `1.5px solid ${active ? 'transparent' : 'rgba(28,26,46,0.1)'}`, background: active ? t.gradient : 'rgba(255,255,255,0.72)', color: active ? '#fff' : '#5b5770', fontSize: 14, fontWeight: 700, cursor: 'pointer', boxShadow: active ? `0 10px 22px ${t.dim}` : 'none' }}>
                <Icon size={16} /> {t.name}
              </button>
            )
          })}
        </div>

        {/* AI 피드백 탭: 점수 영역 대신 오탈자·문자서식 검사 결과 */}
        {committee === 'ai_feedback' && <AiFeedbackPanel findings={aiFindings} format={formatCheck} />}

        {committee !== 'ai_feedback' && (<>
        {/* 위원 소계 요약 */}
        <div className="card glass" key={`sum-${animKey}`} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', borderLeft: `4px solid ${cm.color}`, marginBottom: 16, padding: '16px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ width: 42, height: 42, borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', background: cm.dim, color: cm.color }}>
              <cm.Icon size={21} />
            </span>
            <div>
              <div style={{ fontSize: 15.5, fontWeight: 700, color: cm.color }}>{cm.name}</div>
              <div style={{ fontSize: 12, color: '#918d9f', marginTop: 2 }}>{cm.desc}</div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <span className="mono" style={{ fontSize: 10.5, color: '#918d9f', fontWeight: 600 }}>이 위원 점수</span>
              <span className="mono" style={{ fontSize: 21, fontWeight: 800 }}>
                {cmBefore != null && <span style={{ fontSize: 14, color: '#918d9f', fontWeight: 700 }}>{cmBefore} → </span>}
                <CountUp value={cmScore} style={{ color: cm.color }} /><span style={{ fontSize: 12, color: '#918d9f' }}> / {cmMax}</span>
              </span>
            </div>
            {cmDelta != null && <DeltaPill value={cmDelta} />}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <span className="badge green mono">✓ 해결 {counts.resolved}</span>
              <span className="badge amber mono">! 남음 {counts.open}</span>
              <span className="badge coral mono">+ 신규 {counts.new}</span>
            </div>
          </div>
        </div>

        {/* 위원 항목 카드 */}
        <div key={`body-${animKey}`} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {cmItems.map((c, i) => (
            <CriterionCard key={c.id} c={c} before={criterionBefore(versions, selectedIndex, c.id)} index={i} animKey={animKey} isDev={committee === 'dev'} profile={profile} accent={cm} realGuides={selectedIndex === ALL.length - 1 ? realGuides : null} />
          ))}
        </div>

        {usingReal ? (
          <p style={{ fontSize: 11.5, color: '#a8a4b2', lineHeight: 1.7, marginTop: 26 }}>
            ※ 이 리포트는 실제 회의 결과(<code style={codeStyle}>GET /projects/{'{id}'}/report</code>)의 점수·위원 피드백·개인화
            구현 가이드를 그대로 렌더한 것입니다.{' '}
            {versions.length > 1
              ? <>버전 간 비교(해결/잔존/신규·점수 증감)는 <code style={codeStyle}>GET /projects/{'{id}'}/comparison</code>(build_version_history) 결과이며, 수정본을 낼 때마다 v1.0 → v1.1 → v1.2 … 로 누적됩니다.</>
              : <>수정본을 제출해 재분석하면 v1.1, v1.2 … 로 버전이 쌓이며 이전/현재가 자동 비교됩니다.</>}
          </p>
        ) : (
          <p style={{ fontSize: 11.5, color: '#a8a4b2', lineHeight: 1.7, marginTop: 26 }}>
            ※ 각 버전의 위원별 피드백은 <code style={codeStyle}>review_output.reviewer_results</code>,
            버전 간 점수 증감·해결/잔존/신규는 <code style={codeStyle}>build_revision_comparison()</code>{' '}
            출력 구조를 그대로 mock으로 넣은 것입니다(단독 데모). 백엔드 연동 시 mock만 실데이터로 교체하면 됩니다.
          </p>
        )}
        </>)}
      </div>
    </div>
  )
}

const codeStyle = { fontFamily: "'JetBrains Mono', monospace", background: 'rgba(124,92,234,0.1)', padding: '1px 5px', borderRadius: 4, fontSize: 11, color: '#7c5cea' }
