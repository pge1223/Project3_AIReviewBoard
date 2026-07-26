import { ClipboardList } from 'lucide-react'
import { FEASIBILITY_LABEL } from './ideationConversationHelpers'

// 작성자: 가은/Claude(2026-07-22)
// 목적: /board "주제 아이디어 회의" 오른쪽 패널의 '아이디어 기획 캔버스'.
//       문제 상황 / 타깃 사용자 / 핵심 해결 방식 / 차별점 / 구현 가능성·리스크 /
//       심사기준 대응 포인트 6개 항목을 회의가 진행되는 동안 자동으로 채워 보여준다.
//
// 채움 방식(요청: "LLM이 자동으로 판단해서 채우기"): 프런트에서 새 LLM 호출을 하지 않는다.
//       ① 후보 선택 시점 — 백엔드 그래프의 LLM이 이미 구조화해 주는 selected_idea
//       (problem/target_user/solution/core_value/differentiation/feasibility/risks/
//       contest_fit)를 매핑한다. ② 매 라운드 후 — canvas_update 노드(2026-07-22 추가,
//       경이 협의 완료, ai/meeting/graph/ideation_conv_nodes.py::make_canvas_update_node)가
//       이번 라운드 위원 발언·사용자 답변을 반영해 갱신한 idea_canvas(selected_idea와 같은
//       키)를 우선 사용한다. ③ 심사기준 대응 포인트의 심사기준 목록은 공모전 분석
//       (getAnnouncementAnalysis → official_facts.evaluation_criteria)에서 회의 시작
//       전부터 시드된다.

const EMPTY_HINT = '회의에서 정리될 예정입니다.'

function CanvasRow({ label, source, filled, first = false, children }) {
  return (
    <div
      style={{
        padding: '13px 12px',
        marginTop: first ? 0 : 6,
        border: filled ? '1px solid rgba(28,26,46,0.10)' : '1px dashed rgba(98,93,114,0.42)',
        borderRadius: 10,
        background: filled ? 'rgba(255,255,255,0.72)' : 'rgba(241,238,229,0.48)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8, marginBottom: 7 }}>
        <div style={{ fontSize: 14.5, fontWeight: 600, color: '#4f466f' }}>{label}</div>
        {filled && source && (
          <span
            style={{
              fontSize: 13, fontWeight: 600, color: '#625d72', flexShrink: 0,
              background: 'var(--bg-2)', borderRadius: 999, padding: '2px 7px',
            }}
          >
            {source}
          </span>
        )}
      </div>
      {filled ? (
        <div style={{ fontSize: 15.5, fontWeight: 500, color: 'var(--text-0)', lineHeight: 1.7 }}>{children}</div>
      ) : (
        <div style={{ fontSize: 14.5, fontWeight: 500, color: '#6f697d', lineHeight: 1.65 }}>{EMPTY_HINT}</div>
      )}
    </div>
  )
}

function textOf(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : ''
}

function listOf(value, max) {
  if (!Array.isArray(value)) return []
  return value.filter((v) => typeof v === 'string' && v.trim()).slice(0, max)
}

// 가은/Claude(2026-07-23, 요청: 4개로 자르지 말고 전부 보여주되 2단 배치 + 부문명은 한 번만):
// evaluation_criteria는 "기업 평가 · 혁신성: 20점"처럼 "부문 · 기준명: 배점" 형식이라,
// 항목마다 부문명을 반복해서 보여주는 대신 부문별로 묶어 부문 라벨을 한 번만 보여주고
// 그 아래 기준명만 나열한다(2단 그리드로 배치하면 자연히 부문별로 한 칸씩 차지한다).
// 구분자가 없는 항목은 전부 하나의 "부문 없음" 묶음으로 합쳐 기존처럼 라벨 없이 보여준다.
function groupByCategory(value) {
  if (!Array.isArray(value)) return []
  const items = value.filter((v) => typeof v === 'string' && v.trim())
  const groups = []
  const groupIndexByKey = new Map()
  for (const item of items) {
    const sepIndex = item.indexOf(' · ')
    const category = sepIndex >= 0 ? item.slice(0, sepIndex) : null
    const label = sepIndex >= 0 ? item.slice(sepIndex + 3) : item
    const key = category || '__no_category__'
    if (!groupIndexByKey.has(key)) {
      groupIndexByKey.set(key, groups.length)
      groups.push({ category, labels: [] })
    }
    groups[groupIndexByKey.get(key)].labels.push(label)
  }
  return groups
}

export default function IdeaCanvasPanel({ ideationConv, analysis }) {
  if (!ideationConv) return null

  // idea_canvas(매 라운드 canvas_update 노드가 갱신한 최신 값)가 있으면 그것을, 아직
  // 없으면(첫 라운드 진행 전) selected_idea를 쓴다 — 두 값은 같은 키 구조다. 후보 선택
  // 전에는 둘 다 없다 — 그때는 심사기준(공모전 분석 시드)만 채워지고 나머지 항목은
  // "회의에서 채워질 예정"으로 남는다. 후보 카드가 여럿일 때 특정 후보의 값을 미리
  // 채우지 않는 것은 의도된 동작이다(아직 사용자의 선택이 아니므로).
  const idea = ideationConv.idea_canvas || ideationConv.selected_idea || null

  const problem = textOf(idea?.problem)
  const targetUser = textOf(idea?.target_user)
  const solution = textOf(idea?.solution)
  const coreValue = textOf(idea?.core_value)
  const differentiation = textOf(idea?.differentiation)
  const contestFit = textOf(idea?.contest_fit)
  const feasibility = idea?.feasibility ? FEASIBILITY_LABEL[idea.feasibility] || null : null
  const risks = listOf(idea?.risks, 4)
  const criteriaGroups = groupByCategory(analysis?.official_facts?.evaluation_criteria)

  return (
    <div className="card glass" style={{ marginBottom: 12, padding: 18 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
        <ClipboardList size={17} color="var(--purple)" />
        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-0)' }}>아이디어 기획 캔버스</div>
      </div>
      <div style={{ fontSize: 14.5, fontWeight: 500, color: '#625d72', lineHeight: 1.65, marginBottom: 12 }}>
        위원 발언과 공모전 분석을 바탕으로 자동으로 정리돼요.
      </div>

      <CanvasRow label="문제 상황" source="회의" filled={!!problem} first>
        {problem}
      </CanvasRow>

      <CanvasRow label="타깃 사용자" source="회의" filled={!!targetUser}>
        {targetUser}
      </CanvasRow>

      <CanvasRow label="핵심 해결 방식" source="회의" filled={!!(solution || coreValue)}>
        {solution}
        {coreValue && (
          <div style={{ marginTop: solution ? 6 : 0, color: '#514a61', fontSize: 15, fontWeight: 600 }}>
            핵심 가치 · {coreValue}
          </div>
        )}
      </CanvasRow>

      <CanvasRow label="차별점" source="회의" filled={!!differentiation}>
        {differentiation}
      </CanvasRow>

      <CanvasRow label="구현 가능성 / 리스크" source="회의" filled={!!(feasibility || risks.length > 0)}>
        {feasibility && <div>실현 가능성 {feasibility}</div>}
        {risks.length > 0 && (
          <ul style={{ margin: feasibility ? '4px 0 0' : 0, paddingLeft: 16, lineHeight: 1.7 }}>
            {risks.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        )}
      </CanvasRow>

      <CanvasRow
        label="심사기준 대응 포인트"
        source={contestFit ? '공모전 분석 + 회의' : '공모전 분석'}
        filled={criteriaGroups.length > 0 || !!contestFit}
      >
        {criteriaGroups.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 14px' }}>
            {criteriaGroups.map((group, gi) => (
              <div key={gi}>
                {group.category && (
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#514a61', marginBottom: 3 }}>
                    {group.category}
                  </div>
                )}
                <ul style={{ margin: 0, paddingLeft: 16, lineHeight: 1.6 }}>
                  {group.labels.map((label, li) => <li key={li}>{label}</li>)}
                </ul>
              </div>
            ))}
          </div>
        )}
        {contestFit && (
          <div style={{ marginTop: criteriaGroups.length > 0 ? 6 : 0 }}>
            <strong style={{ color: '#514a61', fontWeight: 700 }}>대응 · </strong>
            {contestFit}
          </div>
        )}
      </CanvasRow>
    </div>
  )
}
