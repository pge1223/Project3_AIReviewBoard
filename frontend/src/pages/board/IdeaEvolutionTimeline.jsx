import { useState } from 'react'
import { CheckCircle2, GitMerge, Lightbulb, MessageSquareWarning, Pencil, Trash2, FlaskConical, MousePointerClick } from 'lucide-react'

// 용준/Claude(2026-07-27): idea_evolution(append-only, 백엔드가 이미 시간순으로 쌓아주는
// 배열)을 세로 타임라인으로 그리는 순수 표시 컴포넌트. API 호출 없음. action_type/changed_by
// 값 자체는 raw 데이터에 있는 그대로만 쓰고(요청: "don't invent data that isn't in the
// array"), 여기서는 그 값을 사람이 읽을 라벨/아이콘으로만 바꾼다.
const ACTION_TYPE_META = {
  divergence: { label: '해결 방향 생성', icon: Lightbulb, color: '#5e3ec8' },
  critique: { label: '반론', icon: MessageSquareWarning, color: '#c2410c' },
  merge: { label: '방향 결합', icon: GitMerge, color: '#2f6fd6' },
  revision: { label: '방향 수정', icon: Pencil, color: '#805800' },
  drop: { label: '방향 폐기', icon: Trash2, color: '#9a3b3b' },
  candidate_generation: { label: '잠정 후보 생성', icon: Lightbulb, color: '#5e3ec8' },
  validation: { label: '검증 결과', icon: FlaskConical, color: '#087557' },
  confirmation: { label: '사용자 최종 확정', icon: CheckCircle2, color: '#5e3ec8' },
  // 용준/Claude(2026-07-27, 요청: "problem focus 변화 이력 추가") — 문제 영역 선택/결합.
  problem_focus_selected: { label: '문제 영역 선택', icon: MousePointerClick, color: '#1f6f6f' },
  problem_focus_merged: { label: '문제 영역 결합', icon: GitMerge, color: '#1f6f6f' },
}

const CHANGED_BY_LABEL = {
  planning_expert: '기획위원',
  dev_expert: '개발위원',
  ideation_facilitator: '진행자',
  user: '사용자',
}

const CONTENT_TRUNCATE_LENGTH = 160

function directionTitlesFor(record, solutionDirections, problemAreas) {
  const ids = record?.target_direction_ids || []
  if (ids.length === 0) return ''
  const byDirectionId = new Map((solutionDirections || []).map((d) => [d.direction_id, d.title]))
  const byAreaId = new Map((problemAreas || []).map((a) => [a.area_id, a.title]))
  return ids.map((id) => byDirectionId.get(id) || byAreaId.get(id) || id).join(' · ')
}

function TimelineEntry({ record, solutionDirections, problemAreas, isLast }) {
  const [expanded, setExpanded] = useState(false)
  const meta = ACTION_TYPE_META[record.action_type] || { label: record.action_type || '기록', icon: Lightbulb, color: 'var(--text-2)' }
  const Icon = meta.icon
  const changedByLabel = CHANGED_BY_LABEL[record.changed_by] || record.changed_by
  const directionTitles = directionTitlesFor(record, solutionDirections, problemAreas)
  const content = record.content || ''
  const isLong = content.length > CONTENT_TRUNCATE_LENGTH
  const displayContent = expanded || !isLong ? content : `${content.slice(0, CONTENT_TRUNCATE_LENGTH)}…`

  return (
    <div style={{ display: 'flex', gap: 10 }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
        <div
          style={{
            width: 26, height: 26, borderRadius: '50%', display: 'grid', placeItems: 'center',
            background: '#fff', border: `2px solid ${meta.color}`, color: meta.color, flexShrink: 0,
          }}
        >
          <Icon size={13} />
        </div>
        {!isLast && <div style={{ flex: 1, width: 2, background: 'var(--glass-border)', marginTop: 2 }} />}
      </div>
      <div style={{ paddingBottom: isLast ? 0 : 16, flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: meta.color }}>{meta.label}</span>
          {changedByLabel && (
            <span className="badge grey mono" style={{ fontSize: 11.5, padding: '2px 7px' }}>{changedByLabel}</span>
          )}
        </div>
        {record.title && (
          <div style={{ marginTop: 3, fontSize: 14.5, fontWeight: 600, color: 'var(--text-0)' }}>{record.title}</div>
        )}
        {directionTitles && (
          <div style={{ marginTop: 2, fontSize: 12.5, color: 'var(--text-2)' }}>대상 · {directionTitles}</div>
        )}
        {content && (
          <div style={{ marginTop: 4, fontSize: 13.5, color: 'var(--text-1)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
            {displayContent}
          </div>
        )}
        {isLong && (
          <button
            type="button"
            className="btn-ghost"
            style={{ marginTop: 4, padding: '2px 8px', fontSize: 12.5 }}
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? '접기' : '더보기'}
          </button>
        )}
      </div>
    </div>
  )
}

// pge/Claude(2026-07-28, 요청: "탭이랑 패널이 붙어있어야 해") — IdeaCanvasPanel과 같은
// 이유로 bare 옵션을 추가한다(그 파일 주석 참고).
export default function IdeaEvolutionTimeline({
  idea_evolution: ideaEvolution,
  solution_directions: solutionDirections,
  problem_areas: problemAreas,
  bare = false,
}) {
  const records = ideaEvolution || []
  if (records.length === 0) return null

  return (
    <div className={bare ? undefined : 'card glass'} style={{ padding: 14 }}>
      {/* pge/Claude(2026-07-28, 요청: "탭에 로고 박고 패널 안 타이틀은 지워줘") — 제목 줄은
          이제 탭 버튼(IdeationConversationScreen.jsx)이 보여주므로 중복 제거. */}
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {records.map((record, i) => (
          <TimelineEntry
            key={record.record_id || i}
            record={record}
            solutionDirections={solutionDirections}
            problemAreas={problemAreas}
            isLast={i === records.length - 1}
          />
        ))}
      </div>
    </div>
  )
}
