import { FileText } from 'lucide-react'

// 가은/Claude(2026-07-23) — /board "주제 아이디어 회의" 오른쪽 패널에 신청서 양식 항목을
// 보여준다. IdeationConversationScreen.jsx의 runStart()/세션 재개 경로가 이미
// getApplicationFormAnalysis()로 받아오던 값(지금까지는 백엔드 프롬프트 주입용으로만
// 쓰고 화면엔 안 보여주고 있었다)을 그대로 렌더링만 추가한 것 — 새 API 호출 없음.
// 신청서 양식을 안 올렸거나 항목을 못 찾았으면(items가 빈 배열) 패널 자체를 숨긴다.
export default function ApplicationFormPanel({ items, draft = [] }) {
  if (!items || items.length === 0) return null
  const draftById = new Map(draft.map((row) => [row.field_id, row]))

  return (
    <div className="card glass" style={{ marginBottom: 12, padding: 18 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
        <FileText size={17} color="var(--purple)" />
        <div style={{ fontSize: 16, fontWeight: 700 }}>신청서 양식 항목</div>
      </div>
      <div style={{ fontSize: 14, color: 'var(--text-2)', marginBottom: 12 }}>
        업로드한 신청서 양식에서 자동으로 뽑았어요. 회의 내용을 이 항목에 맞춰 준비하면 좋아요.
      </div>

      {items.map((item, i) => {
        const draftRow = draftById.get(item.field_id || `form_field_${i + 1}`)
        return (
        <div key={i} style={{ padding: '13px 0', borderTop: i === 0 ? 'none' : '1px solid var(--glass-border)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8 }}>
            <div style={{ fontSize: 15.5, fontWeight: 700, color: 'var(--text-1)' }}>{item.field_name}</div>
            {item.char_limit != null && (
              <span style={{ fontSize: 13, fontFamily: 'var(--mono)', color: 'var(--text-2)', flexShrink: 0 }}>
                {item.char_limit}자
              </span>
            )}
          </div>
          {item.description && (
            <div style={{ fontSize: 14, color: 'var(--text-2)', marginTop: 3, lineHeight: 1.55 }}>
              {item.description}
            </div>
          )}
          <div
            style={{
              marginTop: 8,
              padding: '10px 12px',
              borderRadius: 8,
              background: 'var(--bg-0)',
              border: '1px solid var(--glass-border)',
              fontSize: 15,
              lineHeight: 1.65,
              color: draftRow?.value ? 'var(--text-1)' : 'var(--text-2)',
              whiteSpace: 'pre-wrap',
            }}
          >
            {draftRow?.value || '아직 작성되지 않았어요.'}
          </div>
        </div>
        )
      })}
    </div>
  )
}
