import { useState } from 'react'
import { Check, FileText } from 'lucide-react'
import { isAdministrativeFormField } from './ideationConversationHelpers'

// 가은/Claude(2026-07-24, 요청: "신청기관/도시명/홈페이지 같은 건 회의로 할 이야기가
// 아니다") — 회의를 시작하기 전, 추출된 신청서 양식 항목 중 실제로 AI 위원과 상의해서
// 내용을 정할 항목만 사용자가 고르게 한다. 담당자·연락처·기관 식별 정보처럼 명백한
// 행정/개인정보 항목은 기본으로 꺼둔다(isAdministrativeFormField 휴리스틱). 체크박스
// 대신 눌렀을 때 색이 바뀌는 버튼(칩) 형태로 — 요청한 그대로.
export default function ApplicationFormFieldSelectModal({ items, onConfirm }) {
  const [selected, setSelected] = useState(() => {
    const initial = new Set()
    items.forEach((item, i) => {
      if (!isAdministrativeFormField(item.field_name)) initial.add(i)
    })
    return initial
  })

  function toggle(i) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(28,26,46,0.45)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: 20,
      }}
    >
      <div
        className="card glass"
        style={{ maxWidth: 560, width: '100%', maxHeight: '80vh', display: 'flex', flexDirection: 'column', padding: 22 }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <FileText size={16} color="var(--purple)" />
          <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>회의에서 다룰 신청서 항목을 골라주세요</h3>
        </div>
        <p style={{ fontSize: 12.5, color: 'var(--text-2)', lineHeight: 1.6, marginBottom: 14 }}>
          담당자 연락처나 신청 기관명처럼 위원과 상의할 내용이 아닌 항목은 기본으로
          꺼뒀어요. 필요하면 눌러서 켜고 끌 수 있어요 — 선택한 항목만 회의 중 진행자가
          같이 채워나갑니다.
        </p>

        <div style={{ display: 'flex', gap: 10, marginBottom: 12, fontSize: 11.5 }}>
          <button type="button" className="btn-ghost" style={{ padding: '4px 8px' }} onClick={() => setSelected(new Set(items.map((_, i) => i)))}>
            전체 선택
          </button>
          <button type="button" className="btn-ghost" style={{ padding: '4px 8px' }} onClick={() => setSelected(new Set())}>
            전체 해제
          </button>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, overflowY: 'auto', paddingBottom: 4 }}>
          {items.map((item, i) => {
            const isSelected = selected.has(i)
            return (
              <button
                key={i}
                type="button"
                onClick={() => toggle(i)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 5,
                  padding: '7px 12px',
                  borderRadius: 999,
                  fontSize: 12.5,
                  fontWeight: 600,
                  cursor: 'pointer',
                  border: isSelected ? '1px solid var(--purple)' : '1px solid var(--glass-border)',
                  background: isSelected ? 'var(--purple)' : 'var(--bg-1)',
                  color: isSelected ? '#fff' : 'var(--text-1)',
                  transition: 'background 0.12s ease, color 0.12s ease, border-color 0.12s ease',
                }}
              >
                {isSelected && <Check size={12} />}
                {item.field_name}
              </button>
            )
          })}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 18 }}>
          <span style={{ fontSize: 11.5, color: 'var(--text-2)' }}>
            {selected.size}개 선택됨
            {selected.size === 0 && ' · 전부 끄면 신청서 코칭 없이 일반 회의로 진행돼요'}
          </span>
          <button
            type="button"
            className="btn-primary"
            onClick={() => onConfirm(items.filter((_, i) => selected.has(i)))}
          >
            이 항목으로 시작하기
          </button>
        </div>
      </div>
    </div>
  )
}
