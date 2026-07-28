import { useState } from 'react'
import { Check, ChevronDown, ChevronUp, FileText } from 'lucide-react'
import { isAdministrativeFormField } from './ideationConversationHelpers'

// 가은/Claude(2026-07-24, 요청: "신청기관/도시명/홈페이지 같은 건 회의로 할 이야기가
// 아니다") — 회의를 시작하기 전, 추출된 신청서 양식 항목 중 실제로 AI 위원과 상의해서
// 내용을 정할 항목만 사용자가 고르게 한다. 담당자·연락처·기관 식별 정보처럼 명백한
// 행정/개인정보 항목(isAdministrativeFormField 휴리스틱)은 기본으로 접혀 있다. 체크박스
// 대신 눌렀을 때 색이 바뀌는 버튼(칩) 형태로 — 요청한 그대로.
//
// pge/Claude(2026-07-28, 실측 요청: "다 보여주긴 해야할 거 같다 — 펼치기 버튼 하나 눌러서
// 선택 안 된 항목도 보여주게") — 한 번은 "행정 항목은 목록에서 아예 제외"로 바꿨다가, 다시
// "완전히 숨기지 말고 펼치기로 볼 수 있게"로 되돌아왔다. 기본 화면에는 행정 항목이 안
// 보이지만(자동 선택된 항목만 보임), "펼치기"를 누르면 전부 보이고 필요하면 켤 수 있다 —
// 선택 상태는 always items 원본 배열의 인덱스를 기준으로 관리해서 펼치기/접기를 오가도
// 안 흐트러진다.
export default function ApplicationFormFieldSelectModal({ items, onConfirm }) {
  const [showHidden, setShowHidden] = useState(false)
  const [selected, setSelected] = useState(() => {
    const initial = new Set()
    items.forEach((item, i) => {
      if (!isAdministrativeFormField(item.field_name)) initial.add(i)
    })
    return initial
  })

  // pge/Claude(2026-07-28, 실측 요청: "펼쳤을 때 정렬은 안 되게 — 이미 노출된 항목은
  // 순서 그대로, 더보기로 나온 항목도 그 자리에 고정") — items 원본 순서(entries) 그대로
  // 쓰면 행정 항목이 원래 자리에 다시 끼어들어와 이미 보이던 항목들의 상대 위치가
  // 바뀐다. 항상 보이는 항목을 앞에 고정하고, 펼쳤을 때는 숨겨진 항목을 그 뒤에만
  // 이어붙인다 — 이미 보이던 칩은 절대 자리를 옮기지 않는다.
  const entries = items.map((item, i) => ({ item, i }))
  const alwaysVisibleEntries = entries.filter(({ item }) => !isAdministrativeFormField(item.field_name))
  const hiddenEntries = entries.filter(({ item }) => isAdministrativeFormField(item.field_name))
  const visibleEntries = showHidden ? [...alwaysVisibleEntries, ...hiddenEntries] : alwaysVisibleEntries

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
          담당자 연락처나 신청 기관명·과제명·과제번호처럼 위원과 상의할 내용이 아닌 항목은
          기본으로 접어뒀어요. 필요하면 아래 "더보기"로 켤 수 있어요 — 선택한 항목만 회의
          중 진행위원이 같이 채워나갑니다.
        </p>

        <div style={{ display: 'flex', gap: 10, marginBottom: 12, fontSize: 11.5 }}>
          <button type="button" className="btn-ghost" style={{ padding: '4px 8px' }} onClick={() => setSelected(new Set(visibleEntries.map(({ i }) => i)))}>
            전체 선택
          </button>
          <button type="button" className="btn-ghost" style={{ padding: '4px 8px' }} onClick={() => setSelected(new Set())}>
            전체 해제
          </button>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, overflowY: 'auto', paddingBottom: 4 }}>
          {visibleEntries.map(({ item, i }) => {
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

        {hiddenEntries.length > 0 && (
          <button
            type="button"
            className="btn-ghost"
            onClick={() => setShowHidden((v) => !v)}
            style={{ alignSelf: 'flex-start', marginTop: 10, padding: '5px 10px', fontSize: 12, display: 'inline-flex', alignItems: 'center', gap: 4 }}
          >
            {showHidden ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            {showHidden ? '접기' : '더보기'}
          </button>
        )}

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
