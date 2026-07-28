import { Component } from 'react'

// 용준/Claude(2026-07-28, 실측: "회의 중간에 파란 화면 뜬다") — 이전엔 앱 어디서든 렌더링
// 중 예외가 하나만 터져도 React가 트리 전체를 언마운트해서, index.css의 body 배경색
// (#eaf3fb, 파란빛)만 남는 "파란 화면"이 됐다. 어떤 에러였는지 콘솔에만 남고 화면엔
// 아무 단서가 없어 재현·특정이 어려웠다. 이 경계가 예외를 잡아 화면은 살려두고 실제
// 에러 메시지를 보여준다 — "다시 시도"는 컴포넌트 트리만 다시 마운트한다(페이지 전체
// 새로고침이 아니므로 회의 세션 자체는 sessionStorage/부모 state에 남아있으면 이어진다).
export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary] 렌더링 중 예외 발생', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            minHeight: '100vh', display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 12,
            padding: 24, textAlign: 'center', fontFamily: 'inherit',
          }}
        >
          <div style={{ fontSize: 18, fontWeight: 700 }}>화면을 표시하는 중 오류가 발생했습니다.</div>
          <pre
            style={{
              maxWidth: 640, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              fontSize: 13, color: '#c23a6b', background: '#fff', border: '1px solid #eee',
              borderRadius: 8, padding: 12, textAlign: 'left',
            }}
          >
            {String(this.state.error?.message || this.state.error)}
          </pre>
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
            style={{
              padding: '10px 20px', borderRadius: 10, border: 'none', cursor: 'pointer',
              background: '#7c5cea', color: '#fff', fontWeight: 600, fontSize: 15,
            }}
          >
            다시 시도
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
