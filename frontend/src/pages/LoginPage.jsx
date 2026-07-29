import { useEffect, useState } from 'react'
import { ArrowRight, CheckCircle2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { login } from '../api/authApi'
import './LoginPage.css'

export default function LoginPage() {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (localStorage.getItem('auth_token')) {
      navigate('/board', { replace: true })
    }
  }, [navigate])

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const { access_token } = await login(email, password)
      localStorage.setItem('auth_token', access_token)
      navigate('/board')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="login-page">
      <div className="login-orb login-orb-purple" />
      <div className="login-orb login-orb-green" />
      <div className="login-orb login-orb-coral" />

      <section className="login-shell" aria-label="AI Review Board 로그인">
        <div className="login-intro">
          <div className="login-brand">
            <img src="/images/logo1.png" alt="AI Review Board" className="login-brand-logo" />
          </div>

          <div className="login-intro-copy">
            <span className="login-eyebrow">AI REVIEW REVIEW</span>
            <h1>혼자 고민하던 기획서,<br />AI 전문가와 함께 완성하세요.</h1>
            <p>
              여러 관점의 AI 전문가가 공모전 기준을 분석하고,
              더 설득력 있는 문서로 발전시켜 드립니다.
            </p>
          </div>

          <ul className="login-benefits">
            <li><CheckCircle2 size={17} /> 공모전 평가 기준 자동 분석</li>
            <li><CheckCircle2 size={17} /> AI 멘토별 구체적인 피드백</li>
            <li><CheckCircle2 size={17} /> 개선 과정과 결과를 한눈에 확인</li>
          </ul>
        </div>

        <div className="login-form-side">
          <div className="login-form-card">
            <div className="login-mobile-brand">
              <img src="/images/logo1.png" alt="AI Review Board" className="login-brand-logo" />
            </div>
            <div className="login-form-heading">
              <span className="login-badge">다시 만나서 반가워요</span>
              <h2>로그인</h2>
              <p>계정에 로그인하고 작업을 이어가세요.</p>
            </div>

            <form onSubmit={handleSubmit} className="login-form">
              <label>
                <span>이메일</span>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  autoComplete="email"
                  required
                />
              </label>

              <label>
                <span>비밀번호</span>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="비밀번호를 입력하세요"
                  autoComplete="current-password"
                  required
                />
              </label>

              {error && <p className="login-error" role="alert">{error}</p>}

              <button type="submit" className="login-submit" disabled={loading}>
                <span>{loading ? '로그인 중...' : '로그인'}</span>
                {!loading && <ArrowRight size={18} />}
              </button>
            </form>

            <div className="login-divider"><span>처음 방문하셨나요?</span></div>
            <button type="button" className="login-register" onClick={() => navigate('/register')}>
              회원가입
            </button>
          </div>
        </div>
      </section>
    </main>
  )
}
