import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ArrowLeft, FolderOpen, LogOut, User,
  GitBranch, GraduationCap, FileText, Star, Users, Loader2, AlertCircle, Save,
} from 'lucide-react'
import { getMyProfile, updateMyProfile } from '../api/profileApi'

function decodeTokenEmail() {
  const token = localStorage.getItem('auth_token')
  if (!token) return ''

  try {
    const payload = token.split('.')[1]
    const normalized = payload.replace(/-/g, '+').replace(/_/g, '/')
    const decoded = JSON.parse(window.atob(normalized))
    return decoded.sub || ''
  } catch {
    return ''
  }
}

// --- 프로필 계약(경이 분류기가 그대로 소비 — ai/meeting/tests/fixtures/user_profile_samples.json) ---
// degree/graduation_status는 영문 enum으로 저장한다. 한글 라벨은 표시용일 뿐이다.
const DEGREE_LABEL = { bachelor: '학사', master: '석사', phd: '박사', other: '기타' }
const GRADUATION_LABEL = { graduated: '졸업', enrolled: '재학', leave: '휴학', completed: '수료' }

// GitHub URL은 계약(UserProfile)에 없는 순수 UI 편의 필드라 백엔드엔 안 보내고
// 로컬에만 캐싱한다(재방문 시 재입력 방지 — 통계는 저장된 profile.github에서 옴).
function githubUrlStorageKey(email) {
  return `mypage_github_url_${email || 'anonymous'}`
}

function defaultProfile() {
  return {
    education: { is_technical_major: false, degree: 'bachelor', graduation_status: 'graduated' },
    experience: { internship_months: 0, competition_count: 0, award_count: 0 },
  }
}

// 경이/Claude(2026-07-22, 가은님 위임): null-safe 병합. 저장된 레코드에 degree/
// graduation_status가 null로 들어있으면 기존 `{...default, ...remote}` 스프레드가 default를
// null로 덮어써서(state.degree=null) 다음 저장 때 null을 재전송하는 악순환이 있었다(윤한님과
// Network 확인 결과 백엔드는 정상, 프론트가 null을 보내던 것). null/undefined 값은 무시하고
// default를 유지하도록 필드별로 병합한다.
function mergeSection(base, over) {
  const merged = { ...base }
  for (const key of Object.keys(over || {})) {
    if (over[key] !== null && over[key] !== undefined) merged[key] = over[key]
  }
  return merged
}

function mergeProfile(remote) {
  const base = defaultProfile()
  return {
    education: mergeSection(base.education, remote?.education),
    experience: mergeSection(base.experience, remote?.experience),
  }
}

function extractGithubUsername(url) {
  if (!url) return ''
  const match = url.trim().match(/github\.com\/([^/?#\s]+)/i)
  return match ? match[1] : ''
}

// GitHub 통계 실시간 조회 (공개 REST API, 인증 없이 호출 — 시간당 60회 제한).
function useGithubStats(username) {
  const [state, setState] = useState({ status: 'idle', data: null, error: '' })

  useEffect(() => {
    if (!username) {
      setState({ status: 'idle', data: null, error: '' })
      return
    }
    let cancelled = false
    setState({ status: 'loading', data: null, error: '' })

    async function run() {
      try {
        const [userRes, reposRes] = await Promise.all([
          fetch(`https://api.github.com/users/${username}`),
          fetch(`https://api.github.com/users/${username}/repos?per_page=100&sort=updated`),
        ])
        if (userRes.status === 404) throw new Error('GitHub 사용자를 찾을 수 없습니다.')
        if (userRes.status === 403 || reposRes.status === 403) {
          throw new Error('GitHub API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.')
        }
        if (!userRes.ok || !reposRes.ok) throw new Error('GitHub 정보를 불러오지 못했습니다.')

        const user = await userRes.json()
        const repos = await reposRes.json()
        const langCounts = {}
        let totalStars = 0
        for (const repo of Array.isArray(repos) ? repos : []) {
          if (repo.language) langCounts[repo.language] = (langCounts[repo.language] || 0) + 1
          totalStars += repo.stargazers_count || 0
        }
        // 계약: github.primary_languages(예: ["Python","TypeScript"]) — 저장소 수 기준 상위 3개.
        const primaryLanguages = Object.entries(langCounts).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([lang]) => lang)

        if (!cancelled) {
          setState({
            status: 'success',
            data: {
              publicRepos: user.public_repos ?? repos.length,
              followers: user.followers ?? 0,
              totalStars,
              primaryLanguages,
            },
            error: '',
          })
        }
      } catch (err) {
        if (!cancelled) {
          setState({ status: 'error', data: null, error: err.message || 'GitHub 정보를 불러오지 못했습니다.' })
        }
      }
    }

    run()
    return () => { cancelled = true }
  }, [username])

  return state
}

// 입력 중 매 keystroke마다 GitHub API를 호출하지 않도록 디바운스.
function useDebouncedValue(value, delay = 600) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return debounced
}

export default function MyPage() {
  const navigate = useNavigate()
  const email = useMemo(() => decodeTokenEmail(), [])
  const [profile, setProfile] = useState(defaultProfile)
  const [githubUrl, setGithubUrl] = useState(() => localStorage.getItem(githubUrlStorageKey(email)) || '')
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [savedAt, setSavedAt] = useState(null)

  useEffect(() => {
    let cancelled = false
    getMyProfile()
      .then((data) => { if (!cancelled) setProfile(mergeProfile(data)) })
      .catch((err) => { if (!cancelled) setLoadError(err.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [])

  const debouncedGithubUrl = useDebouncedValue(githubUrl)
  const githubUsername = useMemo(() => extractGithubUsername(debouncedGithubUrl), [debouncedGithubUrl])
  const github = useGithubStats(githubUsername)

  function updateEducation(field, value) {
    setProfile((prev) => ({ ...prev, education: { ...prev.education, [field]: value } }))
  }

  function updateExperience(field, value) {
    setProfile((prev) => ({ ...prev, experience: { ...prev.experience, [field]: Number(value) || 0 } }))
  }

  async function handleSave() {
    // 계약: github는 api.github.com 조회로 얻은 값만 담는다(commit 수·백엔드 이력 등 조회 불가 필드 제외).
    const contractProfile = {
      ...profile,
      github: github.status === 'success' && github.data
        ? {
            connected: true,
            public_repos: github.data.publicRepos,
            followers: github.data.followers,
            total_stars: github.data.totalStars,
            primary_languages: github.data.primaryLanguages,
          }
        : { connected: false, public_repos: 0, followers: 0, total_stars: 0, primary_languages: [] },
    }
    setSaving(true)
    setSaveError('')
    try {
      const saved = await updateMyProfile(contractProfile)
      setProfile(mergeProfile(saved))
      localStorage.setItem(githubUrlStorageKey(email), githubUrl)
      setSavedAt(new Date())
    } catch (err) {
      setSaveError(err.message)
    } finally {
      setSaving(false)
    }
  }

  function handleLogout() {
    localStorage.removeItem('auth_token')
    navigate('/login')
  }

  const experienceSummary =
    `인턴 ${profile.experience.internship_months}개월 · 공모전 참여 ${profile.experience.competition_count}회 · 수상 ${profile.experience.award_count}회`
  // 경이/Claude(2026-07-22, 가은님 위임): degree/graduation_status가 null로 올 때
  // DEGREE_LABEL[null]=undefined라 "undefined undefined"가 뜨던 표시 버그 방어.
  // (근본 원인인 백엔드 degree 저장 누락은 윤한님이 별도 수정 예정 — 여기선 표시만 방어)
  const degreeLabel = DEGREE_LABEL[profile.education.degree] ?? '학위 미입력'
  const gradLabel = GRADUATION_LABEL[profile.education.graduation_status] ?? ''
  const educationSummary =
    `${degreeLabel}${gradLabel ? ' ' + gradLabel : ''} · ${profile.education.is_technical_major ? '전공' : '비전공'}`

  return (
    <div style={styles.page}>
      <main style={styles.main}>
        <button type="button" style={styles.backButton} onClick={() => navigate('/board')}>
          <ArrowLeft size={17} /> 돌아가기
        </button>

        <section style={styles.panel}>
          <div style={styles.avatar}>
            <User size={28} />
          </div>
          <div>
            <p style={styles.eyebrow}>MY PAGE</p>
            <h1 style={styles.title}>마이페이지</h1>
            <p style={styles.subtitle}>{email || '로그인 정보가 없습니다.'}</p>
          </div>
        </section>

        {/* 프로필 입력 폼: 전공여부/학위/졸업, 인턴개월/공모전참여/수상, GitHub URL */}
        <section style={styles.formCard}>
          <h2 style={styles.sectionTitle}>프로필</h2>
          {loading && <p style={styles.savedHint}>불러오는 중...</p>}
          {loadError && <p style={{ ...styles.savedHint, color: '#c05339' }}>{loadError} — 기본값으로 시작합니다.</p>}

          <div style={styles.fieldGroup}>
            <span style={styles.groupLabel}>학력</span>
            <div style={styles.fieldRow}>
              <label style={styles.field}>
                <span style={styles.label}>전공여부</span>
                <select
                  style={styles.select}
                  value={profile.education.is_technical_major ? '전공' : '비전공'}
                  onChange={(e) => updateEducation('is_technical_major', e.target.value === '전공')}
                >
                  <option value="전공">전공</option>
                  <option value="비전공">비전공</option>
                </select>
              </label>
              <label style={styles.field}>
                <span style={styles.label}>학위</span>
                <select style={styles.select} value={profile.education.degree} onChange={(e) => updateEducation('degree', e.target.value)}>
                  {Object.entries(DEGREE_LABEL).map(([enumVal, label]) => <option key={enumVal} value={enumVal}>{label}</option>)}
                </select>
              </label>
              <label style={styles.field}>
                <span style={styles.label}>졸업</span>
                <select style={styles.select} value={profile.education.graduation_status} onChange={(e) => updateEducation('graduation_status', e.target.value)}>
                  {Object.entries(GRADUATION_LABEL).map(([enumVal, label]) => <option key={enumVal} value={enumVal}>{label}</option>)}
                </select>
              </label>
            </div>
          </div>

          <div style={styles.fieldGroup}>
            <span style={styles.groupLabel}>경력 · 활동</span>
            <div style={styles.fieldRow}>
              <label style={styles.field}>
                <span style={styles.label}>인턴 (개월)</span>
                <input
                  type="number" min="0" style={styles.input} placeholder="0"
                  value={profile.experience.internship_months}
                  onChange={(e) => updateExperience('internship_months', e.target.value)}
                />
              </label>
              <label style={styles.field}>
                <span style={styles.label}>공모전 참여 (회)</span>
                <input
                  type="number" min="0" style={styles.input} placeholder="0"
                  value={profile.experience.competition_count}
                  onChange={(e) => updateExperience('competition_count', e.target.value)}
                />
              </label>
              <label style={styles.field}>
                <span style={styles.label}>수상 (회)</span>
                <input
                  type="number" min="0" style={styles.input} placeholder="0"
                  value={profile.experience.award_count}
                  onChange={(e) => updateExperience('award_count', e.target.value)}
                />
              </label>
            </div>
          </div>

          <div style={styles.fieldGroup}>
            <span style={styles.groupLabel}>GitHub</span>
            <label style={styles.field}>
              <span style={styles.label}>GitHub URL</span>
              <input
                type="url" style={styles.input} placeholder="https://github.com/username"
                value={githubUrl}
                onChange={(e) => setGithubUrl(e.target.value)}
              />
            </label>
          </div>

          <div style={styles.saveRow}>
            <button type="button" style={styles.saveButton} disabled={github.status === 'loading' || saving} onClick={handleSave}>
              <Save size={15} /> {github.status === 'loading' ? 'GitHub 통계 조회 중...' : saving ? '저장 중...' : '제출 정보 저장'}
            </button>
            {savedAt && !saveError && <span style={styles.savedHint}>최근 저장 {savedAt.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}</span>}
            {saveError && <span style={{ ...styles.savedHint, color: '#c05339' }}>{saveError}</span>}
          </div>
        </section>

        {/* 제출 정보 카드 (version-test에서 이동) */}
        <section style={styles.submissionCard}>
          <div style={styles.submissionHeader}>
            <h2 style={styles.sectionTitle}>제출 정보</h2>
            <span style={styles.submissionHint}>GitHub · 이력을 등록하면 개발 위원 피드백이 개인 맞춤형으로 바뀝니다</span>
          </div>

          <div style={styles.submissionRows}>
            <div style={styles.submissionRow}>
              <span style={styles.submissionIcon}><FileText size={16} /></span>
              <div style={styles.submissionRowBody}>
                <div style={styles.submissionRowLabel}>수정본 (기본)</div>
                <div style={styles.submissionRowValue}>문서 재제출 — 버전 추적의 기본 입력</div>
              </div>
              <span style={{ ...styles.tag, ...styles.tagGreen }}>필수</span>
            </div>

            <div style={styles.submissionRow}>
              <span style={styles.submissionIcon}><GitBranch size={16} /></span>
              <div style={styles.submissionRowBody}>
                <div style={styles.submissionRowLabel}>GitHub 저장소</div>
                <div style={styles.submissionRowValue}>
                  {githubUrl || '등록된 GitHub URL이 없습니다.'}
                </div>
                {githubUsername && (
                  <div style={styles.githubStats}>
                    {github.status === 'loading' && (
                      <span style={styles.githubStatusText}><Loader2 size={13} className="spin" /> GitHub 통계 조회 중...</span>
                    )}
                    {github.status === 'error' && (
                      <span style={{ ...styles.githubStatusText, color: '#c05339' }}><AlertCircle size={13} /> {github.error}</span>
                    )}
                    {github.status === 'success' && github.data && (
                      <>
                        <span style={styles.githubStatChip}>공개 저장소 {github.data.publicRepos}</span>
                        <span style={styles.githubStatChip}><Users size={11} /> 팔로워 {github.data.followers}</span>
                        <span style={styles.githubStatChip}><Star size={11} /> 스타 {github.data.totalStars}</span>
                        <span style={styles.githubStatChip}>주요 언어 {github.data.primaryLanguages.join(', ') || '—'}</span>
                      </>
                    )}
                  </div>
                )}
              </div>
              <span style={{ ...styles.tag, ...styles.tagPurple }}>선택</span>
            </div>

            <div style={styles.submissionRow}>
              <span style={styles.submissionIcon}><GraduationCap size={16} /></span>
              <div style={styles.submissionRowBody}>
                <div style={styles.submissionRowLabel}>이력 · 교육 수준</div>
                <div style={styles.submissionRowValue}>{educationSummary} · {experienceSummary}</div>
              </div>
              <span style={{ ...styles.tag, ...styles.tagPurple }}>선택</span>
            </div>
          </div>
        </section>

        <section style={styles.card}>
          <button type="button" style={styles.rowButton} onClick={() => navigate('/projects')}>
            <span style={styles.rowLeft}>
              <FolderOpen size={18} />
              내 프로젝트
            </span>
            <span style={styles.rowArrow}>›</span>
          </button>
          <button type="button" style={{ ...styles.rowButton, ...styles.logoutButton }} onClick={handleLogout}>
            <span style={styles.rowLeft}>
              <LogOut size={18} />
              로그아웃
            </span>
            <span style={styles.rowArrow}>›</span>
          </button>
        </section>
      </main>
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100vh',
    background:
      'radial-gradient(1100px 600px at 12% -10%, rgba(124,92,234,0.10), transparent 60%), ' +
      'radial-gradient(900px 500px at 100% 10%, rgba(22,163,122,0.07), transparent 55%), ' +
      'radial-gradient(800px 500px at 50% 110%, rgba(224,96,61,0.06), transparent 55%), #faf8f4',
    color: '#1c1a2e',
    fontFamily: "'Pretendard Variable', Pretendard, -apple-system, sans-serif",
    padding: '32px 20px',
  },
  main: {
    maxWidth: 640,
    margin: '0 auto',
  },
  backButton: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 7,
    marginBottom: 22,
    padding: '10px 14px',
    borderRadius: 12,
    border: '1px solid rgba(28,26,46,0.10)',
    background: 'rgba(255,255,255,0.72)',
    color: '#5b5770',
    cursor: 'pointer',
    fontSize: 15.5,
  },
  panel: {
    display: 'flex',
    alignItems: 'center',
    gap: 18,
    padding: 22,
    borderRadius: 16,
    border: '1px solid rgba(28,26,46,0.10)',
    background: 'rgba(255,255,255,0.72)',
    boxShadow: '0 2px 14px rgba(28,26,46,0.05)',
    backdropFilter: 'blur(14px)',
    marginBottom: 16,
  },
  avatar: {
    width: 58,
    height: 58,
    borderRadius: 18,
    background: 'rgba(124,92,234,0.12)',
    color: '#7c5cea',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  eyebrow: {
    margin: '0 0 4px',
    fontSize: 12.5,
    fontWeight: 700,
    color: '#7c5cea',
    letterSpacing: '0.08em',
  },
  title: {
    margin: 0,
    fontSize: 27,
    fontWeight: 700,
  },
  subtitle: {
    margin: '6px 0 0',
    fontSize: 14.5,
    color: '#918d9f',
  },
  formCard: {
    borderRadius: 16,
    border: '1px solid rgba(28,26,46,0.10)',
    background: 'rgba(255,255,255,0.72)',
    boxShadow: '0 2px 14px rgba(28,26,46,0.05)',
    backdropFilter: 'blur(14px)',
    padding: '20px 22px',
    marginBottom: 16,
  },
  sectionTitle: {
    margin: '0 0 14px',
    fontSize: 17.5,
    fontWeight: 700,
  },
  fieldGroup: {
    marginBottom: 16,
  },
  groupLabel: {
    display: 'block',
    fontSize: 12.5,
    fontWeight: 700,
    color: '#7c5cea',
    letterSpacing: '0.04em',
    marginBottom: 8,
  },
  fieldRow: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
  },
  field: {
    flex: '1 1 140px',
    display: 'flex',
    flexDirection: 'column',
    gap: 5,
  },
  label: {
    fontSize: 13.5,
    fontWeight: 600,
    color: '#5b5770',
  },
  select: {
    padding: '9px 10px',
    borderRadius: 10,
    border: '1px solid rgba(28,26,46,0.14)',
    background: '#fff',
    color: '#1c1a2e',
    fontSize: 15,
  },
  input: {
    padding: '9px 10px',
    borderRadius: 10,
    border: '1px solid rgba(28,26,46,0.14)',
    background: '#fff',
    color: '#1c1a2e',
    fontSize: 15,
  },
  saveRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginTop: 4,
  },
  saveButton: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 7,
    padding: '10px 16px',
    borderRadius: 10,
    border: 'none',
    background: '#7c5cea',
    color: '#fff',
    fontSize: 15,
    fontWeight: 700,
    cursor: 'pointer',
  },
  savedHint: {
    fontSize: 13.5,
    color: '#918d9f',
  },
  submissionCard: {
    borderRadius: 16,
    border: '1px solid rgba(28,26,46,0.10)',
    background: 'rgba(255,255,255,0.72)',
    boxShadow: '0 2px 14px rgba(28,26,46,0.05)',
    backdropFilter: 'blur(14px)',
    padding: '18px 20px',
    marginBottom: 16,
  },
  submissionHeader: {
    marginBottom: 14,
  },
  submissionHint: {
    fontSize: 13.5,
    color: '#918d9f',
  },
  submissionRows: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  submissionRow: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 11,
    padding: '10px 12px',
    borderRadius: 10,
    background: 'rgba(124,92,234,0.04)',
    border: '1px solid rgba(28,26,46,0.06)',
  },
  submissionIcon: {
    flexShrink: 0,
    width: 32,
    height: 32,
    borderRadius: 9,
    background: 'rgba(124,92,234,0.10)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#7c5cea',
  },
  submissionRowBody: {
    flex: 1,
    minWidth: 0,
  },
  submissionRowLabel: {
    fontSize: 14.5,
    fontWeight: 700,
    color: '#3a3750',
  },
  submissionRowValue: {
    fontSize: 13,
    color: '#918d9f',
    marginTop: 2,
    overflowWrap: 'anywhere',
  },
  githubStats: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    flexWrap: 'wrap',
    marginTop: 8,
  },
  githubStatusText: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
    fontSize: 13,
    color: '#918d9f',
  },
  githubStatChip: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    fontSize: 12.5,
    fontWeight: 600,
    color: '#5b5770',
    background: 'rgba(28,26,46,0.06)',
    padding: '3px 9px',
    borderRadius: 99,
  },
  tag: {
    flexShrink: 0,
    fontSize: 12.5,
    fontWeight: 700,
    padding: '3px 10px',
    borderRadius: 8,
  },
  tagGreen: {
    color: '#16a37a',
    background: 'rgba(22,163,122,0.12)',
  },
  tagPurple: {
    color: '#7c5cea',
    background: 'rgba(124,92,234,0.12)',
  },
  card: {
    borderRadius: 16,
    border: '1px solid rgba(28,26,46,0.10)',
    background: 'rgba(255,255,255,0.72)',
    boxShadow: '0 2px 14px rgba(28,26,46,0.05)',
    backdropFilter: 'blur(14px)',
    overflow: 'hidden',
  },
  rowButton: {
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '17px 18px',
    border: 'none',
    borderBottom: '1px solid rgba(28,26,46,0.10)',
    background: 'transparent',
    color: '#1c1a2e',
    cursor: 'pointer',
    fontSize: 16.5,
    fontWeight: 600,
  },
  rowLeft: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 10,
  },
  rowArrow: {
    fontSize: 24,
    color: '#918d9f',
    lineHeight: 1,
  },
  logoutButton: {
    borderBottom: 'none',
    color: '#c05339',
  },
}
