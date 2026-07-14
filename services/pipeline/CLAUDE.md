# services/pipeline — 데이터 수집/동기화 파이프라인 + NCP 서버 설정

담당: 김윤한

## 이 폴더 컨텍스트
- 데이터 소스: K-Startup API + 기업마당 API (dual-source)
- 소스 우선순위:
  1. API 본문(`bsnsSumryCn`) — 1차
  2. 공고 URL 크롤링 — 보강
  3. PDF 파싱 — 정규 단계 (필수)
  4. HWP 파싱 — **제외** (재논의 금지)
- 배치: 매일 upsert, 만료 공고 soft-delete (hard delete 금지 — 이력 추적용)

## NCP 서버 설정
- 서버 구성/배포 스크립트는 이 폴더에서 관리
- 인프라 관련 시크릿(NCP access key 등)은 `.env`로만 주입, 코드에 금지

## 하지 말 것
- soft-delete 대신 물리 삭제 로직 추가하지 말 것
