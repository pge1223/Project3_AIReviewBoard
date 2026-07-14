# 팀 Claude Code 세팅 가이드

## 1. 이 구조를 실제 레포에 적용하는 방법
1. 이 폴더 안의 파일들을 실제 프로젝트 레포 루트에 그대로 복사
   (이미 레포가 있다면 각 CLAUDE.md만 해당 위치에 병합)
2. 루트에서 `git add CLAUDE.md apps/*/CLAUDE.md services/*/CLAUDE.md
   .env.example .claude/settings.json .gitignore` 후 커밋
3. 팀원 각자 `git pull` → 각자 폴더(apps/mobile, apps/server 등)에서
   `claude` 실행하면 루트 CLAUDE.md + 해당 폴더 CLAUDE.md가 자동으로
   컨텍스트에 로드됨 (Claude Code는 현재 디렉토리부터 상위로 올라가며
   CLAUDE.md를 모두 읽음)

## 2. 왜 이렇게 나누나
- 5명이 같은 프로젝트를 보되 담당 영역이 다름 → 루트는 "전체가 알아야 할 것"
  (아키텍처, 팀 규칙, 확정된 의사결정), 하위 폴더는 "그 파트만 알아야 할 것"
- 이렇게 하면 이재인이 apps/server에서 작업할 때 모바일 UI 세부사항까지
  컨텍스트에 안 끌려와서 Claude Code 응답 품질이 더 좋아짐

## 3. 환경변수 공유 (실제 값)
- `.env.example`은 커밋 O, 실제 `.env`는 커밋 X
- 실제 값 전달 방법 (택 1, 팀 합의 필요):
  - **비공개 노션 페이지** — 가장 간단, 접근 로그 없음이 단점
  - **1Password / Bitwarden 공유 Vault** — 접근 관리/로테이션 용이, 추천
  - **NCP Secret Manager** — 서버 배포 시 실제 운영에도 재사용 가능, 인프라
    담당(김윤한)이 설정
- 누가 어떤 값을 갖고 있는지 모르는 상황 방지를 위해, `.env.example`에
  키가 추가될 때마다 PR 설명에 "새 환경변수 추가됨, [채널]에 값 공유함"
  이라고 남기기

## 4. 각자 개인 설정 (git에 안 올라가는 것)
- 로컬에서만 다른 규칙이 필요하면 각자 `CLAUDE.local.md` 생성
  (예: "나는 pnpm 대신 yarn 씀", "로컬 MongoDB는 docker-compose로 띄움")
  → `.gitignore`에 이미 포함되어 있어 커밋 안 됨
- `.claude/settings.local.json`도 개인별 권한 오버라이드용, 마찬가지로
  gitignore 처리됨

## 5. 앞으로 CLAUDE.md 유지보수 원칙
- 요구사항 문서(요구사항정의서)가 바뀌면 관련 CLAUDE.md도 같은 PR에서 갱신
- "이미 결정되고 재논의 안 하기로 한 것"(예: HWP 제외)은 CLAUDE.md에
  명시해서 Claude Code가 매번 같은 걸 다시 제안 안 하게 하기
- 팀 미팅에서 새로 확정되는 사항은 이 문서들에 바로 반영 → 다음 세션부터
  전 팀원의 Claude Code가 같은 최신 맥락으로 시작
