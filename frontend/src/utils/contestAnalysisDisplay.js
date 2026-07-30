// 가은/Claude(2026-07-24, 요청: 공모전 분석 결과 화면 개편) — 백엔드
// AnnouncementAnalysisResponse.official_facts.evaluation_criteria는 평평한 문자열
// 배열("기업 평가 · 혁신성: 20점" 형태)이라, UI에서 기업/도시 탭·배점 막대로 보여주려면
// 표시용으로만 그룹핑/파싱한다. 원문에 구분자(" · ")나 배점("N점")이 없으면 억지로
// 만들어내지 않고 원문 그대로 한 그룹에 담는다 — 값을 지어내지 않는다는 프로젝트 원칙.
export function parseEvaluationCriteria(rawList) {
  if (!Array.isArray(rawList) || rawList.length === 0) return [];

  const order = [];
  const groups = new Map();

  for (const raw of rawList) {
    if (!raw) continue;
    const sepIndex = raw.indexOf(' · ');
    const groupName = sepIndex !== -1 ? raw.slice(0, sepIndex).trim() : '평가 기준';
    const rest = sepIndex !== -1 ? raw.slice(sepIndex + 3).trim() : raw.trim();

    const scoreMatch = rest.match(/^(.*?)[:：]\s*(\d+)\s*점\s*$/) || rest.match(/^(.*?)\s+(\d+)\s*점\s*$/);
    const name = scoreMatch ? scoreMatch[1].trim() : rest;
    const score = scoreMatch ? Number(scoreMatch[2]) : null;

    if (!groups.has(groupName)) {
      groups.set(groupName, []);
      order.push(groupName);
    }
    groups.get(groupName).push({ raw, name, score });
  }

  return order.map((groupName) => {
    const items = groups.get(groupName);
    const scores = items.map((i) => i.score).filter((s) => s != null);
    return {
      groupName,
      items,
      maxScore: scores.length > 0 ? Math.max(...scores) : null,
      totalScore: scores.length > 0 ? scores.reduce((sum, s) => sum + s, 0) : null,
    };
  });
}

// 핵심 키워드 태그: 별도 keywords 필드가 없어서, 실제 평가 기준 항목명(위 파싱 결과)에서
// 중복 없이 추출한다 — 임의의 목업 키워드를 만들지 않는다.
export function deriveKeywordTags(criteriaGroups, limit = 8) {
  const seen = new Set();
  const tags = [];
  for (const group of criteriaGroups) {
    for (const item of group.items) {
      const label = (item.name || item.raw || '').replace(/\s+/g, '');
      if (label && !seen.has(label)) {
        seen.add(label);
        tags.push(label);
      }
    }
    if (tags.length >= limit) break;
  }
  return tags.slice(0, limit);
}

export const CONFIDENCE_LABEL = { high: '확신 높음', medium: '확신 보통', low: '확신 낮음' };

// 서로 다른 공모 부문은 선택 트랙이므로 점수를 합산하면 안 된다. 그룹이 하나일 때만
// totalScore를 제공하고, 여러 그룹이면 부문 수와 부문별 항목 수를 반환해 UI가
// "10개 항목·195점"처럼 서로 다른 평가표를 하나로 더하지 않게 한다.
export function summarizeCriteria(criteriaGroups) {
  const items = criteriaGroups.flatMap((group) => group.items);
  const scored = items.filter((item) => item.score != null);
  const groupCount = criteriaGroups.length;
  const itemCounts = criteriaGroups.map((group) => group.items.length);
  const perGroupItemCount = (
    itemCounts.length > 0 && itemCounts.every((count) => count === itemCounts[0])
  ) ? itemCounts[0] : null;
  return {
    itemCount: items.length,
    groupCount,
    perGroupItemCount,
    totalScore: groupCount === 1 && scored.length > 0
      ? scored.reduce((sum, item) => sum + item.score, 0)
      : null,
  };
}
