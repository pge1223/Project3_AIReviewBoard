# 작성자: 경이
# 목적: 평가 리포트(docs/evaluation/)용 테스트셋 결정론 특성 실측 — 수행보고서 버전
#       시리즈(v1.0~v1.3)를 서버와 동일한 파서(extract_document)로 읽고, 채점 엔진의
#       캘리브레이션 신호(calibration.py의 실제 함수)로 문서 특성과 상한 발동 내역을
#       재현 가능한 표로 출력한다. LLM 호출 없음 — 전부 결정론이라 몇 번을 돌려도 같다.
# 사용: python scripts/eval_report_signals.py <문서 폴더>
#       (폴더에 수행보고서_v1.0.pdf ~ 수행보고서_v1.3*.pdf 가 있어야 함)

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ai" / "meeting"))

from ai.rag.parsers import extract_document  # noqa: E402  (서버 업로드와 동일 경로)
from scoring.calibration import (  # noqa: E402
    _NAMED_ORGANIZATION_RE,
    _QUANTITATIVE_RE,
    _criterion_evidence_text,
    _future_sentence_ratio,
    _normalize,
    _split_sections,
    build_score_cap,
)

# 이 프로젝트 공고문에서 추출된 rubric과 동일한 구성(점수 체계표 실측 기준).
CRITERIA = [
    {"criterion_id": "goal_alignment", "criterion_name": "목표 부합성", "max_score": 15,
     "description": "지역·행정 현안 해결 수준, 문제정의의 명확성, AI 도입 필요성·적절성을 평가합니다."},
    {"criterion_id": "tech_innovation", "criterion_name": "기술성·혁신성", "max_score": 30,
     "description": "데이터 확보·품질 우수성, 알고리즘·인프라 적정성, 서비스 혁신성을 평가합니다."},
    {"criterion_id": "feasibility", "criterion_name": "실현 가능성", "max_score": 30,
     "description": "구현 가능성, 완결성, 법·제도적 제약 수준, 예산·추진체계 적합성을 평가합니다."},
    {"criterion_id": "diffusion_effect", "criterion_name": "확산성·효과성", "max_score": 10,
     "description": "타 기관/분야로의 확산 가능성, 대국민 체감도, 성과 지표 제시 여부를 평가합니다."},
]
RUBRIC = {"criteria": CRITERIA, "total_max_score": 85}


def analyze_file(path: Path) -> dict:
    extraction = extract_document(path)
    text = "\n".join(block.content for block in extraction.blocks)
    norm = _normalize(text)
    compact = re.sub(r"\s+", "", norm)
    sections = _split_sections(norm)
    row = {
        "file": path.name,
        "chars": len(compact),
        "sections": len(sections),
        "named_orgs": len(set(_NAMED_ORGANIZATION_RE.findall(norm))),
        "quant_hits": len(_QUANTITATIVE_RE.findall(norm)),
        "future_ratio": round(_future_sentence_ratio(norm), 3),
        "caps": [],
    }
    for criterion in CRITERIA:
        cap = build_score_cap(RUBRIC, criterion, {"text": text})
        evidence = _criterion_evidence_text(norm, criterion)
        row["caps"].append({
            "criterion": criterion["criterion_name"],
            "evidence_chars": len(evidence),
            "cap": None if cap is None else str(cap.get("cap_score")),
            "signals": [] if cap is None else [s["code"] for s in cap["signals"]],
        })
    return row


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files = sorted(folder.glob("수행보고서_v1.*.pdf"))
    if not files:
        raise SystemExit(f"수행보고서_v1.*.pdf 를 찾지 못했습니다: {folder}")
    print(f"{'파일':<28}{'글자수':>8}{'섹션':>5}{'실명기관':>7}{'정량표현':>7}{'미래형':>7}")
    rows = [analyze_file(f) for f in files]
    for row in rows:
        print(
            f"{row['file']:<28}{row['chars']:>8}{row['sections']:>5}"
            f"{row['named_orgs']:>7}{row['quant_hits']:>7}{row['future_ratio']:>7}"
        )
    print()
    for row in rows:
        capped = [c for c in row["caps"] if c["cap"] is not None]
        if capped:
            detail = ", ".join(f"{c['criterion']}={c['cap']}({'+'.join(c['signals'])})" for c in capped)
            print(f"{row['file']}: 상한 발동 → {detail}")
        else:
            print(f"{row['file']}: 상한 발동 없음 (4개 평가 항목 모두)")


if __name__ == "__main__":
    main()
