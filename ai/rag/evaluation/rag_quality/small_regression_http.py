# 작성자: 용준/Claude(2026-07-29, 요청: 88건 전체 평가 전 실제 LLM 소규모 통합 회귀 —
#       /reply, /reply/stream HTTP 계약 검증)
# 목적: small_regression_llm.py가 그래프 레벨(생성 품질·라우팅)을 검증한다면, 이 스크립트는
#       실제 FastAPI 라우트 코드(app.api.routes.ideation_conversation_preview)를
#       fastapi.testclient.TestClient로 그대로 실행해 HTTP 계약(NDJSON 종료 이벤트, 메시지
#       중복 없음, request_type/phase, /reply와 /reply/stream 결과 동일성)을 실제 OpenAI
#       호출로 확인한다. backend/tests/test_ideation_reply_request_type_e2e.py와 같은
#       구조지만, 거기서는 스텁 LLM을 썼고 여기서는 _build_llm_call/_build_streaming_backends를
#       전혀 monkeypatch하지 않는다(실제 키로 실제 호출) — CI에 상시 넣지 않고 회귀가 필요할
#       때 수동 실행하는 스크립트로 분리했다.
#
#       RAG 근거 품질(evidence_ref 연결·unsupported claim)은 small_regression_llm.py가 이미
#       고정 검색 결과로 검증했으므로, 여기서는 use_rag=False로 시작해 배선(라우팅/스트리밍/
#       세션 상태)만 본다 — 이 스크립트는 로컬에 Chroma 벡터DB가 없어(.env에
#       CHROMA_PERSIST_DIR 미설정) 실검색을 태울 수도 없다.
#
# 보안: API 키는 settings.OPENAI_API_KEY로만 읽고, 로그/리포트에 출력하지 않는다.
#
# 실행 예:
#   python -m ai.rag.evaluation.rag_quality.small_regression_http \
#       --output reports/small_regression_llm_20260729/http_report.json
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI
from fastapi.testclient import TestClient

_CASES = {
    "document_fact_query": "이번 공모전 제출 서류가 뭔가요?",
    "session_state_query": "지금 어떤 단계인가요?",
    "expert_analysis_query": "구현 가능성과 MVP 위험을 분석해 주세요",
    "ideation_discussion_request": "이 부분에 대해 위원들이 좀 더 토론해 주세요",
}
_SINGLE_TURN_TYPES = {"document_fact_query", "session_state_query", "expert_analysis_query"}
_FALLBACK_MARKERS = ("현재 자료로 확인할 수 없", "확인할 수 없습니다")


def _read_ndjson_events(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _start_session(client: TestClient) -> str:
    resp = client.post(
        "/ideation-conversation/start",
        json={"competition_name": "소규모 회귀용 공모전", "user_idea": "소상공인 손님 문의 자동 응대 챗봇"},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"/start 실패: status={resp.status_code} body={resp.text[:300]}")
    return resp.json()["session_id"]


def _run_sync(client: TestClient, conv_route, request_type: str, user_message: str) -> dict:
    session_id = _start_session(client)
    baseline_count = len(conv_route._store.get_record(session_id).state["messages"])
    resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": user_message})
    if resp.status_code != 200:
        return {"error": f"status={resp.status_code} body={resp.text[:300]}"}
    body = resp.json()
    new_messages = body["messages"][baseline_count:]
    speakers = [m["speaker_id"] for m in new_messages]
    content = next((m["content"] for m in reversed(new_messages) if m["speaker_id"] != "user"), "")
    return {
        "status_code": resp.status_code,
        "request_type": body.get("request_type"),
        "phase": body.get("phase"),
        "speaker_sequence": speakers,
        "expected_single_turn": request_type in _SINGLE_TURN_TYPES,
        "actual_single_turn": len(speakers) == 2,
        "has_fallback_phrase": any(marker in content for marker in _FALLBACK_MARKERS),
        "content_sample": content,
    }


def _run_stream(client: TestClient, conv_route, request_type: str, user_message: str) -> dict:
    session_id = _start_session(client)
    baseline_count = len(conv_route._store.get_record(session_id).state["messages"])
    try:
        with client.stream(
            "POST",
            f"/ideation-conversation/{session_id}/reply/stream",
            json={"message": user_message},
        ) as resp:
            status_code = resp.status_code
            content_type = resp.headers.get("content-type", "")
            if status_code != 200:
                return {"error": f"status={status_code}"}
            events = _read_ndjson_events(resp)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"stream_exception: {type(exc).__name__}: {exc}"}

    types = [e.get("type") for e in events]
    final_state = next((e["state"] for e in events if e.get("type") == "state"), None)
    message_ids = [m.get("id") for m in (final_state or {}).get("messages", []) if "id" in m]
    new_messages = (final_state or {}).get("messages", [])[baseline_count:]
    speakers = [m["speaker_id"] for m in new_messages]
    return {
        "status_code": status_code,
        "content_type_ok": content_type.startswith("application/x-ndjson"),
        "event_types": types,
        "ends_with_done": types[-1] == "done" if types else False,
        "done_phase_matches_state": (events[-1].get("phase") == (final_state or {}).get("phase")) if types and types[-1] == "done" else None,
        "request_type": (final_state or {}).get("request_type"),
        "phase": (final_state or {}).get("phase"),
        "speaker_sequence": speakers,
        "expected_single_turn": request_type in _SINGLE_TURN_TYPES,
        "actual_single_turn": len(speakers) == 2,
        "message_ids_unique": len(message_ids) == len(set(message_ids)),
    }


def main(argv: Optional[list[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="/reply, /reply/stream 실제 LLM HTTP 계약 소규모 회귀")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    from app.config import settings

    if not settings.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY가 설정되지 않았습니다(backend/.env 확인) — 키 값은 여기 출력하지 않습니다.")
    if not settings.ENABLE_IDEATION_PREVIEW:
        raise SystemExit("ENABLE_IDEATION_PREVIEW가 꺼져 있습니다(backend/.env 확인).")

    import app.api.routes.ideation_conversation_preview as conv_route

    app = FastAPI()
    app.include_router(conv_route.router)
    client = TestClient(app)

    report: dict[str, Any] = {
        "note": "실제 OpenAI 호출(라우트 기본 모델 사용) — API 키 값은 어디에도 기록하지 않았다. use_rag=False로 시작해 RAG 근거 품질이 아니라 HTTP 계약(NDJSON/중복/상태)만 검증한다.",
        "cases": {},
    }
    for request_type, user_message in _CASES.items():
        sync_result = _run_sync(client, conv_route, request_type, user_message)
        stream_result = _run_stream(client, conv_route, request_type, user_message)
        equivalent = (
            "error" not in sync_result
            and "error" not in stream_result
            and sync_result.get("request_type") == stream_result.get("request_type")
            and sync_result.get("phase") == stream_result.get("phase")
            and sync_result.get("speaker_sequence") == stream_result.get("speaker_sequence")
        )
        report["cases"][request_type] = {
            "user_message": user_message,
            "sync": sync_result,
            "stream": stream_result,
            "sync_stream_equivalent": equivalent,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"HTTP 계약 소규모 회귀 리포트: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
