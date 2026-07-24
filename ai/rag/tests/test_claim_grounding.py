"""
Claim Grounding Unit Tests
==============================
ground_claims()가 "검색된 청크를 프롬프트에 넣었다"는 사실만으로 근거를 활용했다고
판단하지 않고, 주장(claim) 단위로 실제 연결/관련성 검증을 하는지 확인한다. LLM/Chroma에
의존하지 않는 순수 단위 테스트다.
"""

from ai.rag.evidence_linking.claim_grounding import ground_claims, has_hard_grounding_failure

EVIDENCE = [
    {
        "chunk_id": "C1",
        "document_id": "DOC-1",
        "document_name": "WSCE2026 공고문",
        "section": "평가 기준",
        "page": 4,
        "text": "본 사업은 실현 가능성과 경제성을 중점적으로 평가한다.",
    },
    {
        "chunk_id": "C2",
        "document_id": "DOC-1",
        "document_name": "WSCE2026 공고문",
        "section": "신청 자격",
        "page": 2,
        "text": "신청 자격은 만 19세 이상 대한민국 국민으로 한다.",
    },
]


def test_document_fact_with_correct_evidence_ref_is_supported():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C1"],
        }
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["supported_claim_count"] == 1
    assert result["unsupported_claim_count"] == 0
    assert result["linked_evidence_refs"] == ["C1"]
    assert result["evidence_status"] == "grounded"
    assert result["allow_definitive_judgment"] is True


def test_unknown_chunk_id_is_unsupported_and_excluded_from_linked():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C-does-not-exist"],
        }
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["unsupported_claim_count"] == 1
    assert result["linked_evidence_refs"] == []
    assert result["unsupported_claims"][0]["reason"] == "unknown_chunk_id"
    assert result["evidence_status"] == "ungrounded"
    assert has_hard_grounding_failure(result) is True


def test_document_fact_without_evidence_refs_fails_validation():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": [],
        }
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["unsupported_claim_count"] == 1
    assert result["unsupported_claims"][0]["reason"] == "document_fact_missing_evidence"
    assert result["missing_information"]
    assert has_hard_grounding_failure(result) is True


def test_expert_judgment_allowed_without_evidence_and_not_marked_as_document_fact():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "교통 API의 호출 제한을 추가로 확인해야 한다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        }
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["supported_claim_count"] == 1
    assert result["unsupported_claim_count"] == 0
    # expert_judgment는 document_fact가 아니므로 hard failure(재생성 대상)로 잡히지 않는다.
    assert has_hard_grounding_failure(result) is False


def test_irrelevant_chunk_citation_is_filtered_by_relevance():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "반려동물 산책 경로 추천 기능이 필요하다.",
            "claim_type": "document_fact",
            # C1은 실현 가능성/경제성 내용이라 반려동물 산책 경로 주장과 무관하다.
            "evidence_refs": ["C1"],
        }
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["unsupported_claim_count"] == 1
    assert result["unsupported_claims"][0]["reason"] in {
        "evidence_not_relevant",
        "insufficient_claim_coverage",
    }


def test_general_criteria_question_cannot_prove_specific_product_effect():
    evidence = [
        {
            "chunk_id": "CRITERIA-1",
            "document_role": "criteria",
            "document_name": "WSCE2026 공고문",
            "section": "평가 기준",
            "text": "AI 기술을 활용한 도시 운영 혁신이 이루어졌는가?",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "AI 경고 시스템은 도시 범죄를 예측하고 시민 안전을 향상시킨다.",
            "claim_type": "document_fact",
            "evidence_refs": ["CRITERIA-1"],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["evidence_status"] == "ungrounded"
    assert result["unsupported_claims"][0]["reason"] == "criteria_scope_overreach"


def test_criteria_chunk_can_support_fact_about_the_evaluation_criterion():
    evidence = [
        {
            "chunk_id": "CRITERIA-1",
            "document_role": "criteria",
            "document_name": "WSCE2026 공고문",
            "section": "평가 기준",
            "text": "AI 기술을 활용한 도시 운영 혁신이 이루어졌는가?",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "AI 기술을 활용한 도시 운영 혁신 여부가 평가 항목에 포함된다.",
            "claim_type": "document_fact",
            "evidence_refs": ["CRITERIA-1"],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["evidence_status"] == "grounded"


def test_exact_criteria_question_quote_is_not_mistaken_for_product_capability():
    evidence = [
        {
            "chunk_id": "CRITERIA-1",
            "document_role": "criteria",
            "text": "AI 기술이 실제 도시 문제 해결 및 다양한 서비스에 적용 가능한가?",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "AI 기술이 실제 도시 문제 해결 및 다양한 서비스에 적용 가능한가?",
            "claim_type": "document_fact",
            "evidence_refs": ["CRITERIA-1"],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["evidence_status"] == "grounded"


def test_document_role_and_claim_type_must_match():
    evidence = [
        {
            "chunk_id": "TARGET-1",
            "document_role": "target",
            "text": "사용자는 IoT 센서로 대기질 데이터를 수집하는 아이디어를 제안했다.",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "사용자는 IoT 센서로 대기질 데이터를 수집하는 아이디어를 제안했다.",
            "claim_type": "document_fact",
            "evidence_refs": ["TARGET-1"],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["unsupported_claims"][0]["reason"] == "claim_type_document_role_mismatch"


def test_expert_judgment_cannot_be_counted_as_document_grounded():
    evidence = [
        {
            "chunk_id": "CRITERIA-1",
            "document_role": "criteria",
            "text": "혁신성과 실현 가능성을 평가한다.",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "혁신성을 높이려면 실시간 분석 기능을 추가하는 편이 좋다.",
            "claim_type": "expert_judgment",
            "evidence_refs": ["CRITERIA-1"],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["linked_evidence_count"] == 0
    assert result["unsupported_claims"][0]["reason"] == "expert_judgment_cannot_be_document_grounded"


def test_factual_claim_requires_coverage_beyond_one_shared_keyword():
    evidence = [
        {
            "chunk_id": "CRITERIA-1",
            "document_role": "criteria",
            "text": "사업의 실현 가능성을 평가한다.",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "사업은 전국 지자체의 실시간 교통 데이터를 자동 수집한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["CRITERIA-1"],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["unsupported_claims"][0]["reason"] in {
        "criteria_scope_overreach",
        "insufficient_claim_coverage",
    }


def test_numbers_not_in_document_are_unsupported():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "전체 서비스 구축에는 6개월이 소요된다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C1"],
        }
    ]
    result = ground_claims(claims, EVIDENCE)
    # C1에는 "6개월" 관련 내용이 없으므로 수치 정합성 검사에서 걸러진다.
    assert result["unsupported_claim_count"] == 1
    assert result["unsupported_claims"][0]["reason"] == "evidence_missing_numeric_detail"
    assert result["evidence_status"] == "ungrounded"


def test_year_in_document_name_is_not_treated_as_missing_product_metric():
    evidence = [
        {
            "ref": "E1",
            "chunk_id": "C1",
            "document_role": "criteria",
            "document_name": "WSCE2026_어워즈_공고문.pdf",
            "quote": "사회적 가치성은 평가 항목이다.",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "사회적 가치성은 WSCE 2026 Awards의 평가 항목이다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        }
    ]

    result = ground_claims(claims, evidence)

    assert result["linked_evidence_count"] == 1
    assert result["unsupported_claim_count"] == 0


def test_partially_grounded_when_some_claims_supported_and_some_not():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C1"],
        },
        {
            "claim_id": "claim_2",
            "text": "예상 구축 비용은 5천만 원이다.",
            "claim_type": "document_fact",
            "evidence_refs": [],
        },
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["evidence_status"] == "partially_grounded"
    assert result["supported_claim_count"] == 1
    assert result["unsupported_claim_count"] == 1


def test_all_claims_grounded_status_is_grounded():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C1"],
        },
        {
            "claim_id": "claim_2",
            "text": "신청 자격은 만 19세 이상이다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C2"],
        },
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["evidence_status"] == "grounded"
    assert result["unsupported_claim_count"] == 0
    assert set(result["linked_evidence_refs"]) == {"C1", "C2"}


def test_no_claims_at_all_is_no_evidence_available():
    result = ground_claims([], EVIDENCE)
    assert result["evidence_status"] == "no_evidence_available"
    assert result["allow_definitive_judgment"] is False
    assert has_hard_grounding_failure(result) is False


def test_meeting_does_not_stop_when_ungrounded():
    """근거가 전혀 없어도 ground_claims 자체는 예외를 던지거나 회의를 중단시키지 않는다 —
    호출부(ideation_conv_nodes)가 fallback 문구로 이어갈 수 있도록 항상 결과를 반환한다."""
    claims = [
        {
            "claim_id": "claim_1",
            "text": "이 서비스는 반드시 성공한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C-unknown"],
        }
    ]
    result = ground_claims(claims, [])
    assert result["evidence_status"] in ("ungrounded", "no_evidence_available")
    assert result["allow_definitive_judgment"] is False


def test_user_provided_fact_without_evidence_is_supported():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "사용자는 이미 실시간 교통 데이터를 활용하겠다고 밝혔다.",
            "claim_type": "user_provided_fact",
            "evidence_refs": [],
        }
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["supported_claim_count"] == 1
    assert result["unsupported_claim_count"] == 0


def test_role_keywords_help_recognize_relevance():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "경제성 측면 검토가 필요하다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C1"],
        }
    ]
    result = ground_claims(claims, EVIDENCE, role_keywords=["경제성"])
    assert result["supported_claim_count"] == 1


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-22, 요청: linked_evidence_count=0인데 partially_grounded로 오판정되던
# 버그 수정) — claim 통계 의미 분리(accepted/grounded/expert_judgment/linked_evidence_count)와
# evidence_status 판정 규칙 재정의를 검증한다.
# ---------------------------------------------------------------------------


def test_expert_judgment_only_two_claims_is_not_partially_grounded():
    """전문가 판단만 2개 있고 문서 근거가 하나도 연결되지 않았다면(linked_evidence_count=0)
    partially_grounded/grounded로 오판정되면 안 되고 expert_judgment_only여야 한다."""
    claims = [
        {
            "claim_id": "claim_1",
            "text": "데이터 접근성이 중요하다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        },
        {
            "claim_id": "claim_2",
            "text": "인프라 구축 계획이 필요하다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        },
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["accepted_claim_count"] == 2
    assert result["grounded_claim_count"] == 0
    assert result["expert_judgment_count"] == 2
    assert result["linked_evidence_count"] == 0
    assert result["evidence_status"] == "expert_judgment_only"
    assert result["evidence_status"] != "partially_grounded"
    assert result["allow_definitive_judgment"] is False


def test_document_fact_linked_plus_expert_judgment_is_partially_grounded():
    """document_fact 1개가 실제 chunk에 연결되고 expert_judgment 1개가 함께 있으면
    partially_grounded이고, grounded/expert_judgment/linked 값이 정확히 분리된다."""
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C1"],
        },
        {
            "claim_id": "claim_2",
            "text": "데이터 제공기관과의 협력 방안을 구체화해야 한다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        },
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["accepted_claim_count"] == 2
    assert result["grounded_claim_count"] == 1
    assert result["expert_judgment_count"] == 1
    assert result["linked_evidence_count"] == 1
    assert result["evidence_status"] == "partially_grounded"


def test_all_document_facts_linked_status_is_grounded_with_new_fields():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C1"],
        },
        {
            "claim_id": "claim_2",
            "text": "신청 자격은 만 19세 이상이다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C2"],
        },
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["evidence_status"] == "grounded"
    assert result["grounded_claim_count"] == 2
    assert result["expert_judgment_count"] == 0
    assert result["linked_evidence_count"] == 2
    assert result["unsupported_claim_count"] == 0


def test_document_fact_present_but_link_fails_is_ungrounded_with_unsupported_count():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C-does-not-exist"],
        }
    ]
    result = ground_claims(claims, EVIDENCE)
    assert result["evidence_status"] == "ungrounded"
    assert result["unsupported_claim_count"] == 1
    assert result["linked_evidence_count"] == 0
    assert result["grounded_claim_count"] == 0


def test_document_fact_linked_to_criteria_chunk():
    """criteria 문서 사실 grounding — criteria chunk에 연결된다(요청: 선택된 아이디어/사용자
    답변을 target evidence로 인덱싱하는 작업의 claim grounding 회귀 검증, 항목 A)."""
    evidence = [
        {
            "chunk_id": "C1",
            "document_id": "DOC-NOTICE",
            "document_name": "WSCE2026 공고문",
            "section": "평가 기준",
            "document_role": "criteria",
            "text": "WSCE는 기술의 완성도와 경제성을 실현 가능성 항목에서 평가한다.",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 기술의 완성도와 경제성을 실현 가능성 항목에서 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["C1"],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["linked_evidence_refs"] == ["C1"]
    assert result["evidence_status"] == "grounded"


def test_candidate_target_fact_linked_to_target_chunk():
    """candidate target 사실 grounding — 선택된 후보의 target chunk에 연결된다(항목 B)."""
    evidence = [
        {
            "chunk_id": "T1",
            "document_id": "ideation-target::P1::S1::candidate_1",
            "document_name": "[선택한 아이디어] 스마트 에너지 관리 플랫폼",
            "document_role": "target",
            "ideation_source_type": "ideation_candidate",
            "session_id": "S1",
            "text": "현재 아이디어는 스마트 미터링 데이터를 사용해 실시간 에너지 사용량을 분석한다.",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "현재 아이디어는 스마트 미터링 데이터를 사용해 실시간 에너지 사용량을 분석한다.",
            "claim_type": "user_provided_fact",
            "evidence_refs": ["T1"],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["linked_evidence_refs"] == ["T1"]
    assert result["grounded_claim_count"] == 1


def test_user_answer_target_fact_linked_to_user_answer_chunk():
    """사용자 답변 사실 grounding — user answer target chunk에 연결된다(항목 C)."""
    evidence = [
        {
            "chunk_id": "A1",
            "document_id": "ideation-answer::P1::S1::MSG-1",
            "document_name": "[사용자 추가 답변] 회의 답변",
            "document_role": "target",
            "ideation_source_type": "user_session_answer",
            "session_id": "S1",
            "text": "한국전력 API와 아파트 스마트 미터 데이터를 사용하고 5분 단위로 분석하려고 합니다.",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "사용자는 한국전력 API와 아파트 스마트 미터 데이터를 5분 단위로 분석하려 한다.",
            "claim_type": "user_provided_fact",
            "evidence_refs": ["A1"],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["linked_evidence_refs"] == ["A1"]


def test_technical_fact_not_in_target_is_expert_judgment_not_fabricated_link():
    """target에는 없는 기술 사실(예: 데이터 전송 지연 처리 방식) — expert_judgment로만
    표현되고, 존재하지 않는 chunk_id로 근거가 있는 것처럼 표시되지 않는다(항목 D)."""
    evidence = [
        {
            "chunk_id": "T1",
            "document_id": "ideation-target::P1::S1::candidate_1",
            "document_role": "target",
            "ideation_source_type": "ideation_candidate",
            "text": "스마트 미터링 데이터를 수집해 AI로 소비 패턴을 분석한다.",
        }
    ]
    claims = [
        {
            "claim_id": "claim_1",
            "text": "데이터 전송 지연과 실시간 처리 문제가 있을 수 있다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        }
    ]
    result = ground_claims(claims, evidence)
    assert result["linked_evidence_refs"] == []
    assert result["unsupported_claim_count"] == 0
    assert result["evidence_status"] == "expert_judgment_only"


def test_retrieved_evidence_empty_is_no_evidence_available_even_with_accepted_claims():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "데이터 접근성이 중요하다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        }
    ]
    result = ground_claims(claims, [])
    assert result["evidence_status"] == "no_evidence_available"


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-23, 요청: RAG 근거 실제 활용 강화 — evidence 참조 안정화) — chunk_id는
# 20자 안팎의 해시라 LLM이 그대로 베껴 쓰다 실수하기 쉽다. call_evidence_lookup(ai/meeting/
# graph/ideation_nodes.py)이 각 근거에 짧은 순번 참조("ref": "E1")를 부여하면, LLM은 그
# 값으로만 evidence_refs를 채우면 되고 ground_claims는 ref로 조회하되 linked_evidence_refs에는
# 항상 실제 chunk_id를 담아야 한다(frontend가 message.evidence[].chunk_id와 대조하는 기존
# 계약을 그대로 지키기 위함).
# ---------------------------------------------------------------------------

_EVIDENCE_WITH_REF = [
    {
        "ref": "E1",
        "chunk_id": "chk_9f8e7d6c5b4a3210",
        "document_id": "DOC-1",
        "document_name": "WSCE2026 공고문",
        "section": "평가 기준",
        "text": "본 사업은 실현 가능성과 경제성을 중점적으로 평가한다.",
    }
]


def test_claim_citing_ref_resolves_to_actual_chunk_id():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        }
    ]
    result = ground_claims(claims, _EVIDENCE_WITH_REF)
    assert result["evidence_status"] == "grounded"
    assert result["linked_evidence_refs"] == ["chk_9f8e7d6c5b4a3210"]


def test_claim_citing_raw_chunk_id_when_ref_present_is_unknown():
    """ref가 있는 항목은 ref로만 조회된다 — chunk_id를 직접 인용하면 "존재하지 않는 참조"로
    취급된다(LLM이 두 값을 섞어 쓰지 않도록 하나의 정답만 허용)."""
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["chk_9f8e7d6c5b4a3210"],
        }
    ]
    result = ground_claims(claims, _EVIDENCE_WITH_REF)
    assert result["evidence_status"] == "ungrounded"
    assert result["unsupported_claims"][0]["reason"] == "unknown_chunk_id"


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-23, 요청: IDEATION_EVIDENCE_LINKED 로그 매핑 수정) — claim["evidence_refs"]
# 는 LLM이 인용한 ref("E1")를 담고, linked_evidence_refs(전역)는 실제 chunk_id만 담아 서로
# 다른 값 공간이라 "ref in linked_evidence_refs" 같은 직접 비교는 항상 실패한다. 새로 노출한
# claim_evidence_links가 claim 단위로 (ref, chunk_id) 쌍을 명시적으로 짝지어 로그가 올바른
# chunk_id를 남길 수 있게 한다.
# ---------------------------------------------------------------------------


def test_claim_evidence_links_pairs_ref_with_actual_chunk_id():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        }
    ]
    result = ground_claims(claims, _EVIDENCE_WITH_REF)
    assert result["claim_evidence_links"] == [
        {
            "claim_id": "claim_1",
            "evidence_refs": ["E1"],
            "chunk_ids": ["chk_9f8e7d6c5b4a3210"],
        }
    ]


def test_claim_evidence_links_excludes_claims_with_unknown_ref():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E-does-not-exist"],
        }
    ]
    result = ground_claims(claims, _EVIDENCE_WITH_REF)
    assert result["claim_evidence_links"] == []


def test_claim_evidence_links_excludes_expert_judgment_without_refs():
    claims = [
        {
            "claim_id": "claim_1",
            "text": "데이터 접근성이 중요하다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        }
    ]
    result = ground_claims(claims, _EVIDENCE_WITH_REF)
    assert result["claim_evidence_links"] == []


def test_multiple_claims_citing_same_evidence_dedupe_in_linked_evidence_refs():
    """여러 claim이 같은 근거를 인용해도 전역 linked_evidence_refs는 chunk_id 기준으로
    중복 제거되지만, claim_evidence_links에는 claim별로 각각 남는다."""
    claims = [
        {
            "claim_id": "claim_1",
            "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        },
        {
            "claim_id": "claim_2",
            "text": "실현 가능성 평가 항목이 있다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        },
    ]
    result = ground_claims(claims, _EVIDENCE_WITH_REF)
    assert result["linked_evidence_refs"] == ["chk_9f8e7d6c5b4a3210"]
    assert len(result["claim_evidence_links"]) == 2
    assert {link["claim_id"] for link in result["claim_evidence_links"]} == {"claim_1", "claim_2"}
    for link in result["claim_evidence_links"]:
        assert link["chunk_ids"] == ["chk_9f8e7d6c5b4a3210"]
