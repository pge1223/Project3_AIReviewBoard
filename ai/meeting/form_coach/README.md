# Form Coach v2

`form_coach_v2` is the application-form writing flow used after the existing
candidate discovery step. It does not execute the LangGraph discussion nodes.
`ideation_conv_discussion_facilitator_02.txt` is the improved facilitator prompt
for that LangGraph path; its useful role and phase rules are mirrored here, but
the prompt itself is not called by the independent form-coach API.

## Boundaries

- `service.py`: one deterministic state transition per user answer
- `phase_resolver.py`: application field to semantic phase ordering
- `prompt_builder.py`: prompt assembly only
- `response_validator.py`: JSON validation and single-field patch enforcement
- `expert_guidance.py`: optional pre-answer planning/development guidance
- `expert_review.py`: optional post-answer planning/development review
- `expert_delegation.py`: recommendation flow when the user delegates a decision
- `evidence_provider.py`: adapter over the existing ideation RAG lookup
- `backend/app/api/routes/ideation_form_coach.py`: OpenAI and in-memory sessions

The frontend keeps using the legacy flow until a candidate is selected. With
`VITE_IDEATION_FLOW=form_coach_v2`, it then starts `/ideation-form-coach`.

The visible turn order is:

1. optional post-answer expert review
2. facilitator reflection and next question
3. optional pre-answer expert guidance
4. choices or direct user input

## Rollback

Set `VITE_IDEATION_FLOW=legacy_graph` or remove the variable. The frontend will
stay on `/ideation-conversation`, and no form-coach state is created.

Set `ENABLE_FORM_COACH_V2=false` in `backend/.env` to remove only the new API
routes. The legacy routes and prompt files are untouched.
