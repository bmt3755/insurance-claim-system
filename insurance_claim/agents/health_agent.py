"""
Health claim agent: evaluates medical code validation, treatment verification, and provider check.
Returns a score and maps it to APPROVE / DENY / NEEDS_REVIEW.

Environment variables required:
    OPENAI_API_KEY         — OpenAI credentials
    LANGCHAIN_TRACING_V2   — set to "true" to enable LangSmith tracing
    LANGCHAIN_API_KEY      — LangSmith credentials
"""

import logging
from typing import Dict

from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel, Field

from insurance_claim.state.schema import (
    AgentStatus,
    ClaimState,
    Decision,
    ProcessingSection,
    TrackingSection,
)

logger = logging.getLogger(__name__)

_client = wrap_openai(OpenAI())

# ── Score thresholds ──────────────────────────────────────────────────────────

_APPROVE_THRESHOLD: float = 7.0   # score > 7  → APPROVE
_DENY_THRESHOLD: float = 5.0      # score < 5  → DENY
                                   # score 5–7 (inclusive) → NEEDS_REVIEW

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a health insurance claim evaluator.

Evaluate the claim against these three checks and score each one from 0 to 10:
- medical_code:  Are diagnosis or treatment codes (e.g. ICD, CPT) present and plausible?
- treatment:     Is the described treatment consistent, medically appropriate, and verifiable?
- provider:      Is the healthcare provider clearly identified and appear legitimate?

Then give an overall score (0–10) that reflects all three checks together.

Scoring guide:
- 10 = strong, clear evidence
- 5  = partial or unclear evidence
- 0  = no evidence or serious red flag

Respond ONLY with valid JSON in exactly this format:
{
  "medical_code_score": <number>,
  "treatment_score": <number>,
  "provider_score": <number>,
  "overall_score": <number>,
  "reasoning": "<one sentence covering all three checks>"
}"""

# ── LLM response model ────────────────────────────────────────────────────────


class _HealthClaimScores(BaseModel):
    """Validated LLM response — all scores enforced to 0–10 by Pydantic."""

    medical_code_score: float = Field(..., ge=0, le=10)
    treatment_score: float = Field(..., ge=0, le=10)
    provider_score: float = Field(..., ge=0, le=10)
    overall_score: float = Field(..., ge=0, le=10)
    reasoning: str


# ── Decision mapping ──────────────────────────────────────────────────────────


def _map_score_to_decision(score: float) -> str:
    """Map overall score to a decision. Boundary values go to NEEDS_REVIEW (safer for compliance)."""
    if score > _APPROVE_THRESHOLD:
        return Decision.APPROVE.value
    if score < _DENY_THRESHOLD:
        return Decision.DENY.value
    return Decision.NEEDS_REVIEW.value


# ── LangGraph node ────────────────────────────────────────────────────────────


def process_health_claim(state: ClaimState) -> dict:
    """LangGraph node: score the health claim across 3 checks and write decision to state."""
    claim_id = state["input"]["claim_id"]
    claim_text = state["input"]["claim_text"]

    logger.info("Health agent received claim %s", claim_id)

    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Claim:\n{claim_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw = response.choices[0].message.content
        scores = _HealthClaimScores.model_validate_json(raw)

        logger.info(
            "Health scores — medical_code=%.1f  treatment=%.1f  provider=%.1f  overall=%.1f",
            scores.medical_code_score,
            scores.treatment_score,
            scores.provider_score,
            scores.overall_score,
        )

        decision = _map_score_to_decision(scores.overall_score)
        logger.info(
            "Health agent decision for claim %s: %s (score=%.1f)",
            claim_id,
            decision,
            scores.overall_score,
        )

        checks: Dict[str, str] = {
            "medical_code": str(scores.medical_code_score),
            "treatment": str(scores.treatment_score),
            "provider": str(scores.provider_score),
            "overall": str(scores.overall_score),
        }

        return {
            "processing": ProcessingSection(
                agent_name="health_agent",
                checks_performed=checks,
                score=scores.overall_score,
                decision=decision,
                decision_reasoning=scores.reasoning,
                agent_status=AgentStatus.SUCCESS.value,
                error_message=None,
            ),
        }

    except Exception as exc:
        logger.error("Health agent failed for claim %s: %s", claim_id, exc)
        return {
            "processing": ProcessingSection(
                agent_name="health_agent",
                checks_performed=None,
                score=None,
                decision=Decision.NEEDS_REVIEW.value,  # safe fallback — human reviews
                decision_reasoning=None,
                agent_status=AgentStatus.FAILED.value,
                error_message=str(exc),
            ),
            "tracking": TrackingSection(
                started_at=state["tracking"]["started_at"],
                completed_at=state["tracking"]["completed_at"],
                errors_encountered=state["tracking"]["errors_encountered"] + 1,
            ),
        }
