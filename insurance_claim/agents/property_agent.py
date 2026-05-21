"""
Property claim agent: evaluates property inspection, weather data, and contractor estimates.
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

_SYSTEM_PROMPT = """You are a property insurance claim evaluator.

Evaluate the claim against these three checks and score each one from 0 to 10:
- inspection:   Is there evidence of a property inspection or damage assessment report?
- weather:      Does the reported weather event (storm, flood, fire, etc.) match the claimed date and location?
- contractor:   Are contractor repair estimates present and do they align plausibly with the reported damage?

Then give an overall score (0–10) that reflects all three checks together.

Scoring guide:
- 10 = strong, clear evidence
- 5  = partial or unclear evidence
- 0  = no evidence or serious red flag

Respond ONLY with valid JSON in exactly this format:
{
  "inspection_score": <number>,
  "weather_score": <number>,
  "contractor_score": <number>,
  "overall_score": <number>,
  "reasoning": "<one sentence covering all three checks>"
}"""

# ── LLM response model ────────────────────────────────────────────────────────


class _PropertyClaimScores(BaseModel):
    """Validated LLM response — all scores enforced to 0–10 by Pydantic."""

    inspection_score: float = Field(..., ge=0, le=10)
    weather_score: float = Field(..., ge=0, le=10)
    contractor_score: float = Field(..., ge=0, le=10)
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


def process_property_claim(state: ClaimState) -> dict:
    """LangGraph node: score the property claim across 3 checks and write decision to state."""
    claim_id = state["input"]["claim_id"]
    claim_text = state["input"]["claim_text"]

    logger.info("Property agent received claim %s", claim_id)

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
        scores = _PropertyClaimScores.model_validate_json(raw)

        logger.info(
            "Property scores — inspection=%.1f  weather=%.1f  contractor=%.1f  overall=%.1f",
            scores.inspection_score,
            scores.weather_score,
            scores.contractor_score,
            scores.overall_score,
        )

        decision = _map_score_to_decision(scores.overall_score)
        logger.info(
            "Property agent decision for claim %s: %s (score=%.1f)",
            claim_id,
            decision,
            scores.overall_score,
        )

        checks: Dict[str, str] = {
            "inspection": str(scores.inspection_score),
            "weather": str(scores.weather_score),
            "contractor": str(scores.contractor_score),
            "overall": str(scores.overall_score),
        }

        return {
            "processing": ProcessingSection(
                agent_name="property_agent",
                checks_performed=checks,
                score=scores.overall_score,
                decision=decision,
                decision_reasoning=scores.reasoning,
                agent_status=AgentStatus.SUCCESS.value,
                error_message=None,
            ),
        }

    except Exception as exc:
        logger.error("Property agent failed for claim %s: %s", claim_id, exc)
        return {
            "processing": ProcessingSection(
                agent_name="property_agent",
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
