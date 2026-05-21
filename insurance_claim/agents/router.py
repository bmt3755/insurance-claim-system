"""
Router agent: scores the claim across all 4 types and routes to the right specialist.

Environment variables required:
    OPENAI_API_KEY         — OpenAI credentials
    LANGCHAIN_TRACING_V2   — set to "true" to enable LangSmith tracing
    LANGCHAIN_API_KEY      — LangSmith credentials
"""

import logging
from typing import Optional

from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel, Field

from insurance_claim.state.schema import (
    AgentStatus,
    ClaimState,
    ClaimType,
    RoutingSection,
    TrackingSection,
)

logger = logging.getLogger(__name__)

# Wrap client so all OpenAI calls appear as spans in LangSmith
_client = wrap_openai(OpenAI())

# ── Routing thresholds ────────────────────────────────────────────────────────

_MIN_CONFIDENCE: float = 6.0   # top score must be at least this
_AMBIGUITY_MARGIN: float = 2.0  # top score must beat second by more than this

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an insurance claim classifier.

Score how well the claim fits each of the four types on a scale of 0–10:
- auto:     vehicle damage, car accidents, police reports, repair estimates
- health:   medical bills, doctor visits, hospital treatment, medical codes
- property: home damage, property inspection, weather events, contractor repairs
- travel:   trip cancellation, flight delays, travel receipts, coverage dates

Rules:
- All scores must be between 0 and 10 (decimals allowed).
- 10 = perfect fit, 0 = no fit at all.
- Provide one sentence explaining your scores.

Respond ONLY with valid JSON in exactly this format:
{
  "auto_score": <number>,
  "health_score": <number>,
  "property_score": <number>,
  "travel_score": <number>,
  "reasoning": "<one sentence>"
}"""

# ── LLM response model ────────────────────────────────────────────────────────


class _RouterScores(BaseModel):
    """Validated LLM response — all scores clamped to 0–10 by Pydantic."""

    auto_score: float = Field(..., ge=0, le=10)
    health_score: float = Field(..., ge=0, le=10)
    property_score: float = Field(..., ge=0, le=10)
    travel_score: float = Field(..., ge=0, le=10)
    reasoning: str


# ── Routing logic ─────────────────────────────────────────────────────────────


def _pick_claim_type(scores: _RouterScores) -> Optional[str]:
    """Return the winning claim type, or None if ambiguous or low-confidence."""
    ranked = sorted(
        [
            (scores.auto_score, ClaimType.AUTO.value),
            (scores.health_score, ClaimType.HEALTH.value),
            (scores.property_score, ClaimType.PROPERTY.value),
            (scores.travel_score, ClaimType.TRAVEL.value),
        ],
        reverse=True,
    )
    top_score, top_type = ranked[0]
    second_score, _ = ranked[1]

    if top_score < _MIN_CONFIDENCE:
        logger.info(
            "Top score %.1f is below confidence threshold %.1f — NEEDS_REVIEW",
            top_score,
            _MIN_CONFIDENCE,
        )
        return None

    if (top_score - second_score) <= _AMBIGUITY_MARGIN:
        logger.info(
            "Ambiguous scores: top=%.1f  second=%.1f  margin=%.1f ≤ %.1f — NEEDS_REVIEW",
            top_score,
            second_score,
            top_score - second_score,
            _AMBIGUITY_MARGIN,
        )
        return None

    return top_type


# ── LangGraph node ────────────────────────────────────────────────────────────


def route_claim(state: ClaimState) -> dict:
    """LangGraph node: classify the incoming claim and write routing decision to state."""
    claim_id = state["input"]["claim_id"]
    claim_text = state["input"]["claim_text"]

    logger.info("Router received claim %s", claim_id)

    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Claim:\n{claim_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,  # deterministic — same claim always routes the same way
        )

        raw = response.choices[0].message.content
        scores = _RouterScores.model_validate_json(raw)

        logger.info(
            "Scores — auto=%.1f  health=%.1f  property=%.1f  travel=%.1f",
            scores.auto_score,
            scores.health_score,
            scores.property_score,
            scores.travel_score,
        )

        claim_type = _pick_claim_type(scores)
        score_summary = (
            f"auto={scores.auto_score}, health={scores.health_score}, "
            f"property={scores.property_score}, travel={scores.travel_score}"
        )

        if claim_type:
            logger.info("Routing claim %s → %s", claim_id, claim_type)
            reasoning = f"{scores.reasoning} | Scores: {score_summary}"
        else:
            logger.info("Routing claim %s → NEEDS_REVIEW (ambiguous or low confidence)", claim_id)
            reasoning = f"NEEDS_REVIEW — {scores.reasoning} | Scores: {score_summary}"

        return {
            "routing": RoutingSection(
                claim_type=claim_type,
                router_reasoning=reasoning,
                router_status=AgentStatus.SUCCESS.value,
            ),
        }

    except Exception as exc:
        logger.error("Router failed for claim %s: %s", claim_id, exc)
        return {
            "routing": RoutingSection(
                claim_type=None,
                router_reasoning=f"Router error: {exc}",
                router_status=AgentStatus.FAILED.value,
            ),
            # Increment error count; preserve the rest of tracking unchanged
            "tracking": TrackingSection(
                started_at=state["tracking"]["started_at"],
                completed_at=state["tracking"]["completed_at"],
                errors_encountered=state["tracking"]["errors_encountered"] + 1,
            ),
        }
