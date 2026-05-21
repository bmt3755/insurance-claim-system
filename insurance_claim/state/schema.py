from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, field_validator
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Enums — locked to exactly the values the system allows
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    AUTO = "auto"
    HEALTH = "health"
    PROPERTY = "property"
    TRAVEL = "travel"


class Decision(str, Enum):
    APPROVE = "APPROVE"
    DENY = "DENY"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class AgentStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# State sections — each agent owns exactly one section
# ---------------------------------------------------------------------------

class InputSection(TypedDict):
    claim_id: str
    claim_text: str
    submitted_at: str           # ISO-8601 UTC string


class RoutingSection(TypedDict):
    claim_type: Optional[str]       # ClaimType value, or None until router runs
    router_reasoning: Optional[str]
    router_status: str              # AgentStatus value


class ProcessingSection(TypedDict):
    agent_name: Optional[str]
    checks_performed: Optional[Dict[str, str]]  # e.g. {"police_report": "found"}
    score: Optional[float]          # 0–10; None until specialist runs
    decision: Optional[str]         # Decision value, or None until specialist runs
    decision_reasoning: Optional[str]
    agent_status: str               # AgentStatus value
    error_message: Optional[str]


class TrackingSection(TypedDict):
    started_at: str                 # ISO-8601 UTC string
    completed_at: Optional[str]
    errors_encountered: int         # incremented each time a section status = FAILED


# ---------------------------------------------------------------------------
# Main LangGraph state — the single object passed between all nodes
# ---------------------------------------------------------------------------

class ClaimState(TypedDict):
    input: InputSection
    routing: RoutingSection
    processing: ProcessingSection
    tracking: TrackingSection


# ---------------------------------------------------------------------------
# Input model — validates raw claim before it enters the graph
# ---------------------------------------------------------------------------

class ClaimInput(BaseModel):
    claim_id: Optional[str] = None
    claim_text: str

    @field_validator("claim_text")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("claim_text cannot be blank")
        return v.strip()

    def to_initial_state(self) -> ClaimState:
        claim_id = self.claim_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        return ClaimState(
            input=InputSection(
                claim_id=claim_id,
                claim_text=self.claim_text,
                submitted_at=now,
            ),
            routing=RoutingSection(
                claim_type=None,
                router_reasoning=None,
                router_status=AgentStatus.PENDING.value,
            ),
            processing=ProcessingSection(
                agent_name=None,
                checks_performed=None,
                score=None,
                decision=None,
                decision_reasoning=None,
                agent_status=AgentStatus.PENDING.value,
                error_message=None,
            ),
            tracking=TrackingSection(
                started_at=now,
                completed_at=None,
                errors_encountered=0,
            ),
        )
