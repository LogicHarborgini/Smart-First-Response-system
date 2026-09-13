"""
Pydantic models for the SFR API.

These define the data contracts:
- SFRRequest: what the API expects as input
- SFRResponse: what the API returns
- HealthResponse: for the /health endpoint

FastAPI uses these for automatic validation and OpenAPI docs generation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class TicketPriority(StrEnum):
    """Ticket priority levels — only these values are accepted."""
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class TriageDecision(StrEnum):
    """
    What the triage node decided to do with a ticket.

    ESCALATE is the safe default: every failure path in the triage node resolves
    here. Sending a wrong automated reply to a customer is not recoverable in the
    way that putting a straightforward ticket in front of a human is.
    """
    AUTO_RESPOND = "auto_respond"
    NEEDS_CLARIFICATION = "needs_clarification"
    ESCALATE = "escalate"


class SFRRequest(BaseModel):
    """Request body for POST /api/v1/generate-response."""

    ticket_id: str = Field(..., description="Unique ticket identifier", min_length=1)
    # No min_length here on purpose: the content_must_be_meaningful validator
    # below strips whitespace first, so it catches "          " which min_length
    # would let through. A field constraint would pre-empt it.
    content: str = Field(..., description="Full ticket content")
    priority: TicketPriority = Field(default=TicketPriority.P2)
    customer_name: str | None = Field(default=None)
    # partner and category are optional so the plain endpoint's contract is
    # unchanged. They exist for the triage agent, which routes better when it can
    # see who raised the ticket and what area it falls in — a P1 from a partner
    # in "security" is a different decision from a P1 in "documentation".
    partner: str | None = Field(
        default=None, description="Trading partner or account the ticket came from"
    )
    category: str | None = Field(
        default=None, description="Issue area, e.g. EDI, AS2, SFTP, API, billing"
    )

    @field_validator("ticket_id")
    @classmethod
    def normalise_ticket_id(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("content")
    @classmethod
    def content_must_be_meaningful(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError(f"Content too short ({len(stripped)} chars). Min 10 required.")
        return stripped

    model_config = {
        "json_schema_extra": {
            "example": {
                "ticket_id": "tick-12345",
                "content": (
                    "Production database connection timing out since 14:30 UTC. "
                    "All services affected."
                ),
                "priority": "P1",
                "customer_name": "Acme Corp"
            }
        }
    }


class SFRResponse(BaseModel):
    """Response body from POST /api/v1/generate-response."""

    ticket_id: str
    first_response: str
    model_used: str
    latency_ms: float
    status: str = "success"
    # LangSmith trace ID for this run. None when tracing is disabled. Returning
    # it lets a ticket in your own records be matched to its trace afterwards.
    langsmith_run_id: str | None = Field(default=None)


class SFRAgentResponse(BaseModel):
    """
    Response body from POST /api/v1/generate-response/agent.

    Exactly one of first_response / clarifying_question / escalation_summary is
    populated, determined by `decision`. They are separate fields rather than one
    `output` string because a caller has to treat them differently: one gets sent
    to the customer, one gets sent back asking for more detail, and one goes to a
    human queue. Collapsing them would push that distinction onto the client.
    """

    ticket_id: str
    decision: TriageDecision
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    # The nodes that ran, in order — e.g. ["triage", "escalate"]. This is the
    # audit trail: it answers "why did this ticket get that outcome" without
    # opening LangSmith.
    node_path: list[str]

    first_response: str | None = None
    clarifying_question: str | None = None
    escalation_summary: str | None = None

    model_used: str
    latency_ms: float
    status: str = "success"
    langsmith_run_id: str | None = Field(default=None)


class HealthResponse(BaseModel):
    """Response body from GET /health."""
    status: str = "healthy"
    service: str = "SFR"
    version: str
