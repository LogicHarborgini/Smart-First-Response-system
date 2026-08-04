
"""
Pydantic models for the SFR API.

These define the data contracts:
- SFRRequest: what the API expects as input
- SFRResponse: what the API returns
- HealthResponse: for the /health endpoint

FastAPI uses these for automatic validation and OpenAPI docs generation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TicketPriority(str, Enum):
    """Ticket priority levels — only these values are accepted."""
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class SFRRequest(BaseModel):
    """Request body for POST /api/v1/generate-response."""
    
    ticket_id: str = Field(..., description="Unique ticket identifier", min_length=1)
    content: str = Field(..., description="Full ticket content", min_length=10)
    priority: TicketPriority = Field(default=TicketPriority.P2)
    customer_name: Optional[str] = Field(default=None)

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
                "content": "Production database connection timing out since 14:30 UTC. All services affected.",
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


class HealthResponse(BaseModel):
    """Response body from GET /health."""
    status: str = "healthy"
    service: str = "SFR"
    version: str
