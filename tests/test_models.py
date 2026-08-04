"""
Tests for Pydantic models.

Run with: pytest tests/
"""

import pytest
from pydantic import ValidationError

from app.models import SFRRequest, TicketPriority


def test_valid_request():
    req = SFRRequest(
        ticket_id="tick-001",
        content="Production database connection is failing since 14:30 UTC.",
        priority=TicketPriority.P1,
    )
    assert req.ticket_id == "TICK-001"   # normalised to uppercase
    assert req.priority == TicketPriority.P1


def test_ticket_id_normalised_to_uppercase():
    req = SFRRequest(ticket_id="tick-abc", content="valid content here")
    assert req.ticket_id == "TICK-ABC"


def test_content_too_short_raises():
    with pytest.raises(ValidationError) as exc_info:
        SFRRequest(ticket_id="tick-001", content="short")
    assert "too short" in str(exc_info.value).lower()


def test_invalid_priority_raises():
    with pytest.raises(ValidationError):
        SFRRequest(
            ticket_id="tick-001",
            content="valid content here for testing",
            priority="CRITICAL"   # invalid
        )


def test_default_priority_is_p2():
    req = SFRRequest(ticket_id="tick-001", content="valid content here")
    assert req.priority == TicketPriority.P2


def test_optional_customer_name():
    req = SFRRequest(ticket_id="tick-001", content="valid content here")
    assert req.customer_name is None   # optional field defaults to None