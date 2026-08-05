"""
Tests for the SFR API endpoints.

conftest.py forces LLM_PROVIDER=fake and disables tracing, so these run offline:
no AWS credentials, no local model, no LangSmith calls.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_TICKET = {
    "ticket_id": "tick-001",
    "content": "AS2 messages failing since 09:15. Partner reports MDN timeouts.",
    "priority": "P1",
    "customer_name": "Northwind Logistics",
}


@pytest.fixture(scope="module")
def client():
    # TestClient as a context manager runs the lifespan handler, which is where
    # the chain is pre-warmed. Instantiating it without `with` skips startup and
    # would leave that path untested.
    with TestClient(app) as c:
        yield c


def test_health_reports_healthy(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "SFR"
    assert body["version"]


def test_generate_response_returns_full_payload(client):
    response = client.post("/api/v1/generate-response", json=VALID_TICKET)

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "TICK-001"        # normalised by the validator
    assert body["first_response"].strip()
    assert body["status"] == "success"
    assert body["latency_ms"] > 0


def test_model_used_reports_the_active_provider(client):
    """
    Guards against the response attributing output to the wrong model — the
    original code returned settings.bedrock_model_id unconditionally, which
    would label a fake or Ollama run as Claude 3 Sonnet.
    """
    response = client.post("/api/v1/generate-response", json=VALID_TICKET)

    assert response.json()["model_used"] == "fake:canned-responses"


def test_run_id_is_null_when_tracing_disabled(client):
    """No trace exists, so there must be no run ID pointing at one."""
    response = client.post("/api/v1/generate-response", json=VALID_TICKET)

    assert response.json()["langsmith_run_id"] is None


def test_whitespace_only_content_is_rejected(client):
    response = client.post("/api/v1/generate-response", json={
        "ticket_id": "tick-002",
        "content": "               ",
    })

    assert response.status_code == 422


def test_content_below_minimum_length_is_rejected(client):
    response = client.post("/api/v1/generate-response", json={
        "ticket_id": "tick-003",
        "content": "short",
    })

    assert response.status_code == 422
    assert "too short" in response.text.lower()


def test_invalid_priority_is_rejected(client):
    response = client.post("/api/v1/generate-response", json={
        **VALID_TICKET,
        "priority": "CRITICAL",
    })

    assert response.status_code == 422


def test_customer_name_is_optional(client):
    payload = {k: v for k, v in VALID_TICKET.items() if k != "customer_name"}

    response = client.post("/api/v1/generate-response", json=payload)

    assert response.status_code == 200


def test_messy_content_is_preprocessed_not_rejected(client):
    """Hard-wrapped, whitespace-heavy ticket bodies are normal input."""
    response = client.post("/api/v1/generate-response", json={
        "ticket_id": "tick-004",
        "content": "URGENT\n\n\n     Production   outage.\n\n  All AS2 failing.   ",
        "priority": "P1",
    })

    assert response.status_code == 200
    assert response.json()["first_response"].strip()
