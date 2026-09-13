"""
Tests for the LangGraph triage agent.

Two levels, because they catch different things:

- Node tests call a node directly with a state dict and assert on the dict it
  returns. This is where the triage failure modes live — a node is a plain async
  function, so a malformed model reply can be reproduced exactly.
- Graph tests run the compiled graph end to end and assert on node_path. That is
  the only way to prove the routing actually happened: a triage decision of
  "escalate" alongside a populated response field would pass every node test and
  still be a broken agent.

No model is ever built. Each test replaces the cached chain builder, so nothing
here needs credentials, a network, or a local Ollama.
"""

import pytest
from langchain_core.runnables import RunnableLambda

from app.agent.graph import build_sfr_agent, route_after_triage
from app.agent.nodes import (
    _parse_triage_verdict,
    clarify_node,
    escalate_node,
    respond_node,
    triage_node,
)
from app.models import TriageDecision


def make_state(**overrides) -> dict:
    """A minimal state, as it looks on entry to the graph."""
    state = {
        "ticket_id": "TEST-001",
        "content": "AS2 messages failing since 09:15. Partner reports MDN timeouts.",
        "priority": "P1",
        "customer_name": "Northwind Logistics",
        "partner": "Northwind Logistics",
        "category": "AS2",
        "node_path": [],
    }
    state.update(overrides)
    return state


def stub_chain(monkeypatch, target: str, reply: str) -> list[dict]:
    """
    Replace one node's chain with a runnable returning `reply`.

    Returns the list the stub records its inputs into, so a test can assert on
    what the node actually sent to the model.
    """
    seen: list[dict] = []

    def _record(payload):
        seen.append(payload)
        return reply

    monkeypatch.setattr(f"app.agent.nodes.{target}", lambda: RunnableLambda(_record))
    return seen


# ─────────────────────────────────────────────────────────────────────────────
# Verdict parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_valid_verdict_is_parsed():
    decision, confidence, reasoning = _parse_triage_verdict(
        '{"decision": "auto_respond", "confidence": 0.9, "reasoning": "clear issue"}'
    )

    assert decision == TriageDecision.AUTO_RESPOND
    assert confidence == 0.9
    assert reasoning == "clear issue"


def test_verdict_wrapped_in_prose_and_fences_is_still_parsed():
    """Models wrap JSON in explanations often enough that a bare loads() is not enough."""
    decision, _, _ = _parse_triage_verdict(
        'Here is my assessment:\n```json\n{"decision": "escalate", '
        '"confidence": 0.95, "reasoning": "possible breach"}\n```\nHope that helps.'
    )

    assert decision == TriageDecision.ESCALATE


@pytest.mark.parametrize(
    "reply",
    [
        "I am not able to classify this ticket.",          # no JSON at all
        '{"decision": "auto_respond", confidence: bad}',   # not valid JSON
        '{"decision": "maybe_respond", "confidence": 0.5}',  # unknown decision
        '["auto_respond"]',                                # JSON, but not an object
    ],
)
def test_unusable_verdicts_escalate(reply):
    """
    Every parse failure resolves to escalate with zero confidence.

    This is the safety property of the whole agent: the failure direction is a
    human reading the ticket, never an automated reply sent on a guess.
    """
    decision, confidence, _ = _parse_triage_verdict(reply)

    assert decision == TriageDecision.ESCALATE
    assert confidence == 0.0


def test_unusable_verdict_explains_itself():
    """A fallback that does not say why it fired is indistinguishable from a real decision."""
    _, _, reasoning = _parse_triage_verdict("no json here")

    assert "no JSON" in reasoning


@pytest.mark.parametrize(
    ("raw_confidence", "expected"),
    [("1.8", 1.0), ("-0.5", 0.0), ('"high"', 0.0)],
)
def test_confidence_is_clamped_without_overriding_the_decision(raw_confidence, expected):
    """A nonsense confidence is not grounds to discard an otherwise valid decision."""
    decision, confidence, _ = _parse_triage_verdict(
        f'{{"decision": "auto_respond", "confidence": {raw_confidence}, "reasoning": "x"}}'
    )

    assert decision == TriageDecision.AUTO_RESPOND
    assert confidence == expected


# ─────────────────────────────────────────────────────────────────────────────
# Triage node
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_triage_node_returns_decision_and_records_itself(monkeypatch):
    stub_chain(
        monkeypatch,
        "_triage_chain",
        '{"decision": "auto_respond", "confidence": 0.9, "reasoning": "clear"}',
    )

    result = await triage_node(make_state())

    assert result["triage_decision"] == TriageDecision.AUTO_RESPOND
    assert result["confidence_score"] == 0.9
    assert result["node_path"] == ["triage"]


@pytest.mark.asyncio
async def test_triage_node_normalises_content_for_later_nodes(monkeypatch):
    seen = stub_chain(
        monkeypatch,
        "_triage_chain",
        '{"decision": "escalate", "confidence": 0.9, "reasoning": "x"}',
    )

    result = await triage_node(
        make_state(content="URGENT\n\n\n   Production   outage.\n\n  All AS2 failing.   ")
    )

    assert result["clean_content"] == "URGENT Production outage. All AS2 failing."
    assert seen[0]["content"] == result["clean_content"]


@pytest.mark.asyncio
async def test_triage_node_escalates_when_the_provider_is_down(monkeypatch):
    """
    A dead provider must degrade to human review, not to a 503.

    The retry wrapper has already exhausted its attempts by the time a node sees
    an exception, so this is the genuinely-unavailable case.
    """
    def _explode(_payload):
        raise ConnectionError("provider unreachable")

    monkeypatch.setattr(
        "app.agent.nodes._triage_chain", lambda: RunnableLambda(_explode)
    )

    result = await triage_node(make_state())

    assert result["triage_decision"] == TriageDecision.ESCALATE
    assert result["confidence_score"] == 0.0
    assert "ConnectionError" in result["triage_reasoning"]


@pytest.mark.asyncio
async def test_triage_node_fills_in_missing_optional_fields(monkeypatch):
    """partner and category are optional on the request, so the node must cope without them."""
    seen = stub_chain(
        monkeypatch,
        "_triage_chain",
        '{"decision": "escalate", "confidence": 0.5, "reasoning": "x"}',
    )

    await triage_node(make_state(partner=None, category=None))

    assert seen[0]["partner"] == "unknown"
    assert seen[0]["category"] == "unclassified"


# ─────────────────────────────────────────────────────────────────────────────
# Branch nodes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_respond_node_delegates_to_the_shared_entry_point(monkeypatch):
    """
    The respond node must call ainvoke_sfr_traced, not reimplement generation.

    If it stopped delegating, the plain endpoint and the agent would drift apart
    on the next prompt change and nothing would fail.
    """
    calls: list[dict] = []

    async def _fake_sfr(**kwargs):
        calls.append(kwargs)
        return "Thank you for contacting support.", "run-123"

    monkeypatch.setattr("app.agent.nodes.ainvoke_sfr_traced", _fake_sfr)

    result = await respond_node(make_state(clean_content="cleaned ticket text"))

    assert result["response"] == "Thank you for contacting support."
    assert result["node_path"] == ["respond"]
    assert calls[0]["raw_content"] == "cleaned ticket text"


@pytest.mark.asyncio
async def test_respond_node_returns_only_its_own_fields(monkeypatch):
    """
    A node returns what it changed. Returning triage fields it did not compute
    would overwrite the real ones during the state merge.
    """
    async def _fake_sfr(**_kwargs):
        return "response text", None

    monkeypatch.setattr("app.agent.nodes.ainvoke_sfr_traced", _fake_sfr)

    result = await respond_node(make_state())

    assert set(result) == {"response", "node_path"}


@pytest.mark.asyncio
async def test_clarify_node_produces_a_question(monkeypatch):
    stub_chain(monkeypatch, "_clarify_chain", "  Which endpoint is affected?  ")

    result = await clarify_node(make_state())

    assert result["clarifying_question"] == "Which endpoint is affected?"
    assert result["node_path"] == ["clarify"]


@pytest.mark.asyncio
async def test_escalate_node_produces_a_handoff_note(monkeypatch):
    stub_chain(monkeypatch, "_escalate_chain", "- Possible breach\n- Assign to security")

    result = await escalate_node(make_state(triage_reasoning="possible breach"))

    assert "security" in result["escalation_summary"]
    assert result["node_path"] == ["escalate"]


@pytest.mark.asyncio
async def test_escalate_node_still_hands_off_when_its_model_fails(monkeypatch):
    """
    Escalate is where every failure lands, so it cannot have a failure mode of
    its own. Without a generated note it assembles one from state.
    """
    def _explode(_payload):
        raise TimeoutError("model timed out")

    monkeypatch.setattr(
        "app.agent.nodes._escalate_chain", lambda: RunnableLambda(_explode)
    )

    result = await escalate_node(make_state(triage_reasoning="possible breach"))

    assert "TimeoutError" in result["escalation_summary"]
    assert "possible breach" in result["escalation_summary"]


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (TriageDecision.AUTO_RESPOND, "respond"),
        (TriageDecision.NEEDS_CLARIFICATION, "clarify"),
        (TriageDecision.ESCALATE, "escalate"),
    ],
)
def test_router_sends_each_decision_to_its_branch(decision, expected):
    assert route_after_triage(make_state(triage_decision=decision)) == expected


def test_router_escalates_an_unroutable_decision():
    """No code path produces this today, which is why it must not raise if one ever does."""
    assert route_after_triage(make_state(triage_decision="something_else")) == "escalate"


# ─────────────────────────────────────────────────────────────────────────────
# Full graph
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "branch", "output_field"),
    [
        ("auto_respond", "respond", "response"),
        ("needs_clarification", "clarify", "clarifying_question"),
        ("escalate", "escalate", "escalation_summary"),
    ],
)
async def test_graph_routes_and_populates_only_that_branch(
    monkeypatch, verdict, branch, output_field
):
    """
    End to end for each of the three paths.

    Asserting the other two output fields are absent is the point: a graph that
    ran every branch would satisfy any single-field assertion.
    """
    stub_chain(
        monkeypatch,
        "_triage_chain",
        f'{{"decision": "{verdict}", "confidence": 0.8, "reasoning": "test"}}',
    )
    stub_chain(monkeypatch, "_clarify_chain", "Which endpoint is affected?")
    stub_chain(monkeypatch, "_escalate_chain", "- Assign to a human")

    async def _fake_sfr(**_kwargs):
        return "Thank you for contacting support.", None

    monkeypatch.setattr("app.agent.nodes.ainvoke_sfr_traced", _fake_sfr)

    final = await build_sfr_agent().ainvoke(make_state())

    assert final["triage_decision"] == TriageDecision(verdict)
    assert final["node_path"] == ["triage", branch]
    assert final[output_field].strip()

    unused = {"response", "clarifying_question", "escalation_summary"} - {output_field}
    assert all(field not in final for field in unused)


@pytest.mark.asyncio
async def test_graph_escalates_a_ticket_triage_could_not_classify(monkeypatch):
    """The safety path, proven through the graph rather than the parser alone."""
    stub_chain(monkeypatch, "_triage_chain", "I cannot classify this ticket.")
    stub_chain(monkeypatch, "_escalate_chain", "- Assign to a human")

    final = await build_sfr_agent().ainvoke(make_state())

    assert final["triage_decision"] == TriageDecision.ESCALATE
    assert final["node_path"] == ["triage", "escalate"]
    assert "response" not in final
