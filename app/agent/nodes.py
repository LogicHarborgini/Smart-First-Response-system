"""
The four nodes of the SFR triage agent.

A node is an async function taking the current state and returning only the
fields it changed. Nothing else is required of it — no base class, no
registration. That is what makes each one testable on its own: call it with a
dict, assert on the dict it returns.

Every node is deliberately thin. The respond node in particular delegates to
ainvoke_sfr_traced, the same entry point the plain endpoint uses, so there is one
implementation of "generate a first response" rather than two that can drift.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.agent.state import TriageState
from app.chain import ainvoke_sfr_traced, build_chat_model, with_transient_retry
from app.models import TriageDecision
from app.preprocessing import preprocess_ticket

logger = logging.getLogger(__name__)


def _content(state: TriageState) -> str:
    """
    The ticket text a node should send to a model.

    Falls back to the raw content so a node can be unit-tested with a state that
    never went through triage. In a real graph run clean_content is always set,
    because triage runs first.
    """
    return state.get("clean_content") or state["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

TRIAGE_SYSTEM_PROMPT = """You are a support ticket triage system for an \
enterprise B2B integration platform. Classify the ticket into exactly one of \
three outcomes.

Reply with JSON only, no prose and no code fences:
{{"decision": "auto_respond" | "needs_clarification" | "escalate", \
"confidence": 0.0-1.0, "reasoning": "one sentence"}}

Decision rules:
- "auto_respond": the ticket describes a specific, understandable problem. A \
professional acknowledgement can be sent now without guessing at what is wrong.
- "needs_clarification": the ticket is too vague to acknowledge meaningfully — \
no symptom, no timeframe, no system named. Asking one question is worth more \
than any reply you could write.
- "escalate": data loss, a suspected security breach, legal or contractual \
threat, or a P1 outage where an automated reply would read as dismissive.

Judge on the content of the ticket, not its priority label alone. A P3 \
reporting exposed credentials still escalates; a P1 describing a clear, \
well-scoped failure can still be auto-responded."""

TRIAGE_HUMAN_TEMPLATE = """Ticket ID: {ticket_id}
Partner: {partner}
Priority: {priority}
Category: {category}

Content:
{content}

Classify this ticket."""


CLARIFY_SYSTEM_PROMPT = """You are an enterprise support engineer. A ticket has \
arrived without enough information to act on.

Write exactly ONE clarifying question that would unblock you. Rules:
- One question, not several
- Reference the specific gap in this ticket, never a generic "can you tell us more"
- Professional and courteous — the customer is not at fault for a vague ticket
- Two or three sentences at most, including a brief acknowledgement first"""

CLARIFY_HUMAN_TEMPLATE = """Ticket (Priority: {priority}) from {partner}:
{content}

Write the clarifying question:"""


ESCALATE_SYSTEM_PROMPT = """You are writing an internal handoff note for the \
support engineer who will pick up an escalated ticket. This is not sent to the \
customer.

Cover, in short bullet points:
1. Why this cannot be auto-answered
2. The technical details that matter
3. A suggested first action

Under 120 words. Be direct — the engineer reading this is busy."""

ESCALATE_HUMAN_TEMPLATE = """Ticket ID: {ticket_id}
Partner: {partner}
Priority: {priority}
Category: {category}

Content:
{content}

Triage reasoning: {triage_reasoning}

Write the handoff note:"""


# ─────────────────────────────────────────────────────────────────────────────
# Canned output for the fake provider
# ─────────────────────────────────────────────────────────────────────────────

# The fake provider returns whatever it is handed, so the triage model has to be
# given valid JSON or every fake run would fall through to the escalate default
# and the respond path would never be exercised locally.
#
# One verdict, not a rotating list: a fake run has to be reproducible. That does
# mean a fake-provider run only ever demonstrates the auto_respond branch — the
# other two are covered by tests, which stub the model directly.
_FAKE_TRIAGE_VERDICT = [
    (
        '{"decision": "auto_respond", "confidence": 0.9, '
        '"reasoning": "fake triage — exercises the graph, says nothing about the ticket"}'
    )
]

_FAKE_CLARIFY_RESPONSE = [
    (
        "Thank you for raising this. To investigate further, could you confirm "
        "which system or endpoint is affected and roughly when the behaviour started?"
    )
]

_FAKE_ESCALATE_RESPONSE = [
    (
        "- Cannot be auto-answered: fake provider, no real assessment performed\n"
        "- Details: see original ticket content\n"
        "- Suggested first action: review manually"
    )
]


# ─────────────────────────────────────────────────────────────────────────────
# Chains
# ─────────────────────────────────────────────────────────────────────────────

# Each chain is cached because building a model opens a provider client — a
# boto3 session for Bedrock — which is far too expensive to repeat per ticket.
# The cache is also the seam the tests patch: replacing the builder replaces the
# model without any provider being constructed.


@lru_cache(maxsize=1)
def _triage_chain() -> Runnable:
    """
    Classification chain, pinned to temperature 0.

    Temperature is 0 rather than the 0.1-ish that reads as "nearly
    deterministic": a ticket that triages as escalate at 09:00 and auto_respond
    at 09:05 on identical text is not a system anyone can regression-test, and
    the difference in output quality for a three-way classification is nil.

    max_tokens is small on purpose. The reply is one JSON object; leaving the
    chain's 512-token budget in place just gives a chatty model room to append an
    explanation after the JSON.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", TRIAGE_SYSTEM_PROMPT),
        ("human", TRIAGE_HUMAN_TEMPLATE),
    ])
    model = build_chat_model(
        temperature=0.0,
        max_tokens=256,
        fake_responses=_FAKE_TRIAGE_VERDICT,
        streaming=False,
    )
    return prompt | with_transient_retry(model) | StrOutputParser()


@lru_cache(maxsize=1)
def _clarify_chain() -> Runnable:
    """Question-writing chain. Slightly warm — this text is read by a customer."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", CLARIFY_SYSTEM_PROMPT),
        ("human", CLARIFY_HUMAN_TEMPLATE),
    ])
    model = build_chat_model(
        temperature=0.3,
        max_tokens=256,
        fake_responses=_FAKE_CLARIFY_RESPONSE,
        streaming=False,
    )
    return prompt | with_transient_retry(model) | StrOutputParser()


@lru_cache(maxsize=1)
def _escalate_chain() -> Runnable:
    """Handoff-note chain. Cool — this is an internal summary, not prose."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", ESCALATE_SYSTEM_PROMPT),
        ("human", ESCALATE_HUMAN_TEMPLATE),
    ])
    model = build_chat_model(
        temperature=0.1,
        max_tokens=384,
        fake_responses=_FAKE_ESCALATE_RESPONSE,
        streaming=False,
    )
    return prompt | with_transient_retry(model) | StrOutputParser()


# ─────────────────────────────────────────────────────────────────────────────
# Triage verdict parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_triage_verdict(raw: str) -> tuple[TriageDecision, float, str]:
    """
    Pull the decision out of the model's reply, or fail to escalate.

    Three things can go wrong, and all three land on ESCALATE with confidence 0:
    no JSON object in the reply, JSON that will not parse, or a decision string
    that is not one of the three known values. Each is reported in the reasoning
    so the outcome is explainable rather than mysterious.

    Escalating is the safe direction. A ticket wrongly sent to a human costs
    someone a minute; a wrong automated reply sent to a customer during a
    security incident cannot be taken back.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return TriageDecision.ESCALATE, 0.0, f"triage returned no JSON: {raw[:120]}"

    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return TriageDecision.ESCALATE, 0.0, f"triage returned invalid JSON: {raw[:120]}"

    if not isinstance(verdict, dict):
        return TriageDecision.ESCALATE, 0.0, f"triage returned non-object JSON: {raw[:120]}"

    raw_decision = str(verdict.get("decision", "")).strip().lower()
    try:
        decision = TriageDecision(raw_decision)
    except ValueError:
        return (
            TriageDecision.ESCALATE,
            0.0,
            f"triage returned unknown decision {raw_decision!r}",
        )

    # A confidence that is missing, non-numeric, or outside 0-1 says nothing
    # useful, so it is clamped rather than trusted. The decision still stands —
    # a bad confidence value is not grounds to override a valid classification.
    try:
        confidence = min(1.0, max(0.0, float(verdict.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    reasoning = str(verdict.get("reasoning", "")).strip()[:300] or "no reasoning given"
    return decision, confidence, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────


async def triage_node(state: TriageState) -> dict:
    """
    Decide what happens to this ticket. Everything downstream follows from here.

    Also normalises the ticket text once, for every node after it — see the
    clean_content note in state.py for why that happens here.

    An exception from the model is caught rather than allowed to fail the graph:
    a provider outage should degrade the agent to "a human looks at it", not to a
    503. The retry wrapper has already made three attempts by this point, so
    reaching the except branch means the provider is genuinely unavailable.
    """
    clean_content = preprocess_ticket(state["content"])

    try:
        raw = await _triage_chain().ainvoke({
            "ticket_id": state["ticket_id"],
            "partner": state.get("partner") or "unknown",
            "priority": state["priority"],
            "category": state.get("category") or "unclassified",
            "content": clean_content,
        })
        decision, confidence, reasoning = _parse_triage_verdict(raw)
    except Exception as e:
        logger.error(f"Triage model call failed | ticket_id={state['ticket_id']} | {e}")
        decision = TriageDecision.ESCALATE
        confidence = 0.0
        reasoning = f"triage unavailable ({type(e).__name__}) — escalating for human review"

    logger.info(
        f"Triage | ticket_id={state['ticket_id']} | decision={decision.value} | "
        f"confidence={confidence:.2f}"
    )

    return {
        "clean_content": clean_content,
        "triage_decision": decision,
        "confidence_score": confidence,
        "triage_reasoning": reasoning,
        "node_path": ["triage"],
    }


async def respond_node(state: TriageState) -> dict:
    """
    Generate the customer-facing first response.

    Delegates to ainvoke_sfr_traced — the same call the plain endpoint makes — so
    a prompt change lands on both paths at once. The run ID it returns is
    discarded here because the agent surfaces its own, covering the whole graph
    rather than just this step.
    """
    first_response, _run_id = await ainvoke_sfr_traced(
        ticket_id=state["ticket_id"],
        raw_content=_content(state),
        priority=state["priority"],
        customer_name=state.get("customer_name"),
    )
    return {"response": first_response, "node_path": ["respond"]}


async def clarify_node(state: TriageState) -> dict:
    """Ask the one question that would make the ticket actionable."""
    question = await _clarify_chain().ainvoke({
        "partner": state.get("partner") or state.get("customer_name") or "the customer",
        "priority": state["priority"],
        "content": _content(state),
    })
    return {"clarifying_question": question.strip(), "node_path": ["clarify"]}


async def escalate_node(state: TriageState) -> dict:
    """
    Write the internal handoff note for whoever picks the ticket up.

    This node is the one that must not fail. It is where the safe default sends
    every triage failure, so if it raised, the failure path would have a failure
    path of its own. On error it falls back to a note assembled from state, which
    is worse than a written summary but still tells a human what arrived.
    """
    reasoning = state.get("triage_reasoning") or "no reasoning recorded"
    try:
        summary = await _escalate_chain().ainvoke({
            "ticket_id": state["ticket_id"],
            "partner": state.get("partner") or "unknown",
            "priority": state["priority"],
            "category": state.get("category") or "unclassified",
            "content": _content(state),
            "triage_reasoning": reasoning,
        })
        summary = summary.strip()
    except Exception as e:
        logger.error(f"Escalation note failed | ticket_id={state['ticket_id']} | {e}")
        summary = (
            f"- Escalated without a generated summary: {type(e).__name__}\n"
            f"- Triage reasoning: {reasoning}\n"
            f"- Priority {state['priority']}, category "
            f"{state.get('category') or 'unclassified'}\n"
            f"- Suggested first action: read the ticket in full and triage manually"
        )

    return {"escalation_summary": summary, "node_path": ["escalate"]}
