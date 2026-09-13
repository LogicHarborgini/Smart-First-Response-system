"""
Shared state for the SFR triage agent.

Every node receives this dict, and returns only the fields it changed —
LangGraph merges the update into the state rather than replacing it. That is why
a node can return `{"response": "..."}` without wiping out the triage result.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.models import TriageDecision


class TriageState(TypedDict, total=False):
    """
    The ticket as it moves through the graph.

    `total=False` because the state is filled in progressively: the input fields
    exist from the start, the triage fields appear after the triage node, and
    exactly one of response/clarifying_question/escalation_summary is set by
    whichever branch runs. Reading a field that a branch never set returns None
    via .get() rather than raising.

    node_path uses operator.add as its reducer, so a node returns just its own
    name — `{"node_path": ["triage"]}` — and LangGraph concatenates rather than
    overwrites. Without the reducer each node would clobber the previous entry
    and the audit trail would only ever hold the last node to run.
    """

    # ── Input: set once at entry, never modified ──
    ticket_id: str
    content: str
    priority: str
    customer_name: str | None
    partner: str | None
    category: str | None

    # Whitespace-normalised, length-capped content, written by the triage node
    # and read by every node after it. Kept separate from `content` so the raw
    # ticket as received is still visible in the trace inputs.
    #
    # Normalising inside the first node rather than before the graph runs is
    # deliberate: preprocess_ticket is @traceable, and a traceable called with no
    # run in progress becomes its own root trace. Doing it in a node keeps it a
    # child span of the agent run instead of an orphan sitting beside it.
    clean_content: str

    # ── Set by the triage node ──
    triage_decision: TriageDecision
    confidence_score: float
    triage_reasoning: str

    # ── Set by exactly one branch ──
    response: str
    clarifying_question: str
    escalation_summary: str

    # ── Audit trail: which nodes ran, in order ──
    node_path: Annotated[list[str], operator.add]
