"""
Graph assembly for the SFR triage agent.

    START → triage ─┬─ auto_respond ────────→ respond  → END
                    ├─ needs_clarification ─→ clarify  → END
                    └─ escalate ───────────→ escalate → END

One decision point, three terminal branches, no cycles. That shape is the whole
argument for the agent over the plain chain: the plain chain answers every
ticket, including the ones that should not be answered.

There is deliberately no loop here. A clarify → re-triage cycle is the obvious
next feature and the obvious next way to burn a token budget, and it needs a
turn counter and an exit condition before it is safe to add.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import clarify_node, escalate_node, respond_node, triage_node
from app.agent.state import TriageState
from app.chain import active_model_id, tracing_enabled
from app.config import settings
from app.models import TriageDecision

logger = logging.getLogger(__name__)


def route_after_triage(state: TriageState) -> str:
    """
    Pick the branch. Called by LangGraph once the triage node returns.

    The default is escalate rather than a raised error. Triage already resolves
    every failure to ESCALATE, so reaching the default means a decision value
    appeared that the enum does not cover — a state no code path produces today,
    which is exactly why it should route somewhere safe instead of 500ing.
    """
    decision = state.get("triage_decision")
    if decision == TriageDecision.AUTO_RESPOND:
        return "respond"
    if decision == TriageDecision.NEEDS_CLARIFICATION:
        return "clarify"
    if decision != TriageDecision.ESCALATE:
        logger.warning(f"Unroutable triage decision {decision!r} — escalating")
    return "escalate"


def build_sfr_agent():
    """
    Assemble and compile the graph.

    Returns a compiled graph, which implements the same Runnable interface as the
    LCEL chain — ainvoke, astream, and automatic LangSmith tracing with one span
    per node.
    """
    builder = StateGraph(TriageState)

    builder.add_node("triage", triage_node)
    builder.add_node("respond", respond_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("escalate", escalate_node)

    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {"respond": "respond", "clarify": "clarify", "escalate": "escalate"},
    )
    builder.add_edge("respond", END)
    builder.add_edge("clarify", END)
    builder.add_edge("escalate", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_sfr_agent():
    """Compiled agent, built once — same reasoning as get_sfr_chain()."""
    logger.info(f"Building SFR triage agent with model: {active_model_id()}")
    agent = build_sfr_agent()
    logger.info("SFR triage agent built successfully")
    return agent


async def ainvoke_agent_traced(
    *,
    ticket_id: str,
    raw_content: str,
    priority: str,
    customer_name: str | None = None,
    partner: str | None = None,
    category: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """
    Run one ticket through the agent, traced end to end.

    Mirrors ainvoke_sfr_traced: same naming scheme, same metadata, same
    generate-the-run-id-here approach so the caller and the trace agree on an
    identifier without a callback collector. The trace is one tree per ticket,
    with a span per node, so the branch taken is visible in LangSmith rather than
    only in node_path.

    Returns
    -------
    tuple[dict, str | None]
        The final state, and the LangSmith run ID — None when tracing is off.
    """
    initial_state: TriageState = {
        "ticket_id": ticket_id,
        "content": raw_content,
        "priority": priority,
        "customer_name": customer_name,
        "partner": partner,
        "category": category,
        "node_path": [],
    }

    agent = get_sfr_agent()

    if not tracing_enabled():
        final_state = await agent.ainvoke(initial_state)
        return dict(final_state), None

    run_id = uuid.uuid4()
    final_state = await agent.ainvoke(
        initial_state,
        config={
            "run_id": run_id,
            "run_name": f"SFR-Agent-{ticket_id}",
            "metadata": {
                "ticket_id": ticket_id,
                "priority": priority,
                "customer_name": customer_name or "unknown",
                "partner": partner or "unknown",
                "category": category or "unclassified",
                "model_id": active_model_id(),
                "app_version": settings.app_version,
            },
            "tags": [f"priority:{priority}", "sfr", "agent"],
        },
    )
    return dict(final_state), str(run_id)
