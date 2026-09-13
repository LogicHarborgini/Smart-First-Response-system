"""
FastAPI application for Smart First Response.

An LLM-powered support ticket response generator.
Stack: FastAPI + LangChain LCEL + Amazon Bedrock (Claude 3 Sonnet)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import ainvoke_agent_traced, get_sfr_agent
from app.chain import active_model_id, ainvoke_sfr_traced, get_sfr_chain
from app.config import settings
from app.models import (
    HealthResponse,
    SFRAgentResponse,
    SFRRequest,
    SFRResponse,
    TriageDecision,
)

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup checks.

    The LangSmith check reads configuration only — no network call. Startup
    should not depend on an external service being reachable, and the failure
    this actually needs to catch is local: tracing switched on with no API key,
    which otherwise produces no traces and no error.
    """
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true":
        if os.getenv("LANGSMITH_API_KEY"):
            logger.info(
                "LangSmith tracing enabled | project=%s",
                os.getenv("LANGSMITH_PROJECT", "default"),
            )
        else:
            logger.warning(
                "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is not set — "
                "no traces will be sent"
            )
    else:
        logger.info("LangSmith tracing disabled")

    # Build the chain now so the first request does not pay for boto3 session
    # setup. lru_cache does not memoise exceptions, so a failure here is retried
    # on the first request rather than being permanent.
    try:
        get_sfr_chain()
        get_sfr_agent()
        logger.info("Active model: %s", active_model_id())
    except Exception as e:
        logger.warning(f"Chain pre-warm failed: {e} — retrying on first request")

    yield


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Generates professional first responses for support tickets "
        "using Amazon Bedrock (Claude 3 Sonnet) via LangChain LCEL."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post(
    "/api/v1/generate-response",
    response_model=SFRResponse,
    summary="Generate first response for a support ticket",
    tags=["SFR"],
)
async def generate_first_response(request: SFRRequest) -> SFRResponse:
    """
    Accepts a support ticket and returns a generated first response.

    - Input validated automatically by Pydantic (422 on invalid input)
    - LangChain LCEL chain calls Amazon Bedrock asynchronously
    - Response includes model ID and latency for observability
    """
    start = time.perf_counter()

    logger.info(
        f"SFR request | ticket_id={request.ticket_id} | priority={request.priority.value}"
    )

    try:
        # Preprocessing happens inside ainvoke_sfr_traced so it is traced as a
        # child span of this ticket's run rather than as a separate root trace.
        first_response, run_id = await ainvoke_sfr_traced(
            ticket_id=request.ticket_id,
            raw_content=request.content,
            priority=request.priority.value,
            customer_name=request.customer_name,
        )
    except Exception as e:
        logger.error(f"Chain invocation failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"LLM service unavailable: {e}"
        ) from e

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        f"SFR response | ticket_id={request.ticket_id} | "
        f"latency={latency_ms}ms | run_id={run_id}"
    )

    return SFRResponse(
        ticket_id=request.ticket_id,
        first_response=first_response,
        model_used=active_model_id(),
        latency_ms=latency_ms,
        langsmith_run_id=run_id,
    )


@app.post(
    "/api/v1/generate-response/agent",
    response_model=SFRAgentResponse,
    summary="Triage a ticket, then respond, ask for detail, or escalate",
    tags=["SFR"],
)
async def generate_agent_response(request: SFRRequest) -> SFRAgentResponse:
    """
    Route a support ticket through the LangGraph triage agent.

    Same request body as /api/v1/generate-response, but the ticket is classified
    before anything is written. Three outcomes, and only one of the three output
    fields is populated on any given call:

    - `auto_respond`        → `first_response`, the same generation the plain
                              endpoint produces
    - `needs_clarification` → `clarifying_question`, one question to send back
    - `escalate`            → `escalation_summary`, an internal handoff note

    Costs two model calls against the plain endpoint's one, and it can decline to
    answer. Use the plain endpoint when the ticket is already known to be
    answerable; use this one when it is not.
    """
    start = time.perf_counter()

    logger.info(
        f"SFR agent request | ticket_id={request.ticket_id} | "
        f"priority={request.priority.value}"
    )

    try:
        final_state, run_id = await ainvoke_agent_traced(
            ticket_id=request.ticket_id,
            raw_content=request.content,
            priority=request.priority.value,
            customer_name=request.customer_name,
            partner=request.partner,
            category=request.category,
        )
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"LLM service unavailable: {e}"
        ) from e

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    # The decision is read with a default rather than indexed. Triage always sets
    # it, but a missing key here would turn a degraded response into a KeyError
    # and a 500 — the wrong trade for a field with an obvious safe fallback.
    decision = final_state.get("triage_decision", TriageDecision.ESCALATE)

    logger.info(
        f"SFR agent response | ticket_id={request.ticket_id} | "
        f"decision={decision.value} | path={final_state.get('node_path')} | "
        f"latency={latency_ms}ms | run_id={run_id}"
    )

    return SFRAgentResponse(
        ticket_id=request.ticket_id,
        decision=decision,
        confidence=final_state.get("confidence_score", 0.0),
        reasoning=final_state.get("triage_reasoning", ""),
        node_path=final_state.get("node_path", []),
        first_response=final_state.get("response"),
        clarifying_question=final_state.get("clarifying_question"),
        escalation_summary=final_state.get("escalation_summary"),
        model_used=active_model_id(),
        latency_ms=latency_ms,
        langsmith_run_id=run_id,
    )


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health_check() -> HealthResponse:
    """Health check for load balancer and monitoring."""
    return HealthResponse(version=settings.app_version)
