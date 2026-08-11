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

from app.chain import active_model_id, ainvoke_sfr_traced, get_sfr_chain
from app.config import settings
from app.models import HealthResponse, SFRRequest, SFRResponse

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


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health_check() -> HealthResponse:
    """Health check for load balancer and monitoring."""
    return HealthResponse(version=settings.app_version)
