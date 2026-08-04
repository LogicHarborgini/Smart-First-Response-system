"""
FastAPI application for Smart First Response.

An LLM-powered support ticket response generator.
Stack: FastAPI + LangChain LCEL + Amazon Bedrock (Claude 3 Sonnet)
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.chain import get_sfr_chain
from app.config import settings
from app.models import HealthResponse, SFRRequest, SFRResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Generates professional first responses for support tickets "
        "using Amazon Bedrock (Claude 3 Sonnet) via LangChain LCEL."
    ),
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
    chain = get_sfr_chain()

    logger.info(
        f"SFR request | ticket_id={request.ticket_id} | priority={request.priority.value}"
    )

    try:
        first_response: str = await chain.ainvoke({
            "ticket_content": request.content,
            "priority": request.priority.value,
        })
    except Exception as e:
        logger.error(f"Chain invocation failed: {e}")
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e}")

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(f"SFR response | ticket_id={request.ticket_id} | latency={latency_ms}ms")

    return SFRResponse(
        ticket_id=request.ticket_id,
        first_response=first_response,
        model_used=settings.bedrock_model_id,
        latency_ms=latency_ms,
    )


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health_check() -> HealthResponse:
    """Health check for load balancer and monitoring."""
    return HealthResponse(version=settings.app_version)
