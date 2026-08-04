"""
LangChain LCEL chain for Smart First Response.

This is the core of SFR:
  prompt | llm | parser
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional enterprise support engineer writing \
first responses to customer support tickets. Your goal is to:

1. Acknowledge the issue clearly and empathetically
2. Confirm the ticket has been received and is being investigated
3. Set realistic expectations for next steps and timeline
4. Use professional, clear language — no jargon the customer wouldn't understand

Keep the response concise: 3-4 sentences maximum. Do not attempt to diagnose \
or resolve the issue in the first response — that is the job of the follow-up."""

HUMAN_TEMPLATE = """Support Ticket (Priority: {priority}):
{ticket_content}

Write the first response:"""


@lru_cache(maxsize=1)
def get_sfr_chain() -> Runnable:
    """
    Build and return the SFR LCEL chain.

    @lru_cache(maxsize=1) ensures the chain is built once and reused —
    ChatBedrock initialises a boto3 session on creation, which is expensive.
    In production: consider a proper dependency injection pattern with FastAPI.

    Returns
    -------
    Runnable
        A LangChain LCEL chain: prompt | llm | parser
    """
    logger.info(f"Building SFR chain with model: {settings.bedrock_model_id}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])

    llm = ChatBedrock(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_default_region,
        model_kwargs={
            "max_tokens": settings.bedrock_max_tokens,
            "temperature": settings.bedrock_temperature,
        },
        streaming=True,   # enables chain.astream()
    )

    parser = StrOutputParser()

    chain: Runnable = prompt | llm | parser
    logger.info("SFR chain built successfully")
    return chain
