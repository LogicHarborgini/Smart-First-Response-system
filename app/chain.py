"""
LangChain LCEL chain for Smart First Response.

This is the core of SFR:
  prompt | llm | parser
"""

from __future__ import annotations

import logging
import os
import uuid
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langsmith import traceable

from app.config import settings
from app.preprocessing import preprocess_ticket

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


# ─────────────────────────────────────────────────────────────────────────────
# Provider selection
# ─────────────────────────────────────────────────────────────────────────────

# Canned responses for the "fake" provider. Written to look like real first
# responses so the eval criteria have something meaningful to chew on — but eval
# scores against this provider measure the harness, not the model.
_FAKE_RESPONSES = [
    "Thank you for reporting this. We have received your ticket and our "
    "integration team is investigating with the priority you flagged. To help us "
    "diagnose it quickly, please share the timestamp of the last successful "
    "transmission and your trading partner ID. We will update you shortly.",
    "Thank you for getting in touch. We have received your report and are "
    "reviewing the affected configuration on our side. Could you confirm which "
    "endpoints are impacted and when the behaviour started? Our team will come "
    "back to you with findings.",
    "Thank you for planning ahead on this. We have received your request and our "
    "team is confirming the current details. We will send those over to you well "
    "before your maintenance window. Please let us know if your timeline changes.",
]


def _auto_detect_provider() -> str:
    """
    Pick a provider by probing for AWS credentials.

    Credentials are resolved through boto3 itself rather than by checking AWS_*
    environment variables: `aws configure` writes to ~/.aws/credentials and sets
    no env vars, so an env-var check reports "no AWS" on the most common local
    setup. Kept separate from resolve_provider so tests can stub the probe —
    calling it without credentials falls through to the EC2 metadata endpoint and
    blocks until that times out.
    """
    try:
        import boto3

        if boto3.Session().get_credentials() is not None:
            return "bedrock"
        logger.info("No AWS credentials resolvable — using ollama")
    except Exception as e:
        logger.info(f"boto3 unavailable ({e}) — using ollama")

    return "ollama"


def resolve_provider() -> str:
    """Decide which provider to use. An explicit setting always wins."""
    configured = settings.llm_provider.strip().lower()
    if configured in {"bedrock", "ollama", "fake"}:
        return configured
    if configured != "auto":
        logger.warning(f"Unknown llm_provider '{configured}' — falling back to auto")

    return _auto_detect_provider()


def active_model_id() -> str:
    """
    Identifier for the model actually in use, for responses and trace metadata.

    Reporting settings.bedrock_model_id unconditionally would label an Ollama or
    fake run as Claude 3 Sonnet, which is exactly the kind of thing observability
    is supposed to stop you doing.
    """
    provider = resolve_provider()
    if provider == "bedrock":
        return settings.bedrock_model_id
    if provider == "ollama":
        return f"ollama:{settings.ollama_model}"
    return "fake:canned-responses"


def build_chat_model(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    bedrock_model_id: str | None = None,
    ollama_model: str | None = None,
    fake_responses: list[str] | None = None,
    streaming: bool = True,
) -> BaseChatModel:
    """
    Construct a chat model for the resolved provider (providers imported lazily).

    Shared by the SFR chain and the eval judge so a single LLM_PROVIDER setting
    governs both. A judge pinned to one provider would make the eval harness
    unrunnable in exactly the environments the fallback exists for.

    Overrides let a caller keep the provider choice while changing the model or
    sampling — the judge wants temperature 0 and a cheaper model than the chain.
    """
    provider = resolve_provider()
    temperature = settings.bedrock_temperature if temperature is None else temperature
    max_tokens = settings.bedrock_max_tokens if max_tokens is None else max_tokens

    if provider == "bedrock":
        from langchain_aws import ChatBedrock

        return ChatBedrock(
            model_id=bedrock_model_id or settings.bedrock_model_id,
            region_name=settings.aws_default_region,
            model_kwargs={"max_tokens": max_tokens, "temperature": temperature},
            streaming=streaming,   # enables chain.astream()
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # No streaming flag: ChatOllama streams through .astream() natively.
        return ChatOllama(
            model=ollama_model or settings.ollama_model,
            temperature=temperature,
            num_predict=max_tokens,
        )

    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    logger.warning(
        "llm_provider=fake — responses are canned. Tracing and evals are real; "
        "response quality is not."
    )
    return FakeListChatModel(responses=fake_responses or _FAKE_RESPONSES)


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
    logger.info(f"Building SFR chain with model: {active_model_id()}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])

    llm = build_chat_model()

    parser = StrOutputParser()

    chain: Runnable = prompt | llm | parser
    logger.info("SFR chain built successfully")
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Traced invocation
# ─────────────────────────────────────────────────────────────────────────────


def _tracing_enabled() -> bool:
    """Whether LangSmith tracing is switched on for this process."""
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"


@traceable(run_type="chain")
async def _sfr_pipeline(
    *,
    ticket_id: str,
    raw_content: str,
    priority: str,
    customer_name: str | None,
) -> str:
    """
    The whole SFR operation: preprocess the ticket, then invoke the chain.

    Preprocessing happens *inside* this traced function on purpose. A @traceable
    function called outside any active trace becomes its own root run, so calling
    preprocess_ticket() before the chain produced orphan "preprocess-ticket"
    traces sitting alongside the SFR ones instead of spans nested within them.
    Wrapping both steps in one parent gives the tree LangSmith is meant to show:

        SFR-TICK-2001
        ├── preprocess-ticket   [2ms]
        └── RunnableSequence    [1.2s]
            └── ChatBedrock     [1.2s]

    ticket_id and customer_name are unused in the body but kept in the signature
    because @traceable records arguments as the run's inputs — they are how the
    trace becomes searchable.
    """
    ticket_content = preprocess_ticket(raw_content)

    chain = get_sfr_chain()
    return await chain.ainvoke({
        "ticket_content": ticket_content,
        "priority": priority,
    })


async def ainvoke_sfr_traced(
    *,
    ticket_id: str,
    raw_content: str,
    priority: str,
    customer_name: str | None = None,
) -> tuple[str, str | None]:
    """
    Run SFR for one ticket, traced end to end.

    What turns an anonymous trace into a searchable one:

    - name       the trace title. Without it every trace reads
                 "RunnableSequence" and they cannot be told apart.
    - metadata   key/value pairs to filter and group traces by.
    - tags       categorical labels, e.g. priority:P1, for saved views.

    The run ID is generated here and handed to LangSmith rather than read back
    afterwards, which avoids a callback collector and guarantees the caller and
    the trace agree on the identifier.

    Returns
    -------
    tuple[str, str | None]
        The generated response, and the LangSmith run ID — None when tracing is
        disabled, since there is no trace to point at.
    """
    if not _tracing_enabled():
        response = await _sfr_pipeline(
            ticket_id=ticket_id,
            raw_content=raw_content,
            priority=priority,
            customer_name=customer_name,
        )
        return response, None

    run_id = uuid.uuid4()
    response = await _sfr_pipeline(
        ticket_id=ticket_id,
        raw_content=raw_content,
        priority=priority,
        customer_name=customer_name,
        langsmith_extra={
            "run_id": run_id,
            "name": f"SFR-{ticket_id}",
            "metadata": {
                "ticket_id": ticket_id,
                "priority": priority,
                "customer_name": customer_name or "unknown",
                "model_id": active_model_id(),
                "app_version": settings.app_version,
            },
            "tags": [f"priority:{priority}", "sfr"],
        },
    )

    return response, str(run_id)
