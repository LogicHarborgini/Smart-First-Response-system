"""
Pre- and post-processing for SFR ticket content.

These are plain Python functions, not LangChain runnables, so they would be
invisible in LangSmith by default. The @traceable decorator turns each one into
a child span of whichever trace is active when it is called, giving you:

    SFR-TICK-2001
    ├── preprocess-ticket    [2ms]
    └── RunnableSequence     [1.2s]
        └── ChatBedrock      [1.2s]

Without it you can see that a response was slow but not which stage was slow.
"""

from __future__ import annotations

from langsmith import traceable

# Rough token budget: 1 token is about 4 characters, so 4000 chars is ~1000
# tokens of ticket content — comfortable alongside the system prompt and the
# 512-token response ceiling set in config.
MAX_CONTENT_CHARS = 4000


@traceable(name="preprocess-ticket", tags=["preprocessing"])
def preprocess_ticket(raw_content: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    """
    Normalise whitespace and cap ticket length before it reaches the model.

    Pasted ticket bodies often carry hard-wrapped lines, repeated blank lines,
    and quoted email chains. Collapsing runs of whitespace to single spaces cuts
    tokens without changing meaning.

    Note this runs *after* Pydantic's content validator, which has already
    stripped and length-checked the input — so the minimum-length case cannot
    reach here.
    """
    content = " ".join(raw_content.split())

    if len(content) > max_chars:
        content = content[:max_chars] + "... [truncated]"

    return content


def format_sfr_output(raw_response: str, ticket_id: str) -> str:
    """
    Add a ticket reference header to a generated response.

    This is presentation, not API payload — the JSON response from
    /api/v1/generate-response deliberately returns the unformatted text so
    callers can render it their own way. Use this for terminal output, email
    bodies, or anywhere a human reads the response directly.

    Deliberately not @traceable. It is applied by the caller after the SFR run
    has finished, so tracing it would add a root run per ticket that is a
    sibling of the real trace rather than part of it — noise, not observability.
    """
    header = f"Re: Ticket {ticket_id}\n" + "─" * 40 + "\n"
    return header + raw_response.strip()
