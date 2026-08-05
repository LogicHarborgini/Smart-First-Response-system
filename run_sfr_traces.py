"""
Generate sample SFR traces to populate the LangSmith dashboard.

Run from the project root:

    python run_sfr_traces.py

Each ticket produces one trace named SFR-<ticket_id>, tagged with its priority,
with preprocess-ticket as a child span alongside the model call. Requires AWS
credentials (boto3 credential chain) since it calls Bedrock for real.
"""

from __future__ import annotations

import asyncio

from app.chain import ainvoke_sfr_traced
from app.preprocessing import format_sfr_output

# Deliberately varied across priority and problem type — a dashboard where every
# trace looks the same tells you nothing about how the prompt handles range.
# The messy whitespace in TICK-2004 is intentional: it shows preprocess-ticket
# earning its span.
SAMPLE_TICKETS = [
    {
        "ticket_id": "TICK-2001",
        "priority": "P1",
        "customer_name": "Acme Corp",
        "content": (
            "Our EDI 850 purchase orders stopped processing at 02:00 UTC. "
            "We have 47 orders backed up and our warehouse system is rejecting "
            "all incoming files with error code X12-834."
        ),
    },
    {
        "ticket_id": "TICK-2002",
        "priority": "P2",
        "customer_name": "TechFlow",
        "content": (
            "REST API calls to our order management endpoint are returning "
            "429 Too Many Requests since yesterday afternoon. We reduced our "
            "call frequency but are still being throttled."
        ),
    },
    {
        "ticket_id": "TICK-2003",
        "priority": "P3",
        "customer_name": None,
        "content": (
            "Requesting documentation on rotating SFTP host keys for our "
            "managed file transfer setup. Not urgent, planning a Q3 change."
        ),
    },
    {
        "ticket_id": "TICK-2004",
        "priority": "P1",
        "customer_name": "Northwind Logistics",
        "content": (
            "URGENT\n\n\n     Production   outage.\n\n"
            "All   inbound  AS2 messages    failing since 09:15.\n\n\n"
            "     Trading partner reports MDN timeouts.      "
        ),
    },
    {
        "ticket_id": "TICK-2005",
        "priority": "P2",
        "customer_name": "Globex",
        "content": (
            "Scheduled data mapping job completed but output file contains only "
            "headers and no rows. Source system shows 1,200 records available. "
            "This is the second occurrence this month."
        ),
    },
]


async def main() -> None:
    for ticket in SAMPLE_TICKETS:
        ticket_id = ticket["ticket_id"]
        print(f"→ {ticket_id} ({ticket['priority']})")

        try:
            first_response, run_id = await ainvoke_sfr_traced(
                ticket_id=ticket_id,
                raw_content=ticket["content"],
                priority=ticket["priority"],
                customer_name=ticket["customer_name"],
            )
        except Exception as e:
            # Keep going: one bad ticket should not cost you the whole batch.
            print(f"  failed: {e}\n")
            continue

        print(format_sfr_output(first_response, ticket_id))
        print(f"  run_id: {run_id}\n")

    print("Done — check https://smith.langchain.com under your LANGSMITH_PROJECT")


if __name__ == "__main__":
    asyncio.run(main())
