"""
LangGraph triage agent for SFR.

The plain endpoint runs one chain and always takes the same path. This package
adds the path the chain cannot take: read the ticket first, decide whether it can
be answered at all, and route accordingly.
"""
