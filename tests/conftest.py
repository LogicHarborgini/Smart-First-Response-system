"""
Shared test configuration.

pytest imports conftest.py before any test module, which is the only window in
which these can be set usefully: app.config calls load_dotenv() at import time,
and load_dotenv does not overwrite variables already present in os.environ. So
setting them here wins over .env, and pydantic-settings also reads os.environ
ahead of the .env file.

The point is that the suite must run offline: no LangSmith network calls, no AWS
credentials, no local model. A test suite that needs any of those is a test suite
that fails on someone else's machine.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LLM_PROVIDER"] = "fake"
