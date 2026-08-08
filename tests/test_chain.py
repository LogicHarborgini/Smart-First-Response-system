"""
Tests for provider resolution, model identification, and retry policy.

These cover the reporting path that observability depends on: if active_model_id()
lies, every trace and every API response attributes the output to the wrong model.

The "auto" branch is deliberately not tested here. Resolving it calls
boto3.Session().get_credentials(), which falls through to the EC2 instance
metadata endpoint on a machine with no credentials — a network timeout inside a
unit test. Its behaviour is covered by the explicit branches plus the fallback
test below.
"""

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from app.chain import (
    RETRY_ATTEMPTS,
    _transient_exception_types,
    active_model_id,
    resolve_provider,
    with_transient_retry,
)
from app.config import settings


@pytest.mark.parametrize("provider", ["bedrock", "ollama", "fake"])
def test_explicit_provider_is_honoured(monkeypatch, provider):
    monkeypatch.setattr(settings, "llm_provider", provider)
    assert resolve_provider() == provider


def test_provider_setting_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "  Fake  ")
    assert resolve_provider() == "fake"


def test_unknown_provider_never_leaks_through(monkeypatch):
    """An unrecognised value must not be returned as if it were valid."""
    monkeypatch.setattr(settings, "llm_provider", "gpt-5")
    monkeypatch.setattr(
        "app.chain._auto_detect_provider", lambda: "ollama", raising=True
    )
    assert resolve_provider() == "ollama"


def test_active_model_id_reports_fake(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "fake")
    assert active_model_id() == "fake:canned-responses"


def test_active_model_id_reports_ollama_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_model", "llama3.2")
    assert active_model_id() == "ollama:llama3.2"


def test_active_model_id_reports_bedrock_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "bedrock")
    monkeypatch.setattr(settings, "bedrock_model_id", "anthropic.some-model-v1:0")
    assert active_model_id() == "anthropic.some-model-v1:0"


# ─────────────────────────────────────────────────────────────────────────────
# Retry
# ─────────────────────────────────────────────────────────────────────────────


def test_bedrock_retries_throttling_errors():
    """Bedrock throttling arrives as a botocore ClientError — the case retry exists for."""
    from botocore.exceptions import ClientError

    assert ClientError in _transient_exception_types("bedrock")


def test_ollama_retries_connection_errors():
    """The way a local Ollama daemon fails: not running, or still loading a model."""
    assert httpx.HTTPError in _transient_exception_types("ollama")


def test_fake_provider_has_nothing_to_retry():
    assert _transient_exception_types("fake") == ()


def test_retry_is_not_applied_when_nothing_is_retryable(monkeypatch):
    """
    With no retryable exceptions the runnable is returned untouched rather than
    wrapped in a retry that can never fire.
    """
    monkeypatch.setattr(settings, "llm_provider", "fake")
    runnable = RunnableLambda(lambda x: x)

    assert with_transient_retry(runnable) is runnable


def test_retry_policy_matches_the_provider(monkeypatch):
    """
    Asserted on configuration rather than by exhausting attempts: the real
    backoff waits 1s then 2s, which is not something to spend in a unit test.
    """
    monkeypatch.setattr(settings, "llm_provider", "ollama")

    retrying = with_transient_retry(RunnableLambda(lambda x: x))

    assert retrying.max_attempt_number == RETRY_ATTEMPTS
    assert retrying.wait_exponential_jitter is True
    assert httpx.HTTPError in retrying.retry_exception_types


def test_transient_failure_is_retried_and_succeeds(monkeypatch):
    """One real invocation through the retry, to prove the wiring works end to end."""
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    attempts = []

    def flaky(value: str) -> str:
        attempts.append(value)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection refused")
        return f"ok:{value}"

    retrying = with_transient_retry(RunnableLambda(flaky))

    assert retrying.invoke("ticket") == "ok:ticket"
    assert len(attempts) == 2


def test_programming_errors_are_not_retried(monkeypatch):
    """
    A KeyError from a missing prompt variable fails identically every time.
    Retrying it would only slow the error down and hide the bug.
    """
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    attempts = []

    def broken(value: str) -> str:
        attempts.append(value)
        raise KeyError("ticket_content")

    retrying = with_transient_retry(RunnableLambda(broken))

    with pytest.raises(KeyError):
        retrying.invoke("ticket")
    assert len(attempts) == 1
