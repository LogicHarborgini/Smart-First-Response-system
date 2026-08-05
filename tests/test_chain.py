"""
Tests for provider resolution and model identification.

These cover the reporting path that observability depends on: if active_model_id()
lies, every trace and every API response attributes the output to the wrong model.

The "auto" branch is deliberately not tested here. Resolving it calls
boto3.Session().get_credentials(), which falls through to the EC2 instance
metadata endpoint on a machine with no credentials — a network timeout inside a
unit test. Its behaviour is covered by the explicit branches plus the fallback
test below.
"""

import pytest

from app.chain import active_model_id, resolve_provider
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
