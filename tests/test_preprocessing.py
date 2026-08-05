"""Tests for ticket pre- and post-processing."""

from app.preprocessing import MAX_CONTENT_CHARS, format_sfr_output, preprocess_ticket


def test_collapses_runs_of_whitespace():
    messy = "URGENT\n\n\n     Production   outage.\n\n  All AS2   failing.   "
    assert preprocess_ticket(messy) == "URGENT Production outage. All AS2 failing."


def test_leaves_clean_content_untouched():
    clean = "Database connection timing out since 14:30 UTC."
    assert preprocess_ticket(clean) == clean


def test_truncates_past_the_character_budget():
    result = preprocess_ticket("word " * 2000)

    assert result.endswith("... [truncated]")
    # The marker is appended after the cut, so it sits outside the budget.
    assert len(result) == MAX_CONTENT_CHARS + len("... [truncated]")


def test_custom_max_chars_is_respected():
    assert preprocess_ticket("a" * 100, max_chars=10) == "a" * 10 + "... [truncated]"


def test_content_at_exactly_the_limit_is_not_truncated():
    exact = "a" * MAX_CONTENT_CHARS
    assert preprocess_ticket(exact) == exact


def test_format_output_prepends_ticket_reference():
    result = format_sfr_output("   Thank you for reporting this.   ", "TICK-001")

    assert result.startswith("Re: Ticket TICK-001\n")
    assert result.endswith("Thank you for reporting this.")
