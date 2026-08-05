"""
Local eval harness for SFR — no LangSmith account or judge model required.

Run from the project root:

    python -m evals.simple_eval

Every criterion here is a deterministic string/shape check, so this runs in
milliseconds after the model call and costs nothing beyond the SFR invocations
themselves. It is the regression gate: change the system prompt, rerun, compare
the overall score against evals/baseline_results.json.

Keyword checks are a proxy for quality, not a measure of it. They catch the
failures that matter in practice — a response that forgets to acknowledge the
ticket, or rambles past the length guidance — and they never disagree with
themselves between runs. For judgement calls, see evals/sfr_eval.py.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.chain import ainvoke_sfr_traced

BASELINE_PATH = Path(__file__).parent / "baseline_results.json"


# ─────────────────────────────────────────────────────────────────────────────
# Criteria
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalCriterion:
    """
    One pass/fail check.

    check       (response, test_case) -> bool
    applies_to  (test_case) -> bool. Criteria that only make sense for some
                tickets are skipped elsewhere rather than counted as failures —
                see signals_urgency, which would otherwise punish the model for
                correctly staying calm on a P3.
    """

    name: str
    check: Callable[[str, dict], bool]
    applies_to: Callable[[dict], bool] = lambda tc: True


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in phrases)


def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


CRITERIA: list[EvalCriterion] = [
    EvalCriterion(
        name="acknowledges_ticket",
        check=lambda resp, _: _contains_any(
            resp,
            ["thank you", "thanks", "we have received", "we've received",
             "we understand", "i understand", "apolog"],
        ),
    ),
    EvalCriterion(
        name="sets_expectation",
        check=lambda resp, _: _contains_any(
            resp,
            ["we will", "we'll", "our team", "next step", "shortly", "within",
             "please provide", "please share", "update you", "investigating"],
        ),
    ),
    EvalCriterion(
        # Only P1. The system prompt asks for realistic expectations, so urgency
        # language on a P3 documentation request is a defect, not a pass.
        name="signals_urgency",
        check=lambda resp, _: _contains_any(
            resp,
            ["urgen", "immediat", "priorit", "escalat", "critical", "right away"],
        ),
        applies_to=lambda tc: tc["priority"] == "P1",
    ),
    EvalCriterion(
        # System prompt says "3-4 sentences maximum". Allowing 6 leaves room for
        # a greeting and a sign-off without letting an essay through.
        name="respects_length_guidance",
        check=lambda resp, _: _sentence_count(resp) <= 6,
    ),
    EvalCriterion(
        name="not_too_short",
        check=lambda resp, _: len(resp.split()) >= 25,
    ),
    EvalCriterion(
        name="not_too_long",
        check=lambda resp, _: len(resp.split()) <= 200,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "ticket_id": "EVAL-001",
        "priority": "P1",
        "customer_name": "Acme Corp",
        "content": (
            "EDI 850 purchase orders stopped processing at 02:00 UTC. "
            "47 orders backed up. Error code X12-834."
        ),
    },
    {
        "ticket_id": "EVAL-002",
        "priority": "P2",
        "customer_name": "TechFlow",
        "content": (
            "REST API endpoint returning 429 Too Many Requests since yesterday. "
            "We reduced call frequency but are still throttled."
        ),
    },
    {
        "ticket_id": "EVAL-003",
        "priority": "P3",
        "customer_name": "DataSync",
        "content": (
            "Please confirm the new SFTP server IP address so we can update our "
            "firewall whitelist during the next maintenance window."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    ticket_id: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def score(self) -> float:
        applicable = len(self.passed) + len(self.failed)
        return len(self.passed) / applicable if applicable else 0.0


async def run_evals() -> dict:
    results: list[EvalResult] = []

    for tc in TEST_CASES:
        result = EvalResult(ticket_id=tc["ticket_id"])

        try:
            response, _run_id = await ainvoke_sfr_traced(
                ticket_id=tc["ticket_id"],
                raw_content=tc["content"],
                priority=tc["priority"],
                customer_name=tc["customer_name"],
            )
        except Exception as e:
            # A failed invocation scores 0 rather than aborting the run — a
            # partial baseline is still worth writing.
            result.error = str(e)
            results.append(result)
            print(f"\n{tc['ticket_id']} ({tc['priority']}) — ERROR: {e}")
            continue

        for criterion in CRITERIA:
            if not criterion.applies_to(tc):
                result.skipped.append(criterion.name)
            elif criterion.check(response, tc):
                result.passed.append(criterion.name)
            else:
                result.failed.append(criterion.name)

        results.append(result)

        print(f"\n{tc['ticket_id']} ({tc['priority']}) — score: {result.score:.0%}")
        if result.passed:
            print(f"  pass: {', '.join(result.passed)}")
        if result.failed:
            print(f"  FAIL: {', '.join(result.failed)}")
        if result.skipped:
            print(f"  n/a:  {', '.join(result.skipped)}")

    overall = sum(r.score for r in results) / len(results) if results else 0.0

    print("\n" + "=" * 55)
    print(f"Overall: {overall:.0%} across {len(results)} test case(s)")

    summary = {
        "overall_score": round(overall, 4),
        "test_cases": len(results),
        "results": [
            {
                "ticket_id": r.ticket_id,
                "score": round(r.score, 4),
                "passed": r.passed,
                "failed": r.failed,
                "skipped": r.skipped,
                "error": r.error,
            }
            for r in results
        ],
    }

    _compare_to_baseline(overall)

    BASELINE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Baseline written to {BASELINE_PATH.relative_to(Path.cwd())}")

    return summary


def _compare_to_baseline(overall: float) -> None:
    """Report movement against the previous run, if there is one."""
    if not BASELINE_PATH.exists():
        print("No previous baseline — this run establishes it.")
        return

    try:
        previous = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["overall_score"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Could not read previous baseline ({e}) — overwriting.")
        return

    delta = overall - previous
    if abs(delta) < 0.001:
        print(f"No change vs baseline ({previous:.0%}).")
    elif delta > 0:
        print(f"IMPROVED: {previous:.0%} -> {overall:.0%} (+{delta:.0%})")
    else:
        print(f"REGRESSED: {previous:.0%} -> {overall:.0%} ({delta:.0%})")


if __name__ == "__main__":
    asyncio.run(run_evals())
