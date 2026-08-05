"""
LangSmith eval harness for SFR — golden dataset + LLM-as-judge.

Run from the project root:

    python -m evals.sfr_eval

Requires LANGSMITH_API_KEY and AWS credentials. Results appear in LangSmith
under the experiment prefix "SFR-v1", with per-criterion scores you can compare
across runs.

Why not LangChainStringEvaluator
--------------------------------
langsmith's LangChainStringEvaluator("criteria") defaults to ChatOpenAI as the
judge, so it fails without an OpenAI key even though this project never uses
OpenAI. The evaluators below are plain functions built on the same provider the
app uses — one less credential, one less bill, and no dependency on a wrapper
that has been churning across langsmith releases.

The judge follows LLM_PROVIDER, so it works on Bedrock, on Ollama, or against
the fake provider. Judge quality varies sharply between those: see JUDGE_MODEL
notes below.

The judge runs at temperature 0. A judge that disagrees with itself between runs
cannot detect a regression, which is the entire point of the harness.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.evaluation import aevaluate

from app.chain import (
    active_model_id,
    ainvoke_sfr_traced,
    build_chat_model,
    resolve_provider,
)
from app.config import settings

DATASET_NAME = "SFR Golden Dataset v1"
EXPERIMENT_PREFIX = "SFR-v1"


# ─────────────────────────────────────────────────────────────────────────────
# Golden dataset
# ─────────────────────────────────────────────────────────────────────────────

# Reference outputs are what a good support engineer would send — not what the
# model currently produces. Writing them from the model's output turns the eval
# into a tautology that passes forever.
GOLDEN_DATASET = [
    {
        "inputs": {
            "ticket_id": "EVAL-001",
            "priority": "P1",
            "customer_name": "Acme Corp",
            "content": (
                "EDI 850 purchase orders stopped processing at 02:00 UTC. "
                "47 orders backed up. Error code X12-834."
            ),
        },
        "outputs": {
            "response": (
                "Thank you for reporting this. We have received your P1 ticket "
                "regarding the EDI 850 processing failure with error X12-834 and "
                "the 47 backed-up orders, and our integration team is "
                "investigating with immediate priority. To speed up diagnosis, "
                "please send your trading partner ID and the timestamp of the "
                "last successful transmission. We will update you within one hour."
            )
        },
    },
    {
        "inputs": {
            "ticket_id": "EVAL-002",
            "priority": "P2",
            "customer_name": "TechFlow",
            "content": (
                "REST API endpoint returning 429 Too Many Requests since "
                "yesterday. We reduced call frequency but are still throttled."
            ),
        },
        "outputs": {
            "response": (
                "Thank you for getting in touch. We have received your report of "
                "429 throttling responses on your REST API integration and are "
                "reviewing your account's rate limit configuration. Could you "
                "confirm the affected endpoints and your current requests-per-"
                "second rate? We will come back to you with findings within four "
                "business hours."
            )
        },
    },
    {
        "inputs": {
            "ticket_id": "EVAL-003",
            "priority": "P3",
            "customer_name": "DataSync",
            "content": (
                "Please confirm the new SFTP server IP address so we can update "
                "our firewall whitelist during the next maintenance window."
            ),
        },
        "outputs": {
            "response": (
                "Thank you for planning ahead on this. We have received your "
                "request for the updated SFTP server IP address for your firewall "
                "whitelist. Our team is confirming the current details and will "
                "send them to you within two business days, well ahead of your "
                "maintenance window. Please let us know the window date if it is "
                "sooner than that."
            )
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict evaluator of enterprise support responses. You score one "
     "criterion at a time and you do not award credit for effort.\n\n"
     "Reply with JSON only, no prose and no code fences:\n"
     '{{"score": 0 or 1, "reason": "one sentence"}}'),
    ("human",
     "Criterion: {criterion_name}\n"
     "Definition: {criterion_definition}\n\n"
     "--- Support ticket ---\n"
     "Priority: {priority}\n"
     "Customer: {customer_name}\n"
     "Content: {content}\n\n"
     "--- Response to score ---\n"
     "{response}\n"
     "{reference_block}"),
])

CRITERIA = {
    "professionalism": (
        "The response acknowledges the specific issue described in the ticket "
        "(not a generic greeting), maintains a professional tone, and references "
        "concrete details from the ticket. Score 0 if it is boilerplate that "
        "would fit any ticket."
    ),
    "actionability": (
        "The response contains at least one concrete next step or a specific "
        "request for information that is relevant to this technical issue. "
        "Score 0 if it only promises to look into it."
    ),
    "priority_calibration": (
        "The urgency conveyed matches the ticket priority: P1 signals immediate "
        "escalation, P3 stays measured. Score 0 if a P3 is treated as an "
        "emergency or a P1 is treated as routine."
    ),
    "restraint": (
        "The response does not attempt to diagnose a root cause or prescribe a "
        "fix — a first response should acknowledge and gather information only. "
        "Score 0 if it asserts a cause or instructs the customer to apply a fix."
    ),
}

REFERENCE_CRITERIA = {
    "completeness_vs_reference": (
        "Compared to the reference response, the actual response covers the same "
        "essential elements: acknowledgement, a next step, and an expectation of "
        "timing. Wording may differ freely. Score 0 only if an essential element "
        "present in the reference is missing from the actual response."
    ),
}


DEFAULT_BEDROCK_JUDGE = "anthropic.claude-3-haiku-20240307-v1:0"

# Canned verdict for the fake provider. Valid JSON so _parse_verdict succeeds,
# which lets a full harness run — dataset creation, evaluator wiring, experiment
# upload — be exercised with no model and no credentials. Every score comes back
# 1, so a fake-provider run proves the pipeline works and says nothing whatever
# about response quality.
_FAKE_VERDICT = [
    '{"score": 1, "reason": "fake judge — pipeline check only, not a quality signal"}'
]


def judge_model_label() -> str:
    """Identifier for the judge actually in use, for experiment metadata."""
    provider = resolve_provider()
    if provider == "bedrock":
        return os.getenv("JUDGE_MODEL_ID", DEFAULT_BEDROCK_JUDGE)
    if provider == "ollama":
        return f"ollama:{os.getenv('JUDGE_OLLAMA_MODEL', settings.ollama_model)}"
    return "fake:canned-verdict"


@lru_cache(maxsize=1)
def _judge_chain():
    """
    Judge model, built once, on whichever provider LLM_PROVIDER resolves to.

    On Bedrock this defaults to Haiku rather than Sonnet: the judge runs once per
    criterion per example, making it the most-invoked model in the harness, and
    applying a one-line rubric does not need Sonnet.

    A caveat on Ollama. A small local model (llama3.2 is 3B) is a weak judge — it
    follows rubrics loosely and often breaks the JSON contract, which
    _parse_verdict scores 0 with an "unparseable" comment. Useful for checking
    the harness runs; not a scoreboard to trust. Point JUDGE_OLLAMA_MODEL at
    something larger if you intend to act on the numbers.
    """
    llm = build_chat_model(
        temperature=0,
        max_tokens=256,
        bedrock_model_id=os.getenv("JUDGE_MODEL_ID", DEFAULT_BEDROCK_JUDGE),
        ollama_model=os.getenv("JUDGE_OLLAMA_MODEL", settings.ollama_model),
        fake_responses=_FAKE_VERDICT,
        streaming=False,
    )
    return JUDGE_PROMPT | llm | StrOutputParser()


def _parse_verdict(raw: str) -> tuple[int, str]:
    """
    Pull {"score": …, "reason": …} out of the judge's reply.

    Judges wrap JSON in code fences or add a preamble often enough that a bare
    json.loads is not worth relying on. An unparseable verdict scores 0 and says
    so, rather than silently counting as a pass.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return 0, f"unparseable judge reply: {raw[:120]}"

    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0, f"invalid JSON from judge: {raw[:120]}"

    score = 1 if verdict.get("score") in (1, "1", True) else 0
    return score, str(verdict.get("reason", ""))[:300]


def _make_evaluator(key: str, definition: str, *, use_reference: bool = False):
    """Build a langsmith evaluator that scores one criterion via the judge."""

    async def _evaluator(run, example) -> dict:
        response = (run.outputs or {}).get("response", "")
        if not response:
            return {"key": key, "score": 0, "comment": "no response produced"}

        reference_block = ""
        if use_reference:
            reference = (example.outputs or {}).get("response", "")
            reference_block = f"\n--- Reference response ---\n{reference}\n"

        inputs = example.inputs or {}
        raw = await _judge_chain().ainvoke({
            "criterion_name": key,
            "criterion_definition": definition,
            "priority": inputs.get("priority", "unknown"),
            "customer_name": inputs.get("customer_name") or "unknown",
            "content": inputs.get("content", ""),
            "response": response,
            "reference_block": reference_block,
        })

        score, reason = _parse_verdict(raw)
        return {"key": key, "score": score, "comment": reason}

    _evaluator.__name__ = f"judge_{key}"
    return _evaluator


EVALUATORS = (
    [_make_evaluator(k, v) for k, v in CRITERIA.items()]
    + [_make_evaluator(k, v, use_reference=True) for k, v in REFERENCE_CRITERIA.items()]
)


# ─────────────────────────────────────────────────────────────────────────────
# Target + runner
# ─────────────────────────────────────────────────────────────────────────────


async def run_sfr(inputs: dict) -> dict:
    """Adapter called by langsmith with one example's inputs."""
    response, _run_id = await ainvoke_sfr_traced(
        ticket_id=inputs["ticket_id"],
        raw_content=inputs["content"],
        priority=inputs["priority"],
        customer_name=inputs.get("customer_name"),
    )
    return {"response": response}


def _ensure_dataset(client: Client):
    """Fetch the dataset, creating and populating it on first run."""
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Using existing dataset: {DATASET_NAME}")
        return dataset
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden examples for SFR first-response quality evaluation",
    )
    client.create_examples(
        inputs=[e["inputs"] for e in GOLDEN_DATASET],
        outputs=[e["outputs"] for e in GOLDEN_DATASET],
        dataset_id=dataset.id,
    )
    print(f"Created dataset with {len(GOLDEN_DATASET)} examples: {DATASET_NAME}")
    return dataset


async def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit(
            "LANGSMITH_API_KEY is not set — this harness needs it. "
            "For a local run with no API key, use: python -m evals.simple_eval"
        )

    client = Client()
    _ensure_dataset(client)

    print(f"\nRunning {len(EVALUATORS)} evaluators over {len(GOLDEN_DATASET)} examples...")

    results = await aevaluate(
        run_sfr,
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix=EXPERIMENT_PREFIX,
        metadata={
            "model": active_model_id(),
            "judge_model": judge_model_label(),
            "provider": resolve_provider(),
            "app_version": settings.app_version,
        },
        max_concurrency=2,
    )

    print("\n" + "=" * 55)
    print("SFR EVALUATION COMPLETE")
    print("=" * 55)
    print(f"Experiment: {results.experiment_name}")
    print("Full per-criterion scores: https://smith.langchain.com")


if __name__ == "__main__":
    asyncio.run(main())
