# Smart-First-Response-system

![CI/CD](https://github.com/LogicHarborgini/Smart-First-Response-system/actions/workflows/ci-cd.yml/badge.svg)

> LLM application that automatically generates the first customer response
> for enterprise support tickets using LangChain and Amazon Bedrock.

## Live Demo

**Interactive API docs:**
https://vis-smart-first-response-system-production.up.railway.app/docs

No setup required — open the Swagger UI, click *Try it out*, and send a real
ticket through a live LLM.

```bash
curl -X POST https://vis-smart-first-response-system-production.up.railway.app/api/v1/generate-response \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TKT-001",
    "content": "EDI 850 purchase orders stopped processing. 47 orders backed up since 14:30 UTC.",
    "priority": "P1",
    "customer_name": "Acme Corp"
  }'
```

The deployed demo runs on Groq rather than Bedrock — see [Deployment](#deployment)
for why, and for how the provider is selected without a code change.

## Problem Statement

Support engineers at enterprise companies spend 5–10 minutes drafting the initial
response for every new ticket. This time compounds across hundreds of daily tickets.

**Smart First Response System** eliminates this using LangChain and Amazon Bedrock
to automatically generate the first customer response from ticket content —
reducing initial response time from minutes to seconds.

**This is an LLM application, not a RAG system.** It generates responses from
the current ticket content using prompt engineering and LLM inference. It does
not retrieve from a knowledge base.

The companion project
[past-ticket-knowledge-rag](https://github.com/LogicHarborgini/past-ticket-knowledge-rag)
is the retrieval case: it answers "how was this fixed before?" by searching
resolved tickets and grounding its answer in what it finds. The two are
deliberately separate because the failure modes are not the same — a wrong first
response is a generation problem and nothing else, while a wrong retrieved answer
is either bad retrieval or bad generation, and telling those apart drives most of
the design in that repo.

## Architecture

```
                        SFR — Smart First Response
                        LLM Application (Not RAG)

   Support Engineer                                    Amazon Bedrock
       │                                                    │
       │  New Support Ticket                                │
       ▼                                                    │
 ┌─────────────┐    ┌──────────────────┐    ┌──────────────┴──────────┐
 │  FastAPI    │───▶│  LangChain LCEL  │───▶│  Claude 3 Sonnet        │
 │  POST /api  │    │                  │    │  (claude-3-sonnet-      │
 │  /generate  │    │  Prompt Template │    │   20240229-v1:0)        │
 └─────────────┘    │       +          │    └──────────────┬──────────┘
                    │  ChatBedrock     │                   │
                    │       +          │    Generated      │
                    │  StrOutputParser │◀──  Response  ────┘
                    └──────────────────┘
                             │
                             ▼
                    First Response Delivered
                    to Support Engineer
```

**Flow**

1. Engineer receives new support ticket
2. Ticket content sent to FastAPI endpoint
3. LangChain formats prompt: system context + ticket content
4. ChatBedrock invokes Claude 3 Sonnet on Amazon Bedrock
5. StrOutputParser extracts response text
6. First response returned to engineer

**Key Design Decisions**

- No retrieval (not RAG): response generated purely from ticket context + LLM knowledge
- LangChain LCEL pipe syntax: `prompt | llm | parser`
- Amazon Bedrock: managed LLM service, no GPU infrastructure to maintain
- Credentials resolved from the boto3 credential chain, never from app config
- Transient provider failures retried with exponential backoff and jitter (see Reliability)

## Core Implementation

The SFR chain is built with LangChain LCEL (LangChain Expression Language):

```python
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional support engineer writing first responses..."),
    ("human", "Support Ticket:\n{ticket_content}\n\nGenerate the first response:")
])

# Configure the LLM
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"max_tokens": 512, "temperature": 0.3},
    streaming=True
)

# Build the chain — the | operator wires the components together
chain = prompt | llm | StrOutputParser()

# Async invocation for FastAPI (non-blocking)
response = await chain.ainvoke({"ticket_content": ticket})

# Streaming invocation for real-time token delivery
async for token in chain.astream({"ticket_content": ticket}):
    yield token  # deliver each word as it arrives
```

**Tech Stack**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Layer | FastAPI (async) | Expose SFR as an HTTP service |
| LLM Orchestration | LangChain LCEL | Chain prompt → LLM → parser |
| LLM | Amazon Bedrock (Claude 3 Sonnet) | Generate first responses |
| Validation | Pydantic | Request/response schema enforcement |
| Deployment | Docker → Railway | Containerised serving, deployed from `main` |

## Deployment

| | Local dev | Demo (deployed) | Production |
|---|---|---|---|
| LLM provider | Ollama (`llama3.2`) | Groq (`llama-3.1-8b-instant`) | AWS Bedrock (Claude 3 Sonnet) |
| Selected by | `auto` — boto3 credential probe | `LLM_PROVIDER=groq` | AWS credentials present |
| Runtime | uvicorn | Railway (Docker) | ECS / K8s |

Same codebase, same Dockerfile, different environment variables. Provider
selection lives in `resolve_provider()`: an explicit `LLM_PROVIDER` always wins,
and `auto` probes for credentials through boto3 itself rather than checking
`AWS_*` environment variables — because `aws configure` writes to
`~/.aws/credentials` and sets no env vars, so an env-var check reports "no AWS"
on the most common local setup.

The deployed demo sets `LLM_PROVIDER=groq` explicitly rather than relying on
`auto`. With no credentials present, boto3 walks its full credential chain down
to the EC2 instance-metadata endpoint, which does not exist on Railway — so
`auto` both slows startup and silently selects a provider that cannot serve a
request. Groq itself is the right fit for a public demo: hosted inference on a
free tier, no AWS account, and no GPU to pay for.

A fourth provider, `fake`, returns canned responses with no model call at all.
It exists so the tracing and eval harnesses — and CI — can run with no network
access, no API cost, and no flakiness from a third-party model being slow or
rate-limited.

**Container:** multi-stage build, 563MB. Build tooling (`build-essential`) lives
only in the builder stage and never reaches the runtime image. The container runs
as a non-root user (`appuser`, uid 1000) and declares a `HEALTHCHECK` so the
platform routes traffic only once the app actually responds. `CMD` uses
`exec uvicorn` so uvicorn becomes PID 1 and receives `SIGTERM` directly —
without it the shell swallows the signal and in-flight requests are severed on
every redeploy.

Local stack:

```bash
docker compose up --build      # service on :8000, health check every 30s
docker compose down
```

## CI/CD

```
ruff check  →  pytest (80% coverage gate)  →  Docker build verification
```

Railway deploys on push to `main`. GitHub Actions gates the code; Railway ships
it — one deploy trigger, not two.

Tests run against the `fake` provider, so CI needs no API keys and makes no
network calls. Both `ruff` and its rule set are pinned (`ruff.toml`,
`RUFF_VERSION`) for the same reason `requirements.txt` carries version ceilings:
a new release of a tool must not be able to fail the build on its own. CI runs
Python 3.12 to match the container's base image rather than whatever version is
installed locally.

## Reliability

Bedrock returns `ThrottlingException` under load. Without a retry that is a 503
to the support engineer, at exactly the moment ticket volume is highest — so the
model step is wrapped in three attempts with exponential backoff and jitter:

```python
chain = prompt | llm.with_retry(
    retry_if_exception_type=(ClientError, BotoConnectionError),
    wait_exponential_jitter=True,   # 1s, 2s + random offset
    stop_after_attempt=3,
) | parser
```

Two deliberate choices:

- **Only the model step is wrapped.** A parser failure is not worth another call
  to Bedrock.
- **Only transient exception types are retried.** A `KeyError` from a missing
  prompt variable fails identically three times, so a blanket retry buys nothing
  but a slower error and a hidden bug. The retryable set is chosen per provider:
  botocore errors on Bedrock, connection and timeout errors on Ollama, none on
  `fake`.

Jitter is what makes this safe under concurrency. Without it every request
throttled in the same second retries in the same second, reproducing the burst
that caused the throttle.

One limitation worth stating: botocore raises `ClientError` for throttling
(retryable) and for `AccessDenied` (not), and `with_retry` filters on exception
type with no hook for the error code. A misconfigured IAM role therefore costs
three attempts before the real error surfaces — the cheaper side of the trade,
since the alternative is not retrying throttles at all.

The eval judge in `evals/sfr_eval.py` uses the same policy. It fires once per
criterion per example, making it the most-throttled model in the project, and a
throttled judge does not score 0 — it errors the evaluator and leaves a hole in
the experiment.

## Observability

Every chain execution is traced via LangSmith. Copy `.env.example` to `.env` and
set:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_your_key_here
LANGSMITH_PROJECT=sfr-support-assistant
```

Set `LANGSMITH_TRACING=false` to disable tracing without removing the key. The
app runs identically either way — `langsmith_run_id` in the response is simply
`null`.

Each trace carries:

| Field | Purpose |
|-------|---------|
| `run_name` = `SFR-<ticket_id>` | Identifies the trace at a glance instead of `RunnableSequence` |
| metadata: ticket ID, priority, customer, model ID, app version | Filter and group traces |
| tags: `priority:P1`, `sfr` | Saved views for critical tickets |
| `preprocess-ticket` span | Separates preprocessing cost from model latency |

The API returns the trace ID as `langsmith_run_id`, so a ticket in your own
records can be matched back to the exact model call that produced its response.

Generate sample traces:

```bash
python run_sfr_traces.py
```

See [OBSERVABILITY_NOTES.md](OBSERVABILITY_NOTES.md) for what the traces showed —
measured offline where that is possible, and left as explicit questions where it
needs a real provider.

## Evaluation

Two harnesses, run from the project root:

```bash
python -m evals.simple_eval    # deterministic checks, no API key or judge model
python -m evals.sfr_eval       # LangSmith golden dataset + LLM-as-judge
```

`simple_eval` is the regression gate. It scores responses against deterministic
criteria and writes `evals/baseline_results.json`; rerun it after any prompt or
model change and it reports whether the score moved. Criteria that only apply to
some tickets are skipped rather than failed — urgency language is required on P1
and not expected on P3.

`sfr_eval` handles the judgement calls deterministic checks cannot: whether a
response is genuinely specific to the ticket, whether its urgency matches the
priority, and whether it resists diagnosing the problem in a first response.

The judge runs at temperature 0 on whichever provider `LLM_PROVIDER` resolves to
— never the OpenAI default that `LangChainStringEvaluator` would pull in, so no
OpenAI credentials are needed. Judge strength varies by provider, and so does how
much the scores are worth:

| Provider | Judge | Scores mean |
|---|---|---|
| `bedrock` | Claude 3 Haiku (`JUDGE_MODEL_ID`) | Trustworthy — use as the quality gate |
| `ollama` | Local model (`JUDGE_OLLAMA_MODEL`) | Indicative only; a 3B model follows rubrics loosely and often breaks the JSON contract |
| `fake` | Canned verdict | Nothing — proves the harness runs, no more |

## Note

A reference implementation built to explore production patterns in support
automation. Contains no proprietary code or data — all example tickets are
synthetic.
