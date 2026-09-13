# Smart-First-Response-system

![CI/CD](https://github.com/LogicHarborgini/Smart-First-Response-system/actions/workflows/ci-cd.yml/badge.svg)

> An LLM application that automatically generates an initial customer response
> for enterprise support tickets using LangChain and Amazon Bedrock.

## Live Demo

**Console:** https://h9ye12u7wg.execute-api.us-east-1.amazonaws.com/ui/

No setup is required. Select one of the example tickets, submit it, and view the
generated LLM response.

The console includes four tabs:

* The standard generation chain
* The triage agent, including the route selected and its reasoning
* A health probe
* Request and response schemas loaded from `/openapi.json`

The raw Swagger UI is also available at
[`/docs`](https://h9ye12u7wg.execute-api.us-east-1.amazonaws.com/docs) for direct
API interaction.

```bash
curl -X POST https://h9ye12u7wg.execute-api.us-east-1.amazonaws.com/api/v1/generate-response \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TKT-001",
    "content": "EDI 850 purchase orders stopped processing. 47 orders backed up since 14:30 UTC.",
    "priority": "P1",
    "customer_name": "Acme Corp"
  }'
```

To use the triage workflow, replace the path with
`/api/v1/generate-response/agent`. The [agent](#triage-agent) first evaluates the
ticket and determines whether to respond automatically, request clarification, or
escalate to a human.

The deployed demo currently uses Groq instead of Bedrock. The
[Deployment](#deployment) section explains the reasoning and how the provider is
selected without requiring code changes.

## Problem Statement

Enterprise support engineers may spend several minutes preparing the initial
response to a new ticket. When this happens across a high volume of daily
tickets, the time spent on these responses can add up significantly.

**Smart First Response System** is designed to reduce this initial response time
by using LangChain and a hosted LLM to generate a first response from the current
ticket context.

### LLM Application, Not RAG

This project is intentionally designed as an LLM generation application rather
than a RAG system. Responses are generated from the current ticket content
through prompt engineering and LLM inference; the application does not retrieve
information from a knowledge base.

The companion project,
[past-ticket-knowledge-rag](https://github.com/LogicHarborgini/past-ticket-knowledge-rag),
addresses the retrieval use case: answering questions such as *"How was this
issue resolved previously?"* by searching previously resolved tickets and
grounding the response in retrieved information.

Keeping the two use cases separate makes it easier to evaluate their different
failure modes. In this project, the primary concern is the quality of generated
responses, while the RAG system introduces additional retrieval-quality
considerations.

## Architecture

```text
                       SFR — Smart First Response
                     LLM Application (no retrieval)

    Support Engineer
            │
            │  1. New support ticket
            ▼
  ┌───────────────────┐      ┌───────────────────────┐      ┌─────────────────────┐
  │      FastAPI      │      │     LangChain LCEL    │      │   Amazon Bedrock    │
  │                   │      │                       │      │                     │
  │   POST /api/v1/   │─────▶   ChatPromptTemplate   ─────▶│   Claude 3 Sonnet   │
  │ generate-response │      │      ChatBedrock      │      │  (claude-3-sonnet-  │
  │                   │◀─────    StrOutputParser     ◀─────│   20240229-v1:0)    │
  │ Pydantic schemas  │      │                       │      │                     │
  └───────────────────┘      └───────────────────────┘      └─────────────────────┘
            │
            │  2. First response
            ▼
    Support Engineer
```

### Flow

1. A support engineer receives a new support ticket.
2. The ticket content is sent to the FastAPI endpoint and validated by Pydantic.
3. LangChain constructs the prompt using system context and ticket content.
4. `ChatBedrock` invokes Claude 3 Sonnet through Amazon Bedrock.
5. `StrOutputParser` extracts the generated response.
6. The first response is returned to the support engineer.

### Key Design Decisions

* **No retrieval (not RAG):** The response is generated using the current ticket
  context and the LLM.
* **LangChain LCEL:** Uses the `prompt | llm | parser` composition pattern.
* **Amazon Bedrock:** Provides managed model access without requiring
  application-managed GPU infrastructure.
* **Secure credential resolution:** AWS credentials are resolved through the
  standard boto3 credential chain rather than being stored in application
  configuration.
* **Reliability:** Transient provider failures are retried using exponential
  backoff and jitter.

## Core Implementation

The SFR chain is implemented using LangChain LCEL (LangChain Expression
Language):

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

# Build the chain
chain = prompt | llm | StrOutputParser()

# Async invocation for FastAPI
response = await chain.ainvoke({"ticket_content": ticket})

# Streaming invocation for real-time token delivery
async for token in chain.astream({"ticket_content": ticket}):
    yield token
```

### Tech Stack

| Component         | Technology                       | Purpose                                       |
| ----------------- | -------------------------------- | --------------------------------------------- |
| API Layer         | FastAPI (async)                  | Expose SFR as an HTTP service                 |
| LLM Orchestration | LangChain LCEL                   | Chain prompt → LLM → parser                   |
| LLM               | Amazon Bedrock (Claude 3 Sonnet) | Generate first responses                      |
| Validation        | Pydantic                         | Request/response schema enforcement           |
| Agent Routing     | LangGraph                        | Triage a ticket before generating a response  |
| Deployment        | Docker → AWS Lambda              | Container-based deployment behind API Gateway |

## Triage Agent

The standard endpoint generates a response for every ticket it receives. The
triage endpoint introduces an additional decision step so that tickets requiring
clarification or human involvement can be handled differently.

`POST /api/v1/generate-response/agent` evaluates the ticket first and routes it
based on the result:

```text
START ─→ triage ─┬─ auto_respond ─────────→ respond  ─→ END
                 ├─ needs_clarification ──→ clarify  ─→ END
                 └─ escalate ─────────────→ escalate ─→ END
```

| Decision              | Populated Field       | Meaning                                                                                                                                                                                       |
| --------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `auto_respond`        | `first_response`      | The ticket contains enough information for an initial response.                                                                                                                               |
| `needs_clarification` | `clarifying_question` | Additional information is needed before providing a meaningful response.                                                                                                                      |
| `escalate`            | `escalation_summary`  | The ticket should be reviewed by a human, such as in cases involving data loss, a suspected security issue, legal concerns, or a high-priority incident where automation may be inappropriate. |

For safety and reliability, triage failures such as invalid JSON, unknown
decision values, or unavailable providers default to `escalate` with confidence
`0.0` and a reasoning message describing the failure.

The response also includes `node_path`, for example `["triage", "escalate"]`,
making the selected route easy to review without requiring access to LangSmith.

Both endpoints remain available because they serve different use cases. The
standard endpoint requires a single model call, while the triage workflow adds a
classification step that can determine when automation should not proceed.

For additional design details, including the reasoning behind the temperature
setting and the absence of a clarification loop, see
[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md).

## Deployment

|                    | Local Development               | Demo                              | Production                    |
| ------------------ | ------------------------------- | --------------------------------- | ----------------------------- |
| LLM Provider       | Ollama (`llama3.2`)             | Groq (`openai/gpt-oss-20b`)       | AWS Bedrock (Claude 3 Sonnet) |
| Provider Selection | `auto` — boto3 credential probe | `LLM_PROVIDER=groq`               | AWS credentials               |
| Runtime            | uvicorn                         | AWS Lambda (Mangum) + API Gateway | ECS / Kubernetes              |

The same codebase supports multiple deployment environments through
environment-based provider selection.

Provider resolution is handled by `resolve_provider()`. An explicit
`LLM_PROVIDER` value takes precedence, while `auto` uses boto3's credential chain
to determine whether AWS credentials are available. The credential chain is
probed directly rather than checking `AWS_*` environment variables, because
`aws configure` writes to `~/.aws/credentials` without setting environment
variables.

The deployed demo explicitly sets `LLM_PROVIDER=groq`. This avoids automatically
selecting Bedrock through the Lambda execution role when the demo environment is
not configured with Bedrock permissions.

### Secret Management

The Groq API key is not stored directly in the function configuration.
`Dockerfile.lambda` uses the AWS Lambda Python base image, while
`app/lambda_handler.py` retrieves the key from AWS Systems Manager Parameter
Store (`/sfr/GROQ_API_KEY`, a `SecureString`).

The parameter is loaded during cold start before `app.main` is imported, ensuring
that the application settings are available when the module initializes.

This also allows the key to be rotated through Parameter Store without requiring
an image rebuild or storing the secret directly in the deployment configuration.

### Fake Provider

A fourth provider, `fake`, returns deterministic responses without making a model
call.

It is primarily intended for automated tests, tracing, and evaluation workflows
where external network access, API credentials, latency, and provider rate limits
are undesirable.

### Container

The production-oriented container uses a multi-stage build with a final image
size of approximately 563 MB.

Build tooling such as `build-essential` is kept in the builder stage and is not
included in the runtime image.

The container runs as a non-root user (`appuser`, UID 1000) and defines a
`HEALTHCHECK` so the platform can verify application readiness.

The `CMD` uses `exec uvicorn`, allowing uvicorn to run as PID 1 and receive
termination signals directly.

### Lambda and Local Images

Two Dockerfiles are maintained because Lambda and standard container environments
use different execution models:

* `Dockerfile` is used for local development, Docker Compose, and standard
  container platforms.
* `Dockerfile.lambda` is based on the AWS Lambda Python image and uses the Lambda
  Runtime Interface Client with the Mangum handler.

The application code under `app/` remains shared between both images.

### Local Stack

```bash
docker compose up --build
docker compose down
```

The application is exposed on port `8000`, with a health check running every
30 seconds.

### Redeploy to Lambda

```bash
ECR=<account>.dkr.ecr.us-east-1.amazonaws.com/sfr-lambda

docker buildx build --platform linux/amd64 \
  --provenance=false --sbom=false \
  --output type=image,oci-mediatypes=false,push=true \
  -f Dockerfile.lambda -t $ECR:latest .

aws lambda update-function-code --function-name sfr-api \
  --image-uri $ECR:latest --region us-east-1
```

The BuildKit flags above are required for the Lambda image workflow because
provenance and SBOM attestations can result in an OCI image index that is not
accepted by the Lambda container image deployment path.

## CI/CD

```text
ruff check  →  pytest (80% coverage gate)  →  Docker build verification
```

The `main` branch is protected so that linting, tests, and Docker build
verification must pass before a pull request can be merged.

Lambda deployment remains a deliberate manual step, allowing the live demo to be
updated intentionally rather than after every merge.

CI currently validates `Dockerfile`. Because `Dockerfile.lambda` is built
separately, Lambda-specific Docker issues may only become apparent during the
deployment workflow.

Tests use the `fake` provider, allowing CI to run without API keys, external
network requests, or third-party model dependencies.

Tooling versions are pinned to reduce build variability, and CI uses Python 3.12
to match the container base image.

## Reliability

Model providers can return transient errors such as throttling during periods of
increased demand. To improve resilience, the model invocation is configured with
three attempts using exponential backoff and jitter:

```python
chain = prompt | llm.with_retry(
    retry_if_exception_type=(ClientError, BotoConnectionError),
    wait_exponential_jitter=True,
    stop_after_attempt=3,
) | parser
```

### Retry Strategy

Two principles guide the retry configuration:

* **Only the model invocation is retried.** Parser failures do not trigger
  another model request.
* **Only transient failures are retried.** Configuration and programming errors
  should fail directly rather than being retried repeatedly.

Retryable exception types are provider-specific. Bedrock uses botocore
exceptions, while Ollama uses connection and timeout-related failures. The `fake`
provider does not require retries.

Jitter helps reduce synchronized retry bursts when multiple requests are
throttled at the same time.

One known limitation is that botocore uses `ClientError` for multiple AWS failure
conditions. Because the retry policy is based on exception type rather than the
specific AWS error code, certain non-transient configuration issues, such as
`AccessDenied`, may also be attempted three times before the underlying error is
surfaced.

The evaluation judge follows the same reliability principles. A throttled judge
invocation results in an evaluator error rather than being interpreted as a
failed score.

## Observability

Every chain execution can be traced through LangSmith.

Copy `.env.example` to `.env` and configure:

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_your_key_here
LANGSMITH_PROJECT=sfr-support-assistant
```

Tracing can be disabled by setting:

```text
LANGSMITH_TRACING=false
```

The application continues to operate normally, with `langsmith_run_id` returned
as `null`.

### Trace Metadata

Each trace contains:

| Field                        | Purpose                                                          |
| ---------------------------- | ---------------------------------------------------------------- |
| `run_name = SFR-<ticket_id>` | Provides an easily recognizable trace name                       |
| Metadata                     | Ticket ID, priority, customer, model ID, and application version |
| Tags                         | Filters such as `priority:P1` and `sfr`                          |
| `preprocess-ticket` span     | Separates preprocessing time from model latency                  |

The API also returns the LangSmith trace ID as `langsmith_run_id`, making it
possible to correlate an application request with the model execution that
generated the response.

Generate sample traces with:

```bash
python run_sfr_traces.py
```

Additional findings and measured observations are documented in
[OBSERVABILITY_NOTES.md](OBSERVABILITY_NOTES.md).

## Evaluation

Two evaluation harnesses are available from the project root:

```bash
python -m evals.simple_eval
python -m evals.sfr_eval
```

### Deterministic Evaluation

`simple_eval` provides a lightweight regression check without requiring an API
key or judge model.

It evaluates responses against deterministic criteria and writes results to:

```text
evals/baseline_results.json
```

This can be rerun after prompt or model changes to identify changes in the
evaluation score.

Criteria are applied according to ticket context. For example, urgency language
is expected for P1 tickets but is not required for lower-priority tickets.

### LLM-as-Judge Evaluation

`sfr_eval` is used for evaluation criteria that are difficult to capture with
deterministic rules, such as:

* Whether the response is specific to the ticket
* Whether the response reflects the ticket priority appropriately
* Whether the response avoids prematurely diagnosing the underlying issue

The judge runs at temperature `0` and uses the provider selected through
`LLM_PROVIDER`.

| Provider  | Judge                              | Interpretation                                                                         |
| --------- | ---------------------------------- | -------------------------------------------------------------------------------------- |
| `bedrock` | Claude 3 Haiku (`JUDGE_MODEL_ID`)  | Intended for higher-confidence evaluation                                              |
| `ollama`  | Local model (`JUDGE_OLLAMA_MODEL`) | Useful for local experimentation; results should be treated as indicative              |
| `fake`    | Canned verdict                     | Validates that the evaluation workflow executes, but does not measure response quality |

## Project Note

This repository is a reference implementation created to explore
production-oriented patterns for LLM-based support automation.

It contains no proprietary code or customer data. All example tickets and
supporting data are synthetic.
