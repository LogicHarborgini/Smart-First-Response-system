# Smart-First-Response-system

> LLM application that automatically generates the first customer response
> for enterprise support tickets using LangChain and Amazon Bedrock.

## Problem Statement

Support engineers at enterprise companies spend 5–10 minutes drafting the initial
response for every new ticket. This time compounds across hundreds of daily tickets.

**Smart First Response System** eliminates this using LangChain and Amazon Bedrock
to automatically generate the first customer response from ticket content —
reducing initial response time from minutes to seconds.

**This is an LLM application, not a RAG system.** It generates responses from
the current ticket content using prompt engineering and LLM inference. It does
not retrieve from a knowledge base.

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
| Deployment | AWS Lambda / Docker | Serverless or containerised serving |

## Note

A reference implementation built to explore production patterns in support
automation. Contains no proprietary code or data — all example tickets are
synthetic.
