# Smart-First-Response-system

> LLM application that automatically generates the first customer response 
> for enterprise support tickets using LangChain and Amazon Bedrock.

## Problem Statement

Support engineers spend 5–10 minutes drafting the initial response for every 
new ticket. SFR eliminates this by generating the first response automatically 
from the ticket content and extracted logs.

## Architecture

                        SFR — Smart First Response
                        LLM Application (Not RAG)

   Support Engineer                                    Amazon Bedrock
       │                                                    │
       │  New Support Ticket                                │
       ▼                                                    │
 ┌─────────────┐    ┌──────────────────┐    ┌──────────────┴──────────┐
 │  FastAPI    │───▶│  LangChain LCEL  │───▶│  Claude 3 Sonnet        │
 │  POST /api  │    │                  │    │  (claude-3-sonnet-       │
 │  /generate  │    │  Prompt Template │    │   20240229-v1:0)         │
 └─────────────┘    │       +          │    └──────────────┬──────────┘
                    │  ChatBedrock     │                   │
                    │       +          │    Generated      │
                    │  StrOutputParser │◀──  Response  ────┘
                    └──────────────────┘
                             │
                             ▼
                    First Response Delivered
                    to Support Engineer

Flow:
  1. Engineer receives new support ticket
  2. Ticket content sent to FastAPI endpoint
  3. LangChain formats prompt: system context + ticket content
  4. ChatBedrock invokes Claude 3 Sonnet on Amazon Bedrock
  5. StrOutputParser extracts response text
  6. First response returned to engineer

Key Design Decisions:
  - No retrieval (not RAG): response generated purely from ticket context + LLM knowledge
  - LangChain LCEL pipe syntax: prompt | llm | parser
  - Amazon Bedrock: managed LLM service, no GPU infrastructure to maintain

## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM Orchestration | LangChain (LCEL) |
| LLM Provider | Amazon Bedrock (Claude) |
| API Layer | FastAPI *(coming Week 2)* |
| Containerisation | Docker *(coming Week 2)* |
| Language | Python 3.11 |

## Project Status

🔨 In active development — Week 1

## Note

This is an open-source portfolio version. The production system runs at Cleo with enterprise-specific integrations.

## Problem Statement

Support engineers at enterprise companies spend 5–10 minutes drafting the initial 
response for every new ticket. This time compounds across hundreds of daily tickets.

**Smart First Response System** eliminates this using LangChain and Amazon Bedrock 
to automatically generate the first customer response from ticket content — 
reducing initial response time from minutes to seconds.

**This is an LLM application, not a RAG system.** It generates responses from 
the current ticket content using prompt engineering and LLM inference. It does 
not retrieve from a knowledge base.