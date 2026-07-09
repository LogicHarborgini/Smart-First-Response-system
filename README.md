# Smart-First-Response-system

> LLM application that automatically generates the first customer response 
> for enterprise support tickets using LangChain and Amazon Bedrock.

## Problem Statement

Support engineers spend 5–10 minutes drafting the initial response for every 
new ticket. SFR eliminates this by generating the first response automatically 
from the ticket content and extracted logs.

## Architecture

*(Diagram coming Day 3)*

​```
[Support Ticket] → [LangChain PromptTemplate] → [Amazon Bedrock Claude] → [First Response]
​```

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