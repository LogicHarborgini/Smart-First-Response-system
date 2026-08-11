"""
Application configuration.

Uses pydantic-settings to load config from environment variables (.env file).

AWS credentials are deliberately absent: boto3 resolves them from the standard
credential chain — `aws configure` in development, the IAM execution role on
Lambda/ECS. Keeping keys out of app config means there is no field for a real
key to accidentally land in.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load .env into os.environ. pydantic-settings reads .env into Settings below,
# but it does not populate os.environ — and LangSmith tracing reads the
# LANGSMITH_* vars straight from os.environ. Without this, tracing stays off.
load_dotenv()


class Settings(BaseSettings):
    """Application settings — loaded from environment variables."""

    # AWS
    aws_default_region: str = Field(default="us-east-1")

    # Provider
    llm_provider: str = Field(
        default="auto",
        description=(
            "auto | bedrock | ollama | fake. 'auto' picks bedrock when boto3 can "
            "resolve credentials, otherwise ollama. 'fake' must be set explicitly: "
            "it returns canned text and exists only to exercise the tracing and "
            "eval plumbing without a model."
        ),
    )

    # Model
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Amazon Bedrock model ID"
    )
    bedrock_max_tokens: int = Field(default=512)
    bedrock_temperature: float = Field(default=0.3)
    ollama_model: str = Field(
        default="llama3.2",
        description="Local Ollama model, used when llm_provider resolves to ollama"
    )

    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model, used when llm_provider resolves to groq"
    )

    # API
    app_title: str = Field(default="Smart First Response API")
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO")

    # extra="ignore": .env also holds vars consumed elsewhere (LANGSMITH_*, read
    # from os.environ by the tracer). Without this, pydantic rejects them.
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton — import this everywhere
settings = Settings()
