"""
Application configuration.

Uses pydantic-settings to load config from environment variables (.env file).
In production: these come from AWS Secrets Manager or ECS task definition.
In development: they come from the .env file (never committed to git).
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings — loaded from environment variables."""
    
    # AWS
    aws_access_key_id: str = Field(default="", description="AWS access key")
    aws_secret_access_key: str = Field(default="", description="AWS secret key")
    aws_default_region: str = Field(default="us-east-1")
    
    # Model
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Amazon Bedrock model ID"
    )
    bedrock_max_tokens: int = Field(default=512)
    bedrock_temperature: float = Field(default=0.3)
    
    # API
    app_title: str = Field(default="Smart First Response API")
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO")
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import this everywhere
settings = Settings()
