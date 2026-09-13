"""AWS Lambda entrypoint — serves the FastAPI app behind a Function URL.

Mangum translates Lambda's event/context into the ASGI calls FastAPI expects,
so app.main stays a plain ASGI app and nothing else in the codebase changes.

The Groq key is pulled from SSM Parameter Store at cold start rather than
being set as a Lambda environment variable, so the secret never appears in
the function configuration. The fetch must happen before app.main is
imported, because config.Settings reads the environment at import time.
"""

import os

import boto3
from mangum import Mangum

GROQ_KEY_PARAM = os.getenv("GROQ_KEY_PARAM", "/sfr/GROQ_API_KEY")

# Cold start only — Lambda reuses the module across warm invocations.
if not os.getenv("GROQ_API_KEY"):
    _ssm = boto3.client("ssm")
    os.environ["GROQ_API_KEY"] = _ssm.get_parameter(
        Name=GROQ_KEY_PARAM, WithDecryption=True
    )["Parameter"]["Value"]

from app.main import app  # noqa: E402 — imported after the key is in the environment

# lifespan="auto" runs the app's startup/shutdown hooks on cold start,
# which is where the LangSmith tracer gets configured.
handler = Mangum(app, lifespan="auto")
