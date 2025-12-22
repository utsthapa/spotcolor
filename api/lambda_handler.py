"""AWS Lambda handler for SpotColor API.

Uses Mangum to adapt FastAPI for AWS Lambda + API Gateway.

Deployment options:
1. Lambda + API Gateway (REST API)
2. Lambda + API Gateway (HTTP API) - recommended, lower cost
3. Lambda Function URL - simplest setup

Configuration:
- Set LAMBDA_HANDLER=true environment variable
- Memory: 512MB minimum (1024MB recommended for faster processing)
- Timeout: 30 seconds
- Runtime: Python 3.10+
"""

import os

# Only import mangum in Lambda environment to avoid unnecessary dependency locally
if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("LAMBDA_HANDLER"):
    from mangum import Mangum
    from .main import app

    # Create the Lambda handler
    # Configure for API Gateway HTTP API (v2) or REST API (v1)
    handler = Mangum(
        app,
        lifespan="off",  # Disable lifespan events in Lambda
        api_gateway_base_path=None  # Set if using custom domain with base path
    )

else:
    # Local development - export app directly
    from .main import app
    handler = None
