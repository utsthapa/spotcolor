FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install dependencies
COPY deploy/requirements-lambda.txt .
RUN pip install -r requirements-lambda.txt --target "${LAMBDA_TASK_ROOT}"

# Copy application code
COPY api/ ${LAMBDA_TASK_ROOT}/api/
COPY screenprint/ ${LAMBDA_TASK_ROOT}/screenprint/

# Set the Lambda handler
CMD ["api.lambda_handler.handler"]
