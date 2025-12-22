#!/bin/bash
# SpotColor API Deployment Script for AWS Lambda
#
# Prerequisites:
#   - AWS CLI configured with credentials
#   - AWS SAM CLI installed (pip install aws-sam-cli)
#   - Docker (for SAM build)
#
# Usage:
#   ./deploy.sh [dev|staging|prod]

set -e

STAGE=${1:-prod}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================"
echo "SpotColor API Deployment"
echo "Stage: $STAGE"
echo "======================================"

cd "$PROJECT_DIR"

# Build with SAM
echo ""
echo "Building Lambda package..."
python -m samcli build -t deploy/template.yaml --use-container

# Deploy
echo ""
echo "Deploying to AWS..."
python -m samcli deploy \
    --template-file .aws-sam/build/template.yaml \
    --stack-name "spotcolor-api-$STAGE" \
    --parameter-overrides "Stage=$STAGE" \
    --capabilities CAPABILITY_IAM \
    --resolve-s3 \
    --no-confirm-changeset

echo ""
echo "======================================"
echo "Deployment complete!"
echo ""
echo "Get your API URL with:"
echo "  aws cloudformation describe-stacks --stack-name spotcolor-api-$STAGE --query 'Stacks[0].Outputs'"
echo "======================================"
