#!/bin/bash
# Simple Lambda deployment without SAM (using ZIP upload)
#
# This is a simpler approach that creates a Lambda function directly
# without the SAM framework.
#
# Prerequisites:
#   - AWS CLI configured with credentials
#   - Python 3.11 installed
#
# Usage:
#   ./deploy-simple.sh [function-name]

set -e

FUNCTION_NAME=${1:-spotcolor-api}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/lambda_build"

echo "======================================"
echo "SpotColor API - Simple Lambda Deploy"
echo "Function: $FUNCTION_NAME"
echo "======================================"

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements-lambda.txt" -t "$BUILD_DIR" --quiet

# Copy application code
echo "Copying application code..."
cp -r "$PROJECT_DIR/api" "$BUILD_DIR/"
cp -r "$PROJECT_DIR/screenprint" "$BUILD_DIR/"
cp -r "$PROJECT_DIR/frontend" "$BUILD_DIR/"

# Create ZIP
echo "Creating deployment package..."
cd "$BUILD_DIR"
zip -r -q ../lambda_package.zip .

echo ""
echo "Package created: lambda_package.zip"
echo "Size: $(du -h "$PROJECT_DIR/lambda_package.zip" | cut -f1)"

# Check if function exists
if aws lambda get-function --function-name "$FUNCTION_NAME" 2>/dev/null; then
    echo ""
    echo "Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://$PROJECT_DIR/lambda_package.zip"
else
    echo ""
    echo "Lambda function does not exist. Create it with:"
    echo ""
    echo "  aws lambda create-function \\"
    echo "    --function-name $FUNCTION_NAME \\"
    echo "    --runtime python3.11 \\"
    echo "    --handler api.lambda_handler.handler \\"
    echo "    --role arn:aws:iam::YOUR_ACCOUNT:role/YOUR_LAMBDA_ROLE \\"
    echo "    --zip-file fileb://lambda_package.zip \\"
    echo "    --timeout 30 \\"
    echo "    --memory-size 1024 \\"
    echo "    --environment Variables={LAMBDA_HANDLER=true}"
fi

# Cleanup
rm -rf "$BUILD_DIR"

echo ""
echo "======================================"
echo "Done!"
echo "======================================"
