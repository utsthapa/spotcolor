# Simpler Dockerfile for web deployment (not Lambda)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (use a simpler version for web)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ /app/api/
COPY screenprint/ /app/screenprint/

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
