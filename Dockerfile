# Dockerfile for Bakery Bread Sales Prediction API
# Build: docker build -t mp058-api:1.0 .
# Run: docker run -p 8000:8000 mp058-api:1.0

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY mlflow.db ./
COPY pyproject.toml .
COPY pytest.ini .

# Install the package in development mode
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]