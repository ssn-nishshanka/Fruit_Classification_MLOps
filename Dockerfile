# Use Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install pip packages with retries & timeout
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

# Copy application code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
